"""Live fan-out: one operator's hands, replayed on many robots whose scenes all differ.

Replaying joint angles only works if every scene is identical. Here each follower robot gets
the same objects at other places, turned differently, sometimes in another shape. So the
operator's motion is not copied, it is re-expressed: a follower's hand goes where the
operator's hand is *relative to the things on the table*.

Three ideas make that work while the operator is still moving:

1. Scene warp. Every object on the lead's mat is paired with one on the follower's mat. Each
   pair defines a small rigid map (shift + turn about the vertical) from the lead object's
   neighbourhood to the follower object's. Near an object the hand target follows that
   object's map completely; in between, the maps blend smoothly; far from everything (above
   the bins, which stand at the same place for everyone) the follower simply mirrors the
   lead. The turn is chosen among the grasps that are equivalent for the follower's shape,
   so a pinch across a cube's faces becomes a pinch across the faces of a cube turned by
   30 degrees, or a valid pinch on a triangle.
2. Live anchors. The maps use where the objects are *now*, in both scenes, and what the lead
   robot actually did (its measured hand pose), not what it was asked to do. If a follower's
   part slides while the fingers close, its hand goes after it.
3. Elastic time. Followers read the operator's motion from a short history, each at its own
   pace: one whose arm has further to travel falls behind instead of closing its fingers too
   early, and catches up when it can. Like echoes, they arrive a little later.

The anchors are live only while the hand is open. As soon as the fingers start to close on a
part the map for that hand is frozen (a hand must not chase a part it is holding), and once
the part is carried away the frozen offset fades out, so every follower converges onto the
operator's own path and releases above the same bin. Over the bins, which stand at the same
place for everyone, followers mirror the lead exactly.

All arrays are batched over F followers; nothing here touches the simulator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kaiku.sim.grasp import NATURAL_DIR, TRIANGLE_PHI, arm_for
from kaiku.sim.reach import ReachMap, level_palm, part_radius
from kaiku.sim.scene import SHAPES, Layout, Placement, Scene

CUBE, CYLINDER, TRIANGLE = 0, 1, 2
SHAPE_CODE = {"cube": CUBE, "cylinder": CYLINDER, "triangle": TRIANGLE}


def wrap(a, period: float = 2 * np.pi):
    return (np.asarray(a) + period / 2) % period - period / 2


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def quat_yaw(q: np.ndarray) -> np.ndarray:
    """Yaw about +z of wxyz quaternions (..., 4)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rot_z(a: np.ndarray) -> np.ndarray:
    """Rotation matrices about +z for angles (...,) -> (..., 3, 3)."""
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(np.shape(a) + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 1, 0], R[..., 1, 1], R[..., 2, 2] = c, -s, s, c, 1.0
    return R


def rotate_towards(R_from: np.ndarray, R_to: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Rotations (..., 3, 3) a fraction ``w`` (...) of the way from ``R_from`` to ``R_to``."""
    E = R_to @ np.swapaxes(R_from, -1, -2)
    v = 0.5 * np.stack([E[..., 2, 1] - E[..., 1, 2], E[..., 0, 2] - E[..., 2, 0], E[..., 1, 0] - E[..., 0, 1]], -1)
    s = np.linalg.norm(v, axis=-1)
    angle = np.arctan2(s, np.clip(0.5 * (np.trace(E, axis1=-2, axis2=-1) - 1.0), -1.0, 1.0))
    axis = v / np.maximum(s, 1e-9)[..., None]
    a = (w * angle)[..., None, None]
    K = np.zeros(E.shape)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -axis[..., 2], axis[..., 1], axis[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -axis[..., 0], -axis[..., 1], axis[..., 0]
    return (np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)) @ R_from


def nearest_grasp_dir(shape: np.ndarray, yaw: np.ndarray, prefer: np.ndarray) -> np.ndarray:
    """For each part (shape code, yaw), the finger direction that pinches it across its grasp
    width and is closest to ``prefer``. Cylinders take any direction, so they return ``prefer``."""
    cube = prefer + wrap(yaw - prefer, np.pi / 2)
    apex = yaw + np.pi / 2
    ta = prefer + wrap(apex + TRIANGLE_PHI - prefer, 2 * np.pi / 3)
    tb = prefer + wrap(apex - TRIANGLE_PHI - prefer, 2 * np.pi / 3)
    tri = np.where(np.abs(ta - prefer) <= np.abs(tb - prefer), ta, tb)
    return np.where(shape == CUBE, cube, np.where(shape == TRIANGLE, tri, prefer))


def alignment(shape: np.ndarray, yaw: np.ndarray, fingers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """How a hand whose fingers point along ``fingers`` is lined up with a part: the nearest
    valid grasp direction, and the part of the misalignment worth copying. Close to a valid
    grasp that is the misalignment itself; halfway between two valid grasps (where it is
    unclear which one is meant) it tapers to zero, so the result is continuous in ``fingers``."""
    q = np.pi / 2
    cube = wrap(yaw - fingers, q)[..., None]                                   # offset to the nearest valid
    cube = np.concatenate([cube, cube - np.sign(cube + 1e-12) * q], axis=-1)
    apex = yaw + np.pi / 2
    t = 2 * np.pi / 3
    da, db = wrap(apex + TRIANGLE_PHI - fingers, t)[..., None], wrap(apex - TRIANGLE_PHI - fingers, t)[..., None]
    tri = np.concatenate([da, db, da - np.sign(da + 1e-12) * t, db - np.sign(db + 1e-12) * t], axis=-1)

    def nearest_two(off):
        order = np.argsort(np.abs(off), axis=-1)
        first = np.take_along_axis(off, order[..., :1], axis=-1)[..., 0]
        second = np.abs(np.take_along_axis(off, order[..., 1:2], axis=-1)[..., 0])
        return first, second

    c1, c2 = nearest_two(cube)
    t1, t2 = nearest_two(tri)
    off = np.where(shape == CUBE, c1, np.where(shape == TRIANGLE, t1, 0.0))    # valid = fingers + off
    second = np.where(shape == CUBE, c2, np.where(shape == TRIANGLE, t2, np.pi))
    unclear = np.abs(off) / np.maximum(np.abs(off) + second, 1e-9)             # 0 aligned .. 0.5 halfway
    copy = -off * smoothstep((0.5 - unclear) / 0.2)
    return fingers + off, copy


def pinch_shift(shape: np.ndarray, yaw: np.ndarray, direction: np.ndarray, size: float) -> np.ndarray:
    """Offset (..., 2) from a part's centre to the midpoint of its extent along ``direction``:
    zero for cubes and cylinders, up to 7.5 mm for a triangle (its centroid is not midway)."""
    r = 2.0 * size / 3.0
    v = (yaw + np.pi / 2)[..., None] + np.arange(3) * 2 * np.pi / 3
    f = direction[..., None]
    mid = 0.5 * (r * np.cos(f - v).max(-1) - r * np.cos(f + np.pi - v).max(-1))
    mid = np.where(shape == TRIANGLE, mid, 0.0)
    return mid[..., None] * np.stack([np.cos(direction), np.sin(direction)], axis=-1)


@dataclass
class LeadTick:
    """What the lead contributes at one control tick."""

    pinch: np.ndarray      # (2, 3) measured pinch point of the lead robot, left and right, pelvis frame
    rot: np.ndarray        # (2, 3, 3) measured palm rotation
    hand: np.ndarray       # (14,) Dex3 command in the challenge's order
    obj_xy: np.ndarray     # (n, 2) centres of the lead's episode objects, measured
    obj_yaw: np.ndarray    # (n,)
    on_mat: np.ndarray     # (n,) bool: still on the mat, to be sorted
    held: np.ndarray       # (2,) which episode object each hand has lifted, -1 for none
    arm: np.ndarray | None = None   # (14,) measured arm joints: the posture followers lean towards


@dataclass
class EchoParams:
    attach_radius: float = 0.03      # m: within this of an object the hand follows that object fully
    reach_radius: float = 0.16       # m: beyond this an object has no influence
    height_factor: float = 0.3       # how much height above the object counts as distance
    swing_radius: float = 0.05       # m beyond attach_radius over which the turn about the object fades
    idw_power: float = 4.0           # sharpness of the blend between neighbouring objects
    level_turn: float = 0.6          # rad of turn at which a follower's palm near a part is fully level:
                                     # the lead's tilt suits the lead's direction, not one turned far from it
    centering: float = 0.7           # share of the lead's off-centre error a follower does not copy:
    center_radius: float = 0.05      # within this of a part its hand is drawn onto the part's own grasp centre
    bin_fade_start: float = 0.03     # m beyond the object band (towards the bins) where the warp starts to fade
    bin_fade_length: float = 0.06    # m over which it fades to zero: over the bins followers mirror the lead
    grip_on: float = 0.5             # finger closure (rad, 0.4 = open preshape) above which a hand is gripping
    grip_off: float = 0.3            # and below which it has let go
    carry_fade_start: float = 0.06   # m of travel after gripping before the frozen offset starts fading
    carry_fade_length: float = 0.12  # m over which it fades to zero
    err_ok: float = 0.020            # m of tracking error below which a follower reads at full speed
    err_wait: float = 0.060          # m at which it stops reading ahead
    err_ok_fine: float = 0.005       # the same while the lead's fingers are closing or opening:
    err_wait_fine: float = 0.010     # a follower only grasps or lets go once its hand is in place
    rot_weight: float = 0.05         # m of error per radian of palm misalignment
    fine_lookahead: int = 4          # ticks ahead in which finger motion counts as "about to grasp"
    fine_change: float = 0.15        # rad of finger motion in that window for fully strict tolerances
    catch_up: float = 1.6            # fastest reading speed when behind (1 = real time)
    patience: int = 60               # ticks a follower may stall before it goes on regardless


class Echo:
    def __init__(self, scene: Scene, n_followers: int, params: EchoParams | None = None,
                 reach: ReachMap | None = None, max_ticks: int = 30 * 600):
        self.scene = scene
        self.F = n_followers
        self.p = params or EchoParams()
        self.reach = reach or ReachMap(scene)
        self.max_ticks = max_ticks
        self.natural = np.array([NATURAL_DIR["left"], NATURAL_DIR["right"]])
        self.n = 0
        self.begin(Layout([]), [Layout([]) for _ in range(n_followers)])

    # ------------------------------------------------------------------ episode
    def begin(self, lead: Layout, followers: list[Layout]) -> None:
        """Start an episode. Follower layouts list their objects in the same order as the lead's;
        the k-th of each is the pair (same colour, so the same bin)."""
        sc, F = self.scene, self.F
        n = self.n = len(lead.placements)
        assert all(len(f.placements) == n for f in followers), "followers need one partner per lead object"
        self.lead_idx = np.array([sc.by_name[p.name].index for p in lead.placements], int)
        self.lead_shape = np.array([SHAPE_CODE[sc.by_name[p.name].shape] for p in lead.placements], int)
        self.foll_idx = np.array([[sc.by_name[p.name].index for p in f.placements] for f in followers], int).reshape(F, n)
        self.foll_shape = np.array([[SHAPE_CODE[sc.by_name[p.name].shape] for p in f.placements] for f in followers],
                                   int).reshape(F, n)
        T = self.max_ticks
        self.h_pinch = np.zeros((T, 2, 3))
        self.h_rot = np.zeros((T, 2, 3, 3))
        self.h_hand = np.zeros((T, 14))
        self.h_xy = np.zeros((T, n, 2))
        self.h_yaw = np.zeros((T, n))
        self.h_on = np.zeros((T, n), bool)
        self.h_held = np.full((T, 2), -1, int)
        self.h_arm = np.zeros((T, 14))
        self.len = 0
        self.tau = np.zeros(F)                       # each follower's read position in the history, in ticks
        # The grasp chosen for each (follower, hand, part): the equivalent one that arm reaches best.
        self.g_f = np.zeros((F, 2, n))
        if n and F:
            xy = np.array([[p.pos[:2] for p in f.placements] for f in followers]).reshape(F, n, 2)
            yaw = np.array([[p.yaw for p in f.placements] for f in followers]).reshape(F, n)
            rest = np.array([[j for j in range(n) if j != k] for k in range(n)], int).reshape(n, n - 1)
            others = xy[:, rest]                                                           # (F, n, n-1, 2)
            radius = part_radius(self.foll_shape, sc.size)[:, rest]
            for s in range(2):
                self.g_f[:, s] = self.reach.best(self.foll_shape, yaw, xy, s, others=others, radius=radius)[0]
        self.grip = np.full((F, 2), -1, int)         # part each hand has closed on, -1 for none
        self.a_cf = np.zeros((F, 2, 2))              # the map frozen when it closed: follower's grasp centre,
        self.a_cl = np.zeros((F, 2, 2))              # the lead's,
        self.a_turn = np.zeros((F, 2))               # the turn between them,
        self.a_from = np.zeros((F, 2, 3))            # and where the lead's hand was
        self.last_target = None                      # (F, 2, 3) pinch targets issued on the previous tick
        self.last_rot = None
        self.stall = np.zeros(F, int)
        self.rate = np.ones(F)
        self.finished = False

    def push(self, tick: LeadTick) -> None:
        k = self.len
        if k >= self.max_ticks:
            raise RuntimeError("episode longer than the echo history")
        self.h_pinch[k], self.h_rot[k], self.h_hand[k] = tick.pinch, tick.rot, tick.hand
        self.h_xy[k], self.h_yaw[k], self.h_on[k], self.h_held[k] = tick.obj_xy, tick.obj_yaw, tick.on_mat, tick.held
        if tick.arm is not None:
            self.h_arm[k] = tick.arm
        self.len = k + 1

    def finish(self) -> None:
        """The lead's episode is over; followers read the rest of the history and stop."""
        self.finished = True

    @property
    def lag(self) -> np.ndarray:
        """Seconds each follower is behind the lead (at 30 Hz)."""
        return (self.len - 1 - self.tau) / 30.0

    @property
    def lead_arm(self) -> np.ndarray:
        """(F, 14) the lead's arm joints at the moment each follower is replaying."""
        return self.h_arm[np.minimum(np.floor(self.tau).astype(int), max(self.len - 1, 0))]

    @property
    def done(self) -> np.ndarray:
        """(F,) the follower has replayed everything the lead did."""
        return self.finished & (self.tau >= self.len - 1 - 1e-6)

    # ------------------------------------------------------------------ per tick
    def targets(self, obj_xy: np.ndarray, obj_yaw: np.ndarray, obj_on_mat: np.ndarray,
                pinch_now: np.ndarray, rot_now: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Targets for every follower on this tick.

        ``obj_xy`` (F, n, 2), ``obj_yaw`` (F, n), ``obj_on_mat`` (F, n): each follower's own
        episode objects as measured now. ``pinch_now`` (F, 2, 3) and ``rot_now`` (F, 2, 3, 3):
        where its pinch points and palms are.
        Returns pinch targets (F, 2, 3), palm rotations (F, 2, 3, 3) and hand targets (F, 14).
        """
        p, F, n = self.p, self.F, self.n
        if self.len == 0:
            raise RuntimeError("push a lead tick first")
        self._advance(pinch_now, rot_now)
        i0 = np.floor(self.tau).astype(int)
        i1 = np.minimum(i0 + 1, self.len - 1)
        a = (self.tau - i0)[:, None]
        x = (1 - a[..., None]) * self.h_pinch[i0] + a[..., None] * self.h_pinch[i1]        # (F, 2, 3)
        R = self.h_rot[np.where(a[:, 0] < 0.5, i0, i1)]                                    # (F, 2, 3, 3)
        hand = (1 - a) * self.h_hand[i0] + a * self.h_hand[i1]
        if n == 0:
            self.last_target, self.last_rot = x, R
            return x, R, hand
        h = self.h_hand[i0]
        closure = np.stack([-0.5 * (h[:, 3] + h[:, 5]), 0.5 * (h[:, 10] + h[:, 12])], axis=1)   # (F, 2) rad
        band_end = self.scene.region[1][0]
        l_xy, l_yaw, l_on = self.h_xy[i0], self.h_yaw[i0], self.h_on[i0]                   # (F, n, ..)
        out_x, out_R = x.copy(), R.copy()
        idx = np.arange(F)
        for s in range(2):
            xs = x[:, s]                                                                   # (F, 3)
            f_dir = np.arctan2(R[:, s, 1, 0], R[:, s, 0, 0])                               # finger direction (F,)
            # The equivalent grasp on the partner object. The lead's grasp is the valid one nearest
            # its fingers; the follower's was chosen for its own arm and follows the part if it
            # turns. The follower copies how far the lead is from being lined up.
            g_l, copy = alignment(self.lead_shape[None], l_yaw, f_dir[:, None])            # (F, n)
            g_f = nearest_grasp_dir(self.foll_shape, obj_yaw, self.g_f[:, s])
            self.g_f[:, s] = g_f
            turn = wrap(g_f + (1.0 - p.centering) * copy - f_dir[:, None])
            c_l = l_xy + pinch_shift(self.lead_shape[None], l_yaw, g_l, self.scene.size)   # (F, n, 2)
            c_f = obj_xy + pinch_shift(self.foll_shape, obj_yaw, g_f, self.scene.size)
            # Influence of each pair on this hand: how attached the hand is to the mat's objects
            # at all (gain), and how that attachment is shared between them (inverse distance).
            rel = xs[:, None, :2] - c_l                                                    # (F, n, 2)
            flat = np.sqrt((rel ** 2).sum(-1))
            dz = np.maximum(xs[:, None, 2] - self.scene.rest_z, 0.0) * p.height_factor
            near = smoothstep(1.0 - (np.sqrt(flat ** 2 + dz ** 2) - p.attach_radius) / (p.reach_radius - p.attach_radius))
            usable = l_on & obj_on_mat
            near = np.where(usable, near, 0.0)
            share = np.where(usable, (flat + 1e-3) ** -p.idw_power, 0.0)
            share = share / np.maximum(share.sum(-1, keepdims=True), 1e-12)
            over_bins = smoothstep((xs[:, 0] - band_end - p.bin_fade_start) / p.bin_fade_length)   # (F,)
            gain = (1.0 - np.prod(1.0 - near, axis=-1)) * (1.0 - over_bins)
            wn = gain[:, None] * share                                                     # (F, n)
            # Close to a part the hand's place is turned with it; further away it is only shifted.
            d_xy = (wn[..., None] * self._map(c_f, c_l, turn, rel)).sum(1)                 # (F, 2)
            d_yaw = (wn * turn).sum(1)
            # Fingers closing on a part: freeze that part's map for this hand. Fingers open: back to live.
            top = np.argmax(wn, axis=1)
            start = (self.grip[:, s] < 0) & (closure[:, s] > p.grip_on) & (wn[idx, top] > 0.5)
            if start.any():
                r, k = idx[start], top[start]
                self.grip[r, s] = k
                self.a_cf[r, s], self.a_cl[r, s], self.a_turn[r, s] = c_f[r, k], c_l[r, k], turn[r, k]
                self.a_from[r, s] = xs[r]
            self.grip[closure[:, s] < p.grip_off, s] = -1
            gripping = self.grip[:, s] >= 0
            travel = np.linalg.norm(xs - self.a_from[:, s], axis=-1)
            keep = (1.0 - smoothstep((travel - p.carry_fade_start) / p.carry_fade_length)) * (1.0 - over_bins)
            frozen = self._map(self.a_cf[:, s], self.a_cl[:, s], self.a_turn[:, s], xs[:, :2] - self.a_cl[:, s])
            d_xy = np.where(gripping[:, None], keep[:, None] * frozen, d_xy)
            d_yaw = np.where(gripping, keep * self.a_turn[:, s], d_yaw)
            out_x[:, s, :2] += d_xy
            turned = rot_z(d_yaw) @ R[:, s]
            heading = np.arctan2(turned[:, 1, 0], turned[:, 0, 0])
            level = level_palm(np.stack([heading, heading], axis=-1))[:, s]
            attached = np.where(gripping, keep, gain)
            out_R[:, s] = rotate_towards(turned, level, attached * smoothstep(np.abs(d_yaw) / p.level_turn))
        self.last_target, self.last_rot = out_x, out_R
        return out_x, out_R, hand

    def _map(self, c_f: np.ndarray, c_l: np.ndarray, turn: np.ndarray, rel: np.ndarray) -> np.ndarray:
        """Where a point ``rel`` away from the lead's grasp centre goes, as an offset: the shift
        between the two grasp centres plus, close by, the turn about the centre; right at the
        part, most of the lead's own aiming error is left out."""
        p = self.p
        flat = np.sqrt((rel ** 2).sum(-1))
        swing = smoothstep(1.0 - (flat - p.attach_radius) / p.swing_radius)
        c, s = np.cos(turn), np.sin(turn)
        turned = np.stack([c * rel[..., 0] - s * rel[..., 1], s * rel[..., 0] + c * rel[..., 1]], axis=-1)
        placed = rel + swing[..., None] * (turned - rel)
        placed = placed * (1.0 - p.centering * smoothstep(1.0 - flat / p.center_radius))[..., None]
        return (c_f - c_l) + placed - rel

    def _advance(self, pinch_now: np.ndarray, rot_now: np.ndarray | None) -> None:
        """Move each follower's read position: full speed while it tracks well, slower as its
        hands fall behind their last targets, faster (up to ``catch_up``) to close a time gap.
        While the lead's fingers are moving the tolerance is tight, so a follower closes or
        opens its hand only once that hand is where it should be."""
        p = self.p
        end = self.len - 1
        if self.last_target is None:
            self.tau = np.minimum(self.tau, end)
            return
        err = np.linalg.norm(pinch_now - self.last_target, axis=-1)                        # (F, 2)
        if rot_now is not None:
            E = self.last_rot @ np.swapaxes(rot_now, -1, -2)
            ang = np.arccos(np.clip(0.5 * (np.trace(E, axis1=-2, axis2=-1) - 1.0), -1.0, 1.0))
            err = err + p.rot_weight * ang
        err = err.max(axis=1)
        i0 = np.floor(self.tau).astype(int)
        ahead = np.minimum(i0 + p.fine_lookahead, end)
        fine = np.clip(np.abs(self.h_hand[ahead] - self.h_hand[i0]).max(axis=1) / p.fine_change, 0.0, 1.0)
        ok = p.err_ok + fine * (p.err_ok_fine - p.err_ok)
        wait = p.err_wait + fine * (p.err_wait_fine - p.err_wait)
        rate = np.clip((wait - err) / (wait - ok), 0.0, 1.0)
        self.stall = np.where(rate < 0.3, self.stall + 1, 0)
        rate = np.where(self.stall > p.patience, 1.0, rate)
        behind = end - self.tau
        self.rate = rate * np.minimum(p.catch_up, 1.0 + np.maximum(behind - 1.0, 0.0) * 0.1)
        self.tau = np.minimum(self.tau + self.rate, end)


def sample_followers(scene: Scene, lead: Layout, rng: np.random.Generator, count: int, spread: float = 0.08,
                     vary_shape: bool = True, reach: ReachMap | None = None, rounds: int = 6) -> list[Layout]:
    """``count`` variations of the lead's layout. In each, every object keeps its colour (so
    its bin) and may get a new place within ``spread`` metres of the lead's, a new yaw in +-45
    degrees and, if ``vary_shape``, a new shape.

    Each variation starts as the lead's scene and is changed one part at a time; a change is
    kept only if the scene still follows the challenge's spacing rule and, with ``reach``,
    every part still has a grasp its arm reaches clear of the other parts. So a crowded mat
    simply varies less, and nothing is ever handed to a follower that it cannot do."""
    n = len(lead.placements)
    if n == 0 or count == 0:
        return [Layout([], dict(lead.mapping)) for _ in range(count)]
    lo, hi = scene.region
    L = scene.cfg["layout"]
    yaw_lo, yaw_hi = np.radians(L["yaw_range_deg"])
    obs = [scene.by_name[p.name] for p in lead.placements]
    colours = sorted({o.color for o in scene.objects})
    stock: dict[int, list[str]] = {}
    for o in scene.objects:
        stock.setdefault(colours.index(o.color) * 3 + SHAPE_CODE[o.shape], []).append(o.name)
    copies = np.array([len(stock.get(c, ())) for c in range(len(colours) * 3)])
    colour = np.array([colours.index(o.color) for o in obs])
    lead_xy = np.array([p.pos[:2] for p in lead.placements])
    sides = np.array([0 if arm_for(lead.mapping[o.color], p.pos[1]) == "left" else 1
                      for o, p in zip(obs, lead.placements)])
    K = count
    xy = np.tile(lead_xy, (K, 1, 1))
    yaw = np.tile([p.yaw for p in lead.placements], (K, 1))
    shape = np.tile([SHAPE_CODE[o.shape] for o in obs], (K, 1))
    good = reach.parts_ok(shape, yaw, xy, sides) if reach is not None else np.ones((K, n), bool)
    rows = np.arange(K)
    # What a proposal changes: (place within this share of ``spread``, yaw, shape).
    # Small changes first, so that a bolder one replaces them wherever it fits.
    kinds = [(0.0, True, True), (0.3, True, True), (0.6, True, True), (1.0, True, True), (1.0, False, False),
             (1.0, True, True)]
    for rnd in range(rounds):
        scale, new_yaw, new_shape = kinds[rnd % len(kinds)]
        for j in rng.permutation(n):
            x2, y2, s2 = xy.copy(), yaw.copy(), shape.copy()
            if scale:
                r, th = scale * spread * np.sqrt(rng.uniform(size=K)), rng.uniform(0.0, 2 * np.pi, size=K)
                x2[:, j] = lead_xy[j] + r[:, None] * np.stack([np.cos(th), np.sin(th)], axis=-1)
            if new_yaw:
                y2[:, j] = rng.uniform(yaw_lo, yaw_hi, size=K)
            if new_shape and vary_shape:
                s2[:, j] = rng.integers(3, size=K)
            code = colour * 3 + s2
            ok = ((x2[:, j] >= lo) & (x2[:, j] <= hi)).all(axis=1)
            gap = np.linalg.norm(x2 - x2[:, j:j + 1], axis=-1)
            gap[:, j] = np.inf
            ok &= (gap >= L["min_spacing"]).all(axis=1)
            ok &= ((code == code[:, j:j + 1]).sum(axis=1) <= copies[code[:, j]])
            g2 = good
            if reach is not None and ok.any():
                g2 = good.copy()
                g2[ok] = reach.parts_ok(s2[ok], y2[ok], x2[ok], sides)
                ok &= g2[:, j] & (g2 | ~good).all(axis=1)
            xy[ok], yaw[ok], shape[ok], good[ok] = x2[ok], y2[ok], s2[ok], g2[ok]
    out = []
    for k in rows:
        used: dict[int, int] = {}
        parts = []
        for j in range(n):
            c = int(colour[j] * 3 + shape[k, j])
            parts.append(Placement(stock[c][used.get(c, 0)], np.r_[xy[k, j], scene.rest_z], float(yaw[k, j])))
            used[c] = used.get(c, 0) + 1
        out.append(Layout(parts, dict(lead.mapping)))
    return out


def sample_follower(scene: Scene, lead: Layout, rng: np.random.Generator, spread: float = 0.08,
                    vary_shape: bool = True, reach: ReachMap | None = None) -> Layout:
    """One variation of the lead's layout; see :func:`sample_followers`."""
    return sample_followers(scene, lead, rng, 1, spread, vary_shape, reach)[0]
