"""The policy on its own: N robots, N scenes nobody has seen, no operator.

    python -m kaiku.policy.rollout policy.pt [--n 48] [--episodes 2] [--seed 100] [--parts 3]

Every robot gets a fresh scene by the challenge's rule and the policy's commands, through the
same arm IK and safety filter as an operator's. The verdict is the challenge's: an episode
counts when every part rests in its own bin and no command was clipped.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from kaiku.app.session import Session
from kaiku.fanout.echo import SHAPE_CODE
from kaiku.policy.features import HORIZON, MAX_PARTS, Featurizer, plan_for
from kaiku.policy.model import Policy
from kaiku.perception import CameraPerception, PartEstimate
from kaiku.sim.grasp import HAND_CLOSED, HAND_OPEN, hand14
from kaiku.sim.scene import BINS, Layout, SHAPE_SYMMETRY
from kaiku.sim.twin import CONTROL_HZ


# The policy answers with the next 16 ticks. Asked every 12 (0.4 s) it sorted 97 % of fresh scenes;
# asked every 4 the same policy sorted 73 %: asked that often it keeps correcting itself and
# never commits to the grasp.
EVERY = 12
BLEND = 4
_COLOR_BIN = {"red": 0, "yellow": 1, "blue": 2}
_SHAPE_CODE = {"cube": 0, "cylinder": 1, "triangle": 2}
_HAND_OPEN = hand14(HAND_OPEN, HAND_OPEN)
_HAND_CLOSED = hand14(HAND_CLOSED, HAND_CLOSED)
_HAND_DELTA = _HAND_CLOSED - _HAND_OPEN
_HAND_FINGER = np.flatnonzero(np.abs(_HAND_DELTA) > 0.1)


class Pilot:
    """Runs a policy on every env of a session, each with its own scene."""

    def __init__(self, session: Session, policy: Policy, every: int = EVERY, blend: int = BLEND,
                 perception: str | None = None, depth_noise_mm: float = 0.0, edge_noise_px: float = 0.0):
        self.s, self.policy = session, policy
        self.feat = Featurizer(session.scene)
        self.camera = perception == "camera"
        self.perception = (CameraPerception(session.twin, session.scene, depth_noise_mm=depth_noise_mm,
                                            edge_noise_px=edge_noise_px, seed=17)
                           if self.camera else None)
        self.every, self.blend = min(every, HORIZON), min(blend, HORIZON - min(every, HORIZON))
        lo, hi = session.twin.hand_range[:, 0], session.twin.hand_range[:, 1]
        self.hand_lo, self.hand_hi = lo, hi
        self.begin(session.layouts)

    def begin(self, layouts: list[Layout]) -> None:
        """Read each env's scene and make its plan: nearest part first, the arm on the bin's side."""
        sc, n, P = self.s.scene, self.s.n, MAX_PARTS
        self.idx = np.zeros((n, P), int)
        self.shape = np.zeros((n, P), int)
        self.bins = np.zeros((n, P), int)
        self.side = np.zeros((n, P), int)
        self.present = np.zeros((n, P), bool)
        if self.camera:
            # The initial plan is made from the first head-camera frame.  Layouts are
            # deliberately not consulted here: they are simulator bookkeeping only.
            self._camera_estimates = [[] for _ in range(n)]
            self.cam_pos = np.zeros((self.s.n, P, 3))
            self.cam_yaw = np.zeros((self.s.n, P))
            self.cam_shape = np.zeros((self.s.n, P), int)
            self.cam_bins = np.zeros((self.s.n, P), int)
            self.cam_present = np.zeros((self.s.n, P), bool)
            self.cam_done = np.zeros((self.s.n, P), bool)
            self.cam_carried = np.zeros((self.s.n, P), bool)
            self.cam_shape_votes = np.zeros((self.s.n, P, 3), int)
            self.cam_shape_locked = np.zeros((self.s.n, P), bool)
            for i in range(n):
                self._camera_estimates[i] = self.perception.observe(self.s.sim.data[i])
                self._set_camera_parts(i, self._camera_estimates[i])
                # A stationary head camera gives several independent depth-edge
                # samples before the first move.  Decide shape once from that vote.
                for _ in range(4):
                    self._set_camera_parts(i, self.perception.observe(self.s.sim.data[i]))
                self.cam_shape_locked[i, self.cam_present[i]] = True
                k = min(P, len(self._camera_estimates[i]))
                if k:
                    order, side = plan_for(self.cam_pos[i, :k], self.cam_bins[i, :k])
                    self.cam_pos[i, :k] = self.cam_pos[i, order]
                    self.cam_yaw[i, :k] = self.cam_yaw[i, order]
                    self.cam_shape[i, :k] = self.cam_shape[i, order]
                    self.cam_bins[i, :k] = self.cam_bins[i, order]
                    self.cam_present[i, :k] = self.cam_present[i, order]
                    self.side[i, :k] = side
                self.present[i] = self.cam_present[i]
            self.rows = np.arange(n)[:, None]
            self.chunk = None
            self.clock = 0
            self.initial_estimates = [list(x) for x in self._camera_estimates]
            return
        for i, lay in enumerate(layouts):
            obs = [sc.by_name[p.name] for p in lay.placements][:P]
            if not obs:
                continue
            pos = np.array([p.pos for p in lay.placements][:P])
            bins = np.array([BINS.index(lay.mapping[o.color]) for o in obs])
            order, side = plan_for(pos, bins)
            k = len(obs)
            self.idx[i, :k] = [obs[j].index for j in order]
            self.shape[i, :k] = [SHAPE_CODE[obs[j].shape] for j in order]
            self.bins[i, :k], self.side[i, :k], self.present[i, :k] = bins[order], side, True
        self.rows = np.arange(n)[:, None]
        self.chunk = None                    # the policy's last answer, as places
        self.clock = 0
        self.initial_estimates = [[] for _ in range(n)]

    def camera_diagnostics(self, i: int, layout: Layout) -> list[dict]:
        """Match the first camera detections to truth for post-run failure analysis.

        This is diagnostic bookkeeping only; the rollout itself never reads the layout
        or simulator object arrays in camera mode.
        """
        if not self.camera:
            return []
        estimates = list(self.initial_estimates[i])
        used: set[int] = set()
        out = []
        for p in layout.placements:
            truth = self.s.scene.by_name[p.name]
            candidates = [j for j, e in enumerate(estimates)
                          if j not in used and e.color == truth.color]
            if candidates:
                j = min(candidates, key=lambda q: np.linalg.norm(estimates[q].pos[:2] - p.pos[:2]))
                used.add(j)
                e = estimates[j]
                sym = SHAPE_SYMMETRY[truth.shape]
                if sym:
                    period = 2.0 * np.pi / sym
                    yaw_error = abs((e.yaw - p.yaw + period / 2) % period - period / 2)
                else:
                    yaw_error = 0.0
                out.append({"name": p.name, "color": truth.color, "shape": truth.shape,
                            "visible": True, "estimate_shape": e.shape,
                            "position_error_mm": round(float(np.linalg.norm(e.pos - p.pos) * 1000), 3),
                            "yaw_error_deg": round(float(np.degrees(yaw_error)), 3)})
            else:
                out.append({"name": p.name, "color": truth.color, "shape": truth.shape,
                            "visible": False, "estimate_shape": None,
                            "position_error_mm": None, "yaw_error_deg": None})
        return out

    def _closure_score(self, hand: np.ndarray) -> np.ndarray:
        """How closed each commanded/measured hand is, using only finger joint angles."""
        q = np.asarray(hand)
        d = _HAND_DELTA[_HAND_FINGER]
        score = ((q[..., _HAND_FINGER] - _HAND_OPEN[_HAND_FINGER]) * d / (d * d)).mean(-1)
        return np.clip(score, 0.0, 1.0)

    def _set_camera_parts(self, i: int, estimates: list[PartEstimate], obs=None) -> None:
        """Track coloured parts through occlusion, grasping and release.

        A top camera sees a held part's fingers and side faces as a new, moving
        silhouette.  While a hand is closed near a track, retain the last stable
        tabletop estimate.  A track is retired only after the closed hand reaches
        its known bin and opens, or the camera actually sees the coloured part in
        that bin.  Both cues come from the camera and the robot's own commands,
        never from simulator object state.
        """
        P = MAX_PARTS
        if obs is None:
            hand_pos = np.zeros((2, 3))
            close = np.zeros(2)
        else:
            hand_pos = np.asarray(obs.pinch[i])
            close = self._closure_score(obs.hand_q[i])
        # On later frames match by colour and proximity.  A hand can hide a part for
        # several frames, so an unmatched track is intentionally held.
        initial = not bool(self.cam_present[i].any())
        used: set[int] = set()
        order = sorted(range(len(estimates)), key=lambda j: estimates[j].pos[0])
        for j in order:
            e = estimates[j]
            b = _COLOR_BIN[e.color]
            candidates = [] if initial else [k for k in range(P) if self.cam_present[i, k] and k not in used
                                             and int(self.cam_bins[i, k]) == b and
                                             np.linalg.norm(self.cam_pos[i, k, :2] - e.pos[:2]) < 0.08]
            if candidates:
                k = min(candidates, key=lambda q: np.linalg.norm(self.cam_pos[i, q, :2] - e.pos[:2]))
            else:
                # A hand edge can produce a second component of the same colour
                # for one frame.  Do not turn that into a new part when a track of
                # that colour already exists; all episode parts are visible in the
                # initial head-camera frame in this benchmark.
                if not initial and any(self.cam_present[i, q] and int(self.cam_bins[i, q]) == b for q in range(P)):
                    continue
                k = next((q for q in range(P) if not self.cam_present[i, q] and not self.cam_done[i, q]), None)
                if k is None:
                    continue
            near_hand = np.linalg.norm(hand_pos - e.pos[None, :], axis=1) < 0.09
            # Do not call a tabletop approach a grasp.  A track becomes carried
            # only after the commanded fingers close and FK shows the hand lifting.
            lifted_hand = hand_pos[:, 2] > self.feat.rest_z + 0.035
            hand_silhouette = bool(np.any(near_hand & (close > 0.30) & lifted_hand))
            # A component in the expected bin is a camera confirmation that the
            # released part is done.  Put the track at a known safe point inside
            # the bin, rather than trusting a wall/edge depth pixel.
            bin_lo, bin_hi = self.s.scene.bin_inner(BINS[b])
            in_expected_bin = bool(np.all(e.pos >= bin_lo) and np.all(e.pos <= bin_hi))
            if in_expected_bin and not hand_silhouette:
                c = self.s.scene.bin_centers[BINS[b]].copy()
                self.cam_pos[i, k] = c + [0.0, 0.0, 0.04]
                self.cam_done[i, k] = True
                self.cam_present[i, k] = False
                used.add(k)
                continue
            if self.cam_carried[i, k] or hand_silhouette:
                self.cam_carried[i, k] = True
                # The colour track supplies identity and shape, while the robot's
                # own FK hand position supplies the current centre once the part
                # is in the hand.  Feeding the old tabletop point here makes a
                # state policy repeat its grasp forever.
                if obs is not None:
                    side = int(self.side[i, k])
                    self.cam_pos[i, k] = hand_pos[side]
                used.add(k)
                continue
            # Exponential filtering removes frame-to-frame depth jitter while
            # preserving a real tabletop move.  Shape votes make a single noisy
            # outline harmless without inventing an identity.
            alpha = 0.35
            if not self.cam_shape_votes[i, k].any():
                self.cam_pos[i, k] = e.pos
                self.cam_yaw[i, k] = e.yaw
            else:
                self.cam_pos[i, k] = (1.0 - alpha) * self.cam_pos[i, k] + alpha * e.pos
                self.cam_yaw[i, k] = (1.0 - alpha) * self.cam_yaw[i, k] + alpha * e.yaw
            if not self.cam_shape_locked[i, k]:
                self.cam_shape_votes[i, k, _SHAPE_CODE[e.shape]] += 1
                self.cam_shape[i, k] = int(np.argmax(self.cam_shape_votes[i, k]))
            self.cam_bins[i, k] = b
            self.cam_present[i, k] = True
            used.add(k)

        # Once the carried hand reaches its destination and opens, the camera
        # policy can retire the track even when the bin hides the part.
        for k in range(P):
            if not self.cam_present[i, k] or not self.cam_carried[i, k]:
                continue
            if obs is not None:
                self.cam_pos[i, k] = hand_pos[int(self.side[i, k])]
            b = int(self.cam_bins[i, k])
            target = self.s.scene.bin_centers[BINS[b]]
            near_bin = np.linalg.norm(hand_pos - target[None, :], axis=1) < 0.16
            if bool(np.any(near_bin & (close < 0.25))):
                self.cam_present[i, k] = False
                self.cam_done[i, k] = True
            elif bool(np.any(close < 0.25)):
                # A failed grasp or an early release needs a fresh camera match;
                # retaining the old hand position makes the policy repeat a stale
                # track instead of trying the part it can now see again.
                self.cam_present[i, k] = False
                self.cam_carried[i, k] = False
                self.cam_shape_votes[i, k] = 0
                self.cam_shape_locked[i, k] = False

    def _camera_update(self, obs) -> None:
        for i in range(self.s.n):
            estimates = self.perception.observe(self.s.sim.data[i])
            self._set_camera_parts(i, estimates, obs)

    def sorted(self, obs) -> np.ndarray:
        """(N,) parts of each env that are in a bin."""
        if self.camera:
            return self.cam_done.sum(axis=1) + (self.cam_present & self.feat.in_bin(self.cam_pos)).sum(axis=1)
        return (self.present & self.feat.in_bin(obs.obj_pos[self.rows, self.idx])).sum(axis=1)

    def step(self, obs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Targets for this tick: pinch (N, 2, 3), palm (N, 2, 3, 3), fingers (N, 14).

        The policy is asked every ``every`` ticks and its answer is played out in between."""
        if self.clock % self.every == 0:
            if self.camera:
                self._camera_update(obs)
                pos, yaw, shape, bins, present = self.cam_pos, self.cam_yaw, self.cam_shape, self.cam_bins, self.cam_present
            else:
                pos, yaw, shape, bins, present = (obs.obj_pos[self.rows, self.idx], obs.obj_yaw[self.rows, self.idx],
                                                   self.shape, self.bins, self.present)
            arm, asked_fingers = self.s.last_action
            palm, R, _ = self.s.kin.fk(arm.reshape(-1, 2, 7))
            asked = palm + np.einsum("nsij,sj->nsi", R, self.s.off)
            x = self.feat.observe(obs.pinch, obs.rot, obs.hand_q, pos, yaw, shape, bins, self.side, present,
                                  asked, asked_fingers)
            y = self.policy.act(x).astype(float)
            y[..., :6] += asked.reshape(-1, 1, 6)                    # moves -> places
            if self.chunk is not None and self.blend:
                # Ease from what the last answer said about these ticks into the new one.
                old = self.chunk[:, self.every:self.every + self.blend]
                w = (np.arange(1, old.shape[1] + 1) / (old.shape[1] + 1))[None, :, None]
                y[:, :old.shape[1]] = (1 - w) * old + w * y[:, :old.shape[1]]
            self.chunk = y
        now = self.chunk[:, self.clock % self.every]
        self.clock += 1
        pinch, rot, fingers = self.feat.targets(now, 0.0)
        return pinch, rot, np.clip(fingers, self.hand_lo, self.hand_hi)


def run(policy: Policy, n: int = 48, episodes: int = 2, seed: int = 100, parts: int | None = None,
        max_s: float = 80.0, threads: int | None = None, session: Session | None = None, log=None,
        every: int = EVERY, blend: int = BLEND, progress=None, perception: str | None = None,
        depth_noise_mm: float = 0.0, edge_noise_px: float = 0.0, layouts: list[Layout] | None = None,
        scene_records: list[dict] | None = None, scene_offset: int = 0) -> dict:
    """Roll the policy out on ``n * episodes`` fresh scenes. Returns success rates and counts.
    ``progress(fraction)`` is called twice a second of robot time."""
    if layouts is not None:
        if len(layouts) != n * episodes:
            raise ValueError("layouts must contain exactly n * episodes held-out tables")
    s = session or Session(n, seed=seed, threads=threads)
    rng = np.random.default_rng(seed)
    pilot = Pilot(s, policy, every, blend, perception, depth_noise_mm, edge_noise_px)
    tot = {"scenes": 0, "success": 0, "parts": 0, "sorted": 0, "clean": 0, "clipped": 0, "aborted": 0, "seconds": [],
           "outcomes": {}, "by_parts": {}, "untouched": 0, "in_hand": 0}
    # Sharded measurement jobs can discard the preceding deterministic layouts and
    # then run their contiguous slice without changing the global scene stream.
    for _ in range(int(scene_offset)):
        discard_parts = parts if parts else int(rng.choice([3, 4, 5], p=[0.35, 0.40, 0.25]))
        s.sample_lead(discard_parts, rng)
    for ep in range(episodes):
        ep_layouts = (layouts[ep * s.n:(ep + 1) * s.n] if layouts is not None else
                      [s.sample_lead(parts if parts else int(rng.choice([3, 4, 5], p=[0.35, 0.40, 0.25])), rng)
                       for _ in range(s.n)])
        s.free_episode(ep_layouts)
        pilot.begin(ep_layouts)
        # The denominator is the simulator's active layout for every arm.  In camera mode
        # a hidden part is precisely a failure, so counting only detected tracks would bias
        # the parts score upward.
        want = np.array([len(lay.placements) for lay in ep_layouts], int)
        calm = 0
        t0 = time.time()
        for tick in range(int(max_s * CONTROL_HZ)):
            obs = s.observe()
            s.drive(*pilot.step(obs))
            if tick % 15 == 0:
                # Stop once every robot has sorted everything and brought its hands back.
                home = np.linalg.norm(obs.pinch - s.home_pinch, axis=-1).max(axis=1) < 0.03
                calm = calm + 1 if ((pilot.sorted(obs) >= want) & home).all() else 0
                if calm >= 2:
                    break
                if progress:
                    progress((ep + tick / (max_s * CONTROL_HZ)) / episodes)
        r = s.result()
        ok = r.success & ~r.aborted
        good = np.array([sum(v == "correct" for v in o.values()) for o in r.outcomes])
        tot["scenes"] += s.n
        tot["success"] += int(ok.sum())
        tot["clean"] += int((ok & (r.clipped == 0)).sum())
        tot["parts"] += int(want.sum())
        tot["sorted"] += int(good.sum())
        tot["clipped"] += int(r.clipped.sum())
        tot["aborted"] += int(r.aborted.sum())
        tot["seconds"].append(round(r.ticks / CONTROL_HZ, 1))
        for o in r.outcomes:
            for v in o.values():
                tot["outcomes"][v] = tot["outcomes"].get(v, 0) + 1
        for k, good_k in zip(want, ok):
            a = tot["by_parts"].setdefault(int(k), [0, 0])
            a[0] += int(good_k)
            a[1] += 1
        # Of the parts not in a bin: still where they started, or up in a hand.
        obs = s.observe()
        pos = (pilot.cam_pos if pilot.camera else obs.obj_pos[pilot.rows, pilot.idx])
        if pilot.camera:
            # These counters are diagnostic only.  Keep them out of the camera
            # policy path and avoid assigning camera tracks simulator identities.
            start = pos.copy()
        else:
            start = np.zeros_like(pos)
            for i, lay in enumerate(ep_layouts):
                byname = {p.name: p.pos for p in lay.placements}
                for j in range(int(want[i])):
                    start[i, j] = byname[s.scene.objects[pilot.idx[i, j]].name]
        out = pilot.present & ~pilot.feat.in_bin(pos)
        tot["untouched"] += int((out & (np.linalg.norm(pos - start, axis=-1) < 0.01)).sum())
        tot["in_hand"] += int((out & (pos[..., 2] > s.scene.rest_z + 0.03)).sum())
        if scene_records is not None:
            for i, lay in enumerate(ep_layouts):
                scene_records.append({
                    "episode": ep, "scene": int(scene_offset + ep * s.n + i), "seed": int(seed),
                    "parts": int(want[i]), "sorted": int(good[i]),
                    "success": bool(ok[i]), "clean": bool(ok[i] and r.clipped[i] == 0),
                    "clipped": int(r.clipped[i]), "aborted": bool(r.aborted[i]),
                    "seconds": round(float(r.ticks / CONTROL_HZ), 1),
                    "outcomes": dict(r.outcomes[i]),
                    "camera_diagnostics": pilot.camera_diagnostics(i, lay),
                })
        if log:
            log(f"  round {ep + 1}: {int(ok.sum())}/{s.n} scenes sorted, {int(good.sum())}/{int(want.sum())} parts, "
                f"{int(r.clipped.sum())} clipped, {r.ticks / CONTROL_HZ:.0f} s sim in {time.time() - t0:.0f} s")
    if pilot.perception is not None:
        pilot.perception.close()
    if session is None:
        s.close()
    tot["success_rate"] = round(tot["success"] / max(tot["scenes"], 1), 4)
    tot["clean_rate"] = round(tot["clean"] / max(tot["scenes"], 1), 4)
    tot["parts_rate"] = round(tot["sorted"] / max(tot["parts"], 1), 4)
    return tot


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--parts", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--every", type=int, default=EVERY, help="ticks between two questions to the policy")
    ap.add_argument("--blend", type=int, default=BLEND)
    ap.add_argument("--perception", choices=("camera",), default=None,
                    help="estimate parts from the head RGB/depth camera")
    ap.add_argument("--depth-noise-mm", type=float, default=0.0,
                    help="Gaussian depth noise in millimetres for camera perception")
    ap.add_argument("--edge-noise-px", type=float, default=0.0,
                    help="one-pixel-style depth edge dropout (0 disables it)")
    a = ap.parse_args(argv)
    out = run(Policy.load(a.policy), a.n, a.episodes, a.seed, a.parts, threads=a.threads, log=print,
              every=a.every, blend=a.blend, perception=a.perception,
              depth_noise_mm=a.depth_noise_mm, edge_noise_px=a.edge_noise_px)
    print(out)


if __name__ == "__main__":
    main()
