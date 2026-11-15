"""One teleoperation session: the operator's robot plus its echoes, ticked at 30 Hz.

Env 0 is the lead, driven by the operator (gloves, a recording, or the autopilot). Every other
env is a follower with its own variation of the lead's scene. Each tick:

    lead command ─┐
                  ├─> palm targets (N) ─> arm IK ─> shaper ─> safety filter ─> physics (N)
    echo fan-out ─┘                                                              │
          ^───────────── measured hands and parts, lead and followers ──────────┘

The session owns the episode logic too: sampling the scenes, deciding when everybody has
finished, and the challenge's verdict for each robot.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from kaiku.fanout.echo import Echo, EchoParams, LeadTick, quat_yaw, sample_followers
from kaiku.sim.batch import HOME_ARM, BatchSim
from kaiku.sim.grasp import HAND_OPEN, arm_for, grasp_offset, hand14
from kaiku.sim.ik import BatchArmIK
from kaiku.sim.reach import ReachMap
from kaiku.sim.safety import SafetyFilter, Shaper
from kaiku.sim.scene import Layout
from kaiku.sim.twin import CONTROL_HZ, Twin, build_twin

DT = 1.0 / CONTROL_HZ
HOME_HAND = hand14(HAND_OPEN, HAND_OPEN)
# Env state codes shared with the viewer.
IDLE, RUNNING, SUCCESS, FAILED, LEAD = 0, 1, 2, 3, 4


@dataclass
class LeadCommand:
    """What the operator asks of the lead robot on one tick."""

    pinch: np.ndarray     # (2, 3) pinch-point targets, left and right, pelvis frame
    rot: np.ndarray       # (2, 3, 3) palm orientations
    hand: np.ndarray      # (14,) Dex3 targets in the challenge's order


@dataclass
class Observation:
    """Measured state of every env at the start of a tick."""

    arm_q: np.ndarray     # (N, 14)
    hand_q: np.ndarray    # (N, 14)
    pinch: np.ndarray     # (N, 2, 3)
    rot: np.ndarray       # (N, 2, 3, 3)
    obj_pos: np.ndarray   # (N, 18, 3)
    obj_yaw: np.ndarray   # (N, 18)
    obj_quat: np.ndarray  # (N, 18, 4)


@dataclass
class EpisodeResult:
    layouts: list[Layout]
    outcomes: list[dict[str, str]]         # per env: object name -> correct / wrong / counter / off
    score: np.ndarray                      # (N,) challenge score
    success: np.ndarray                    # (N,) every object in its bin
    clipped: np.ndarray                    # (N,) safety-filter clips during the episode
    aborted: np.ndarray                    # (N,)
    ticks: int
    lag: np.ndarray = field(default_factory=lambda: np.zeros(0))   # (N-1,) worst echo lag, s
    own: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))   # (N,) took a part by itself after the lead's mistake


def draw_lead(scene, reach: ReachMap, rng: np.random.Generator, n: int | None = None,
              reachable_only: bool = True) -> Layout:
    """A scene by the challenge's rule; with ``reachable_only``, one in which the arm that
    serves each part's bin can pinch it with its open hand clear of the other parts and of
    the bins (about one draw in three is, for three parts; one in seventy for five)."""
    for _ in range(4000):
        lay = scene.sample_layout(rng, n)
        if not reachable_only:
            return lay
        sides = [0 if arm_for(lay.mapping[scene.by_name[p.name].color], p.pos[1]) == "left" else 1
                 for p in lay.placements]
        if reach.layout_ok(lay, sides):
            return lay
    return lay


def draw_scene(scene, reach: ReachMap, rng: np.random.Generator, followers: int, n: int | None = None,
               spread: float = 0.08, vary_shape: bool = True, reachable_only: bool = True
               ) -> tuple[Layout, list[Layout]]:
    """The operator's scene and one variation of it for every follower. Needs no session, so
    it can be drawn in another process while the robots work (``kaiku.app.sampler``)."""
    lead = draw_lead(scene, reach, rng, n, reachable_only)
    return lead, sample_followers(scene, lead, rng, followers, spread, vary_shape, reach)


class Session:
    def __init__(self, n_envs: int = 16, seed: int = 0, spread: float = 0.08, vary_shape: bool = True,
                 twin: Twin | None = None, threads: int | None = None, echo: EchoParams | None = None,
                 settle_s: float = 0.5, timeout_s: float = 10.0,
                 reachable_only: bool = True, lean_gain: float = 0.08, stuck_err: float = 0.010):
        self.twin = twin or build_twin()
        self.scene = self.twin.scene
        self.n = int(n_envs)
        self.rng = np.random.default_rng(seed)
        self.spread, self.vary_shape, self.reachable_only = spread, vary_shape, reachable_only
        self.lean_gain, self.stuck_err = lean_gain, stuck_err
        self.sim = BatchSim(self.twin, self.n, threads=threads)
        self.kin = BatchArmIK()
        self.safety = SafetyFilter(self.twin, self.n, self.kin)
        self.shaper = Shaper(self.safety)
        self.reach = ReachMap(self.scene, self.kin, twin=self.twin)
        self.echo = Echo(self.scene, self.n - 1, echo, self.reach)
        self.off = np.stack([grasp_offset("left"), grasp_offset("right")])          # (2, 3)
        self.settle_ticks, self.timeout_ticks = int(settle_s * CONTROL_HZ), int(timeout_s * CONTROL_HZ)
        home_p, home_R, _ = self.kin.fk(HOME_ARM.reshape(1, 2, 7))
        self.home_rot = home_R[0]
        self.home_pinch = home_p[0] + np.einsum("sij,sj->si", home_R[0], self.off)
        self.episode = 0
        self.copy = False                 # this scene every follower gets the lead's own command: no echo
        self.timing = {"ik": 0.0, "echo": 0.0, "safety": 0.0, "physics": 0.0, "observe": 0.0}
        self.new_episode()

    # ------------------------------------------------------------------ episodes
    def new_episode(self, layout: Layout | None = None, followers: list[Layout] | None = None,
                    copy: bool = False) -> None:
        """``copy``: the comparison. The followers do not echo, they are sent what the lead is sent, so
        every robot gets the same joint targets whatever lies on its own table."""
        sc = self.scene
        self.copy = copy
        lead = layout or self.sample_lead()
        fol = followers or sample_followers(sc, lead, self.rng, self.n - 1, self.spread, self.vary_shape, self.reach)
        self.layouts = [lead] + list(fol)
        for i, lay in enumerate(self.layouts):
            self.sim.reset(i, lay, hand=HOME_HAND)
            self.safety.reset(i, HOME_ARM, HOME_HAND)
        self.echo.begin(lead, list(fol))
        self.obj_idx = np.concatenate([self.echo.lead_idx[None], self.echo.foll_idx], axis=0)   # (N, n)
        self.tick_no = 0
        self.lead_done = False
        self.after_done = 0
        self.behind = 0
        self.rest_ticks = 0
        self.max_lag = np.zeros(self.n - 1)
        self.clipped = np.zeros(self.n, int)
        self.pos_err = np.zeros((self.n, 2))
        self.last_action = (np.tile(HOME_ARM, (self.n, 1)), np.tile(HOME_HAND, (self.n, 1)))
        self.episode += 1
        self._obs = None

    def free_episode(self, layouts: list[Layout]) -> None:
        """Every env gets a scene of its own and nobody leads: for a policy that drives them all
        through ``drive``."""
        assert len(layouts) == self.n
        self.layouts = list(layouts)
        for i, lay in enumerate(self.layouts):
            self.sim.reset(i, lay, hand=HOME_HAND)
            self.safety.reset(i, HOME_ARM, HOME_HAND)
        self.echo.begin(Layout([]), [Layout([]) for _ in range(self.n - 1)])
        self.copy = False
        self.tick_no = 0
        self.lead_done = False
        self.after_done = self.rest_ticks = self.behind = 0
        self.max_lag = np.zeros(self.n - 1)
        self.clipped = np.zeros(self.n, int)
        self.last_action = (np.tile(HOME_ARM, (self.n, 1)), np.tile(HOME_HAND, (self.n, 1)))
        self.episode += 1
        self._obs = None

    def sample_lead(self, n: int | None = None, rng: np.random.Generator | None = None) -> Layout:
        return draw_lead(self.scene, self.reach, rng or self.rng, n, self.reachable_only)

    def sample_scene(self, n: int | None = None, rng: np.random.Generator | None = None
                     ) -> tuple[Layout, list[Layout]]:
        """The operator's scene and one variation of it for every follower."""
        return draw_scene(self.scene, self.reach, rng or self.rng, self.n - 1, n, self.spread, self.vary_shape,
                          self.reachable_only)

    def home_command(self) -> LeadCommand:
        return LeadCommand(self.home_pinch.copy(), self.home_rot.copy(), HOME_HAND.copy())

    def finish_lead(self) -> None:
        """The operator is done with this scene; followers play out what is left."""
        if not self.lead_done:
            self.lead_done = True
            self.echo.finish()
            # Followers that fell behind (they waited for their own hands) still get to play out what is left.
            if self.n > 1 and not self.copy and self.echo.len:
                self.behind = int(min(np.ceil(self.echo.lag.max() * CONTROL_HZ), 60 * CONTROL_HZ))

    @property
    def settled(self) -> bool:
        """Everybody has finished (or run out of time) and things have come to rest."""
        if not self.lead_done:
            return False
        if self.after_done >= self.timeout_ticks + self.behind:
            return True
        if self.n > 1 and not self.copy and not self.echo.done.all():
            return False
        return self.rest_ticks >= self.settle_ticks

    # ------------------------------------------------------------------ per tick
    def observe(self) -> Observation:
        if self._obs is None:
            t0 = time.perf_counter()
            arm_q, hand_q = self.sim.arm_q(), self.sim.hand_q()
            palm, rot, _ = self.kin.fk(arm_q.reshape(self.n, 2, 7))
            pinch = palm + np.einsum("nsij,sj->nsi", rot, self.off)
            quat = self.sim.obj_quat()
            self._obs = Observation(arm_q, hand_q, pinch, rot, self.sim.obj_pos(), quat_yaw(quat), quat)
            self.timing["observe"] += time.perf_counter() - t0
        return self._obs

    def _on_mat(self, pos: np.ndarray, lifted: float) -> np.ndarray:
        """Parts still to be sorted: inside (a margin around) the band and not lifted off it."""
        lo, hi = self.scene.region
        m = 0.06
        return ((pos[..., 0] > lo[0] - m) & (pos[..., 0] < hi[0] + m) & (pos[..., 1] > lo[1] - m)
                & (pos[..., 1] < hi[1] + m) & (pos[..., 2] < self.scene.rest_z + lifted)
                & (pos[..., 2] > self.scene.rest_z - 0.02))

    def lead_tick(self, obs: Observation) -> LeadTick:
        idx = self.echo.lead_idx
        pos = obs.obj_pos[0, idx]                                                    # (n, 3)
        held = np.full(2, -1, int)
        if len(idx):
            d = np.linalg.norm(pos[None] - obs.pinch[0][:, None], axis=-1)           # (2, n)
            up = pos[:, 2] > self.scene.rest_z + 0.012
            for s in range(2):
                k = int(np.argmin(d[s]))
                if d[s, k] < 0.05 and up[k]:
                    held[s] = k
        return LeadTick(obs.pinch[0], obs.rot[0], self.safety.hand_cmd[0].copy(), pos[:, :2], obs.obj_yaw[0, idx],
                        self._on_mat(pos, 0.03), held, obs.arm_q[0].copy())

    def tick(self, cmd: LeadCommand) -> Observation:
        """Advance every env by one control tick. Returns the observation the tick started from."""
        obs = self.observe()
        n, F = self.n, self.n - 1
        pinch_t = np.empty((n, 2, 3))
        rot_t = np.empty((n, 2, 3, 3))
        hand_t = np.empty((n, 14))
        pinch_t[0], rot_t[0], hand_t[0] = cmd.pinch, cmd.rot, cmd.hand
        t0 = time.perf_counter()
        if not self.lead_done:
            self.echo.push(self.lead_tick(obs))
        if F and self.copy:
            pinch_t[1:], rot_t[1:], hand_t[1:] = cmd.pinch, cmd.rot, cmd.hand
        elif F:
            rows = np.arange(1, n)[:, None]
            fpos = obs.obj_pos[rows, self.echo.foll_idx]                              # (F, n, 3)
            fyaw = obs.obj_yaw[rows, self.echo.foll_idx]
            pinch_t[1:], rot_t[1:], hand_t[1:] = self.echo.targets(
                fpos[..., :2], fyaw, self._on_mat(fpos, 0.06), obs.pinch[1:], obs.rot[1:])
            self.max_lag = np.maximum(self.max_lag, self.echo.lag)
            if self.lead_done and self.echo.tidying.any():        # a follower taking a part of its own gets the time
                self.behind = int(min(max(self.behind, self.after_done), 60 * CONTROL_HZ))
        self.timing["echo"] += time.perf_counter() - t0
        lean = np.broadcast_to(self.kin.rest, (n, 2, 7)).copy()
        if F:
            lean[1:] = self.echo.lead_arm.reshape(F, 2, 7)
        self.drive(pinch_t, rot_t, hand_t, lean)
        if self.lead_done:
            self.after_done += 1
            moving = (self.sim.obj_speed() > 0.02) & self.sim.active
            self.rest_ticks = 0 if moving.any() else self.rest_ticks + 1
        return obs

    def drive(self, pinch_t: np.ndarray, rot_t: np.ndarray, hand_t: np.ndarray, lean: np.ndarray | None = None) -> None:
        """Send every env its targets for one control tick: pinch points (N, 2, 3), palm
        rotations (N, 2, 3, 3) and finger targets (N, 14). They become joint targets through
        the arm IK, the shaper and the safety filter, and the physics advances one tick."""
        n = self.n
        t1 = time.perf_counter()
        palm_t = pinch_t - np.einsum("nsij,sj->nsi", rot_t, self.off)
        # Solve from the last command, not from a free-running solution: the answer can then
        # never run ahead of what the rate limit lets the robot do. The arm's spare degree of
        # freedom leans towards a posture known to work: over the mat the one tabulated for
        # that grasp, elsewhere ``lean`` (for an echo, the lead robot's own posture).
        lean = np.broadcast_to(self.kin.rest, (n, 2, 7)).copy() if lean is None else lean
        table, trust = self.reach.posture(pinch_t, np.arctan2(rot_t[..., 1, 0], rot_t[..., 0, 0]))
        lean += trust[..., None] * (table - lean)
        q, err, _ = self.kin.solve(self.safety.arm_cmd, palm_t, rot_t, rest=lean, rest_gain=self.lean_gain)
        # An arm that cannot get there from where it is (a joint against its stop) starts again
        # from that posture; the rate limit turns the jump into a smooth sweep.
        stuck = (err > self.stuck_err).any(axis=1)
        if stuck.any():
            q2, err2, _ = self.kin.solve(lean[stuck].reshape(-1, 14), palm_t[stuck], rot_t[stuck], iters=8,
                                         rest=lean[stuck], rest_gain=self.lean_gain)
            better = err2 < 0.5 * err[stuck]
            qs, es = q[stuck].reshape(-1, 2, 7), err[stuck]
            qs[better], es[better] = q2.reshape(-1, 2, 7)[better], err2[better]
            q[stuck], err[stuck] = qs.reshape(-1, 14), es
        self.pos_err = err
        t2 = time.perf_counter()
        arm, hand = self.shaper.shape(q, hand_t)
        arm, hand, clipped = self.safety.filter(arm, hand)
        self.clipped += clipped
        t3 = time.perf_counter()
        self.sim.step(arm, hand)
        t4 = time.perf_counter()
        for k, v in (("ik", t2 - t1), ("safety", t3 - t2), ("physics", t4 - t3)):
            self.timing[k] += v
        self.last_action = (arm, hand)
        self.tick_no += 1
        self._obs = None

    # ------------------------------------------------------------------ results
    def progress(self) -> np.ndarray:
        """(N,) parts already resting in their own bin."""
        return np.array([sum(o == "correct" for o in self.sim.outcomes(i).values()) for i in range(self.n)])

    def result(self) -> EpisodeResult:
        outcomes = [self.sim.outcomes(i) for i in range(self.n)]
        score = np.array([self.scene.score(o.values()) for o in outcomes])
        success = np.array([all(v == "correct" for v in o.values()) and len(o) > 0 for o in outcomes])
        success &= ~self.safety.aborted
        # A follower that finished a part with the stand-in's pick did not get there on the operator's motion.
        own = np.zeros(self.n, bool)
        if self.n > 1 and not self.copy:
            own[1:] = self.echo.helped
        return EpisodeResult(list(self.layouts), outcomes, score, success, self.clipped.copy(),
                             self.safety.aborted.copy(), self.tick_no, self.max_lag.copy(), own)

    def close(self) -> None:
        self.sim.close()
