"""Parity checks against the vendored Aaltonomous runtime.

The workspace comparison allows 2e-5 rad in the returned command.  On the
checked-in models the measured maximum difference at the bisection boundary is
0 (1.1e-7 m when comparing the two violation calculations); the allowance
keeps this test honest about Pinocchio/MuJoCo floating-point differences.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "third_party" / "aaltonomous_safety"
sys.path.insert(0, str(ROOT))

from kaiku.sim.safety import SafetyConfig as KaikuSafetyConfig  # noqa: E402
from kaiku.sim.safety import SafetyFilter as KaikuSafetyFilter  # noqa: E402
from kaiku.sim.twin import build_twin  # noqa: E402


def _load_official():
    path = VENDORED / "safety.py"
    spec = importlib.util.spec_from_file_location("aaltonomous_official_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # The official source's default points at its checkout layout; the vendored URDF is local here.
    module.WorkspaceChecker.__init__.__defaults__ = (VENDORED / "g1_body29_hand14.urdf",)
    module.log.setLevel(logging.ERROR)
    return module


official = _load_official()
TWIN = build_twin(with_scene=False)
JOINT_SPEC = official.JointSpec(VENDORED / "joint_spec.yaml")
ATOL = 2e-5


def _pair(*, workspace: bool = True):
    if workspace:
        kc = KaikuSafetyConfig(max_clipped_steps=10**6)
        oc = official.SafetyConfig(max_clipped_steps=10**6)
    else:
        # Keep every other rule active while making the workspace geometrically unreachable.
        far_lo, far_hi = (-100.0,) * 3, (100.0,) * 3
        keep_lo, keep_hi = (200.0,) * 3, (201.0,) * 3
        kc = KaikuSafetyConfig(workspace_min=far_lo, workspace_max=far_hi,
                               counter_keepout_min=keep_lo, counter_keepout_max=keep_hi,
                               max_clipped_steps=10**6)
        oc = official.SafetyConfig(workspace_min=far_lo, workspace_max=far_hi,
                                   counter_keepout_min=keep_lo, counter_keepout_max=keep_hi,
                                   max_clipped_steps=10**6)
    return KaikuSafetyFilter(TWIN, 1, cfg=kc), official.SafetyFilter(JOINT_SPEC, oc)


def _replay(targets: list[tuple[np.ndarray, np.ndarray]], *, workspace: bool = True,
            measured: np.ndarray | None = None):
    kaiku, ref = _pair(workspace=workspace)
    measured = np.zeros(28) if measured is None else np.asarray(measured)
    first_measured = measured[0] if measured.ndim == 2 else measured
    kaiku.reset(0, first_measured[:14], first_measured[14:])
    ref.reset(first_measured[:14], first_measured[14:], 0.0)
    max_diff = 0.0
    flags = []
    reasons = []
    for i, (arm, hand) in enumerate(targets):
        got_arm, got_hand, got_clip = kaiku.filter(arm[None], hand[None])
        current = measured[i] if measured.ndim == 2 else measured
        want = ref.filter({"arm_q_target": arm, "hand_q_target": hand}, (i + 1) / 30.0,
                          arm_q=current[:14], hand_q=current[14:])
        got = np.concatenate([got_arm[0], got_hand[0]])
        expected = np.concatenate([want.arm_q_cmd, want.hand_q_cmd])
        max_diff = max(max_diff, float(np.max(np.abs(got - expected))))
        flags.append((bool(got_clip[0]), bool(want.clipped)))
        reasons.append(want.reasons)
    return max_diff, flags, reasons


def test_rule_breaking_sequences_match():
    """Joint limits, command-rate limits, workspace and counter keep-out agree tick by tick."""
    rng = np.random.default_rng(2026)
    limit_seq = [(rng.uniform(-10.0, 10.0, 14), rng.uniform(-10.0, 10.0, 14))
                 for _ in range(4)]
    rate_seq = [(rng.uniform(-0.8, 0.8, 14), np.zeros(14)) for _ in range(16)]

    # Choose an in-URDF pose that violates the geometric rule, then repeatedly ask for it
    # so the command reaches the boundary through the 1 rad/s velocity cap.
    probe, _ = _pair(workspace=True)
    workspace_target = None
    for _ in range(1000):
        candidate = rng.uniform(probe.arm_lo, probe.arm_hi)
        if probe.violation(candidate[None])[0] > 0.05:
            workspace_target = candidate
            break
    assert workspace_target is not None
    workspace_seq = [(workspace_target, np.zeros(14))] * 120

    for sequence, workspace, expected_reason in ((limit_seq, False, ("arm_position", "hand_position")),
                                                  (rate_seq, False, ("arm_velocity",)),
                                                  (workspace_seq, True, ("workspace",))):
        max_diff, flags, reasons = _replay(sequence, workspace=workspace)
        assert max_diff <= ATOL
        assert all(a == b for a, b in flags), (max_diff, flags)
        assert any(any(r in expected_reason for r in rs) for rs in reasons)

    # A second random in-limit pose penetrates the configured counter keep-out.  Removing only
    # that box removes the violation in both kinematics implementations.
    counter_target = np.zeros(14)
    counter_target[[3, 10]] = rng.uniform(0.45, 0.55, 2)
    kaiku, ref = _pair(workspace=True)
    assert kaiku.violation(counter_target[None])[0] > 0
    assert ref.ws.violation(counter_target) > 0
    no_counter = official.SafetyConfig(counter_keepout_min=None, counter_keepout_max=None)
    no_counter_ws = official.WorkspaceChecker(JOINT_SPEC, no_counter,
                                               VENDORED / "g1_body29_hand14.urdf")
    assert no_counter_ws.violation(counter_target) < ref.ws.violation(counter_target)


def test_recorded_demo_fixture_matches():
    """A 96-tick real recorder slice is accepted identically by both filters."""
    data = np.load(ROOT / "tests" / "fixtures" / "safety_demo.npz")
    state, action = data["state"], data["action"]
    targets = [(row[:14], row[14:]) for row in action]
    max_diff, flags, _ = _replay(targets, measured=state)
    assert max_diff <= ATOL
    assert all(a == b for a, b in flags)
    assert not any(a for a, _ in flags)
