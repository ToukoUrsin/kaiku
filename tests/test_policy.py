"""The policy's own arithmetic: what it sees, what it commands, and the round trip between them."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kaiku.policy.features import ACT_DIM, HORIZON, MAX_PARTS, Featurizer, from_rot6, plan_for, rot6  # noqa: E402
from kaiku.policy.model import Net, Policy  # noqa: E402
from kaiku.policy.train import jitter  # noqa: E402
from kaiku.sim.scene import BINS, Scene  # noqa: E402


def turns(rng, n):
    q, _ = np.linalg.qr(rng.normal(size=(n, 3, 3)))
    return q * np.sign(np.linalg.det(q))[:, None, None]


def scene_of(feat, rng, B=6, n=3):
    """B robots with n parts on the mat, both hands above it."""
    P = MAX_PARTS
    pos = np.zeros((B, P, 3))
    pos[:, :n] = np.array([0.30, 0.0, feat.rest_z]) + rng.uniform(-0.03, 0.03, (B, n, 3)) * [1, 5, 0]
    present = np.broadcast_to(np.arange(P) < n, (B, P))
    return {"pinch": rng.uniform(-0.1, 0.1, (B, 2, 3)) + [0.3, 0.0, 0.2], "rot": turns(rng, B * 2).reshape(B, 2, 3, 3),
            "fingers": rng.uniform(-1, 1, (B, 14)), "pos": pos, "yaw": rng.uniform(-0.7, 0.7, (B, P)),
            "shape": rng.integers(0, 3, (B, P)), "bins": rng.integers(0, 3, (B, P)),
            "side": rng.integers(0, 2, (B, P)), "present": present}


def test_rotation_survives_six_numbers():
    R = turns(np.random.default_rng(0), 50)
    assert np.allclose(from_rot6(rot6(R)), R, atol=1e-9)


def test_command_and_targets_are_inverse():
    rng = np.random.default_rng(1)
    pinch, asked = rng.normal(size=(4, HORIZON, 2, 3)), rng.normal(size=(4, 1, 2, 3))
    rot, fingers = turns(rng, 4 * HORIZON * 2).reshape(4, HORIZON, 2, 3, 3), rng.normal(size=(4, HORIZON, 14))
    y = Featurizer.command(pinch, rot, fingers, asked)
    assert y.shape == (4, HORIZON, ACT_DIM)
    p2, r2, f2 = Featurizer.targets(y, asked)
    assert np.allclose(p2, pinch) and np.allclose(r2, rot, atol=1e-9) and np.allclose(f2, fingers)


def test_sees_the_first_part_still_on_the_mat():
    feat, rng = Featurizer(Scene()), np.random.default_rng(2)
    s = scene_of(feat, rng)
    obs = lambda: feat.observe(s["pinch"], s["rot"], s["fingers"], s["pos"], s["yaw"], s["shape"], s["bins"],  # noqa: E731
                               s["side"], s["present"], s["pinch"], s["fingers"])
    x, sl = obs(), feat.slices
    assert x.shape == (6, feat.dim)
    assert np.allclose(x[:, sl["part"]], s["pos"][:, 0], atol=1e-6)
    assert np.allclose(x[:, sl["left"]], 3) and np.allclose(x[:, sl["lead"]], 0)
    # The first part goes into its bin: the second is next, and its bin has one part in it if it is the same bin.
    s["pos"][:, 0] = feat.bin_c[s["bins"][:, 0]] + [0, 0, 0.02]
    x = obs()
    assert np.allclose(x[:, sl["part"]], s["pos"][:, 1], atol=1e-6)
    assert np.allclose(x[:, sl["slot"]][:, 0], s["bins"][:, 0] == s["bins"][:, 1])
    # Everything sorted: nothing left to look at.
    s["pos"][:, :3] = feat.bin_c[s["bins"][:, :3]] + [0, 0, 0.02]
    x = obs()
    assert not x[:, sl["has"]].any() and not x[:, sl["to_part"]].any() and not x[:, sl["bin"]].any()


def test_plan_is_the_operators_habit():
    pos = np.array([[0.32, 0.10, 0.0], [0.27, -0.05, 0.0], [0.30, 0.02, 0.0]])
    bins = np.array([BINS.index("right"), BINS.index("center"), BINS.index("left")])
    order, side = plan_for(pos, bins)
    assert list(order) == [1, 2, 0]                    # nearest first
    assert list(side) == [1, 0, 1]                     # centre bin: the nearer arm; left bin: left arm; right: right


def test_jitter_keeps_the_commanded_places():
    feat = Featurizer(Scene())
    sl = feat.slices
    x, y = torch.randn(32, feat.dim), torch.randn(32, HORIZON, ACT_DIM)
    x[:, sl["has"]] = 1.0
    x2, y2 = jitter(x, y, sl, 0.004, 0.002)
    asked, asked2 = x[:, sl["pinch"]] + x[:, sl["lead"]], x2[:, sl["pinch"]] + x2[:, sl["lead"]]
    assert not torch.allclose(asked, asked2)
    assert torch.allclose(asked[:, None] + y[:, :, :6], asked2[:, None] + y2[:, :, :6], atol=1e-6)
    # The part has not moved: seen from the moved hand it is as far the other way.
    assert torch.allclose(x[:, sl["pinch"]] + x[:, sl["to_part"]], x2[:, sl["pinch"]] + x2[:, sl["to_part"]], atol=1e-6)
    assert torch.equal(y[:, :, 6:], y2[:, :, 6:])


def test_policy_file_round_trip(tmp_path):
    feat = Featurizer(Scene())
    rng = np.random.default_rng(3)
    policy = Policy(Net(feat.dim, 32, 1), rng.normal(size=feat.dim), np.ones(feat.dim), rng.normal(size=ACT_DIM),
                    np.ones(ACT_DIM), {"success": 0.5})
    x = rng.normal(size=(5, feat.dim)).astype(np.float32)
    policy.save(tmp_path / "p.pt")
    again = Policy.load(tmp_path / "p.pt")
    assert again.info == {"success": 0.5}
    assert again.act(x).shape == (5, HORIZON, ACT_DIM)
    assert np.allclose(again.act(x), policy.act(x), atol=1e-6)
