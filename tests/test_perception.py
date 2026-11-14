"""The head camera can replace the simulator's part arrays."""
import numpy as np

from kaiku.perception import CameraPerception
from kaiku.policy.model import Policy
from kaiku.policy.rollout import run
from kaiku.sim.scene import Scene
from kaiku.app.session import Session


def test_camera_estimate_one_scene():
    s = Session(1, seed=12, threads=1, reachable_only=False)
    try:
        lay = s.sample_lead(3, s.rng)
        s.free_episode([lay])
        p = CameraPerception(s.twin, width=320, height=240)
        got = p.observe(s.sim.data[0])
        assert len(got) == len(lay.placements)
        truth = [s.scene.by_name[x.name] for x in lay.placements]
        for ob in truth:
            same = [e for e in got if e.color == ob.color]
            assert same
            assert min(np.linalg.norm(e.pos - next(x.pos for x in lay.placements if x.name == ob.name))
                       for e in same) < 0.025
            assert any(e.shape == ob.shape for e in same)
        p.close()
    finally:
        s.close()


def test_camera_rollout_two_robots():
    policy = Policy.load("kaiku/policy/pretrained/kaiku.pt")
    result = run(policy, n=2, episodes=1, seed=19, max_s=0.5, threads=1,
                 perception="camera")
    assert result["scenes"] == 2
