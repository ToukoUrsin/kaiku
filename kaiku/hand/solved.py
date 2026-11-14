"""Fingers from a hand that somebody else solved (Rokoko's ``rkk-hand-solver``, see ``kaiku.rgmp.solved``).

The other solver has a skeleton of its own, with other bone lengths. What carries over is where every bone
points, so the joints are read from the bone directions and the kaiku hand model does the rest: fingertips,
curl and pinch in the same terms as ``GloveSolver``, for the same calibration and retargeting.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from kaiku.hand.model import HandModel
from kaiku.types import HumanHand

# The side-to-side angle of a digit is read from its first bone. Curled past this, that bone points too
# nearly down to say which way it leans, and the digit counts as straight ahead.
_LEVEL_ENOUGH = 0.2


def joints_from_points(model: HandModel, points: np.ndarray) -> np.ndarray:
    """Joint values (5, 4) whose bones point the way the chains in ``points`` do.

    ``points`` is (5, 4, 3): [joint 0, joint 1, joint 2, tip] of every digit in the hand frame, the layout of
    ``HandModel.points``. Exact inverse of it for a hand the model can make; bone lengths do not matter.
    """
    bones = np.diff(np.asarray(points, dtype=float), axis=1)                    # (5, 3, 3)
    bones /= np.maximum(np.linalg.norm(bones, axis=-1, keepdims=True), 1e-9)
    local = np.einsum("dji,dkj->dki", model.base_rot, bones)                    # in each digit's own frame
    first = local[:, 0]
    yaw = np.where(first[:, 0] > _LEVEL_ENOUGH, np.arctan2(first[:, 1], first[:, 0]), 0.0)
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    pitch = np.arctan2(-local[:, :, 2], local[:, :, 0] * c + local[:, :, 1] * s)
    joints = np.empty((5, 4))
    joints[:, 0] = pitch[:, 0]
    # A curled digit's later bones can be pitched past 180 degrees in total; unwrap each bend into (-90, 270).
    joints[:, 1:3] = (np.diff(pitch, axis=1) + 0.5 * np.pi) % (2.0 * np.pi) - 0.5 * np.pi
    joints[:, 3] = model.mirror * yaw
    return model.clamp(joints)


def hand_length(points: np.ndarray) -> float:
    """Wrist joint to middle fingertip along the bones, the measure ``HandModel`` is sized by."""
    middle = points[2]
    return float(np.linalg.norm(middle[0]) + np.linalg.norm(np.diff(middle, axis=0), axis=1).sum())


def with_fingers(hand: HumanHand, model: HandModel, points: np.ndarray) -> HumanHand:
    """``hand`` (wrist and all) with its fingers replaced by the ones in ``points``.

    The joints are read from the bone directions. The thumb-to-fingertip distances are the other solver's
    own, at the size of the model's hand, because those close the robot's fingers and nothing should sit
    between Rokoko's measurement and them.
    """
    points = np.asarray(points, dtype=float)
    joints = joints_from_points(model, points)
    tips = points[:, 3] * (model.hand_length / max(hand_length(points), 1e-6))
    gaps = np.linalg.norm(tips[1:] - tips[0], axis=1)
    return replace(hand, joints=joints, tips=model.tips(joints), curl=model.curl(joints), pinch=gaps, gaps=gaps)
