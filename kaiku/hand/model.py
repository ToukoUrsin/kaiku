"""Parametric human hand skeleton in the kaiku hand frame, with forward kinematics and joint limits.

Hand frame (see ``kaiku.types``): origin at the wrist joint, +x to the middle fingertip, +z out of the back of
the hand, +y = z cross x, so the thumb is on +y for a right hand and on -y for a left hand.

Every digit is the same three-bone chain. The first joint has two axes (abduction about the dorsal axis, then
flexion), the other two are hinges parallel to the first joint's flexion axis:

    digit    joint 0 (2 axes)   bone 0            joint 1   bone 1             joint 2   bone 2
    thumb    CMC                metacarpal        MCP       proximal phalanx   IP        distal phalanx
    fingers  MCP                proximal phalanx  PIP       middle phalanx     DIP       distal phalanx

The thumb differs from the fingers only in where its first joint sits and how that joint's frame is turned
(``_THUMB_*`` below), which is what lets it swing across the palm and oppose the fingers.

Joint values use the ``HumanHand.joints`` layout, one row per digit in ``FINGERS`` order:
``[flex 0, flex 1, flex 2, abduction]``, radians, 0 = straight, positive flexion curls towards the palm.
Positive abduction moves a finger towards the thumb side, and moves the thumb away from the palm (palmar
abduction). Values mean the same on both sides: a left hand with the same joints is the mirror image of the
right hand.
"""
from __future__ import annotations

import numpy as np

from kaiku.hand.geometry import rot_x, rot_y, rot_z
from kaiku.types import SIDES

# ----------------------------------------------------------------------------------------------------------
# Anatomy of an average adult RIGHT hand. Lengths are fractions of the hand length (wrist joint to middle
# fingertip), so one number scales the whole hand. Sources: bone-length ratios from hand anthropometry
# surveys (phalanges roughly 0.24 / 0.15 / 0.13 of hand length for the middle finger, including the soft
# fingertip), knuckle arch from the usual 2:1 palm proportions. Good to a few millimetres for most adults.
# ----------------------------------------------------------------------------------------------------------

DEFAULT_HAND_LENGTH = 0.185
"""Wrist joint to middle fingertip of an average adult, metres (women about 0.172, men about 0.193)."""

# First joint of each digit in the hand frame: thumb CMC, then the four MCP knuckles.
_KNUCKLES = np.array([
    [0.110, 0.125, -0.050],   # thumb CMC: at the wrist, radial and slightly palmar
    [0.465, 0.120, 0.000],    # index MCP
    [0.475, 0.000, 0.000],    # middle MCP (on the x axis, by definition of the hand frame)
    [0.450, -0.105, -0.010],  # ring MCP
    [0.405, -0.195, -0.025],  # little MCP: the knuckle arch falls away towards the little finger
])

# Bone lengths, proximal to distal; the last one ends at the fingertip. The middle row sums with its knuckle
# offset to exactly 1.0, which is what makes ``hand_length`` the wrist-to-middle-fingertip distance.
_BONES = np.array([
    [0.250, 0.170, 0.150],    # thumb: metacarpal, proximal phalanx, distal phalanx
    [0.215, 0.130, 0.115],    # index
    [0.245, 0.150, 0.130],    # middle
    [0.225, 0.140, 0.125],    # ring
    [0.175, 0.105, 0.110],    # little
])

# Orientation of the thumb's first joint: with all thumb joints at zero the thumb lies beside the palm,
# pointing _THUMB_AZIMUTH away from the fingers towards the radial side, and its nail is rolled
# _THUMB_PRONATION away from the plane of the finger nails. The roll is what turns thumb flexion into a
# sweep across the palm and thumb abduction into lifting away from the palm.
_THUMB_AZIMUTH = np.radians(40.0)
_THUMB_ELEVATION = np.radians(-5.0)
_THUMB_PRONATION = np.radians(55.0)

# The ring and little fingers flex slightly towards the thumb (their knuckle axes are oblique), which is
# part of how the hand cups and how the thumb reaches the little finger.
_FINGER_SUPINATION = np.radians([0.0, 0.0, 6.0, 12.0])   # index, middle, ring, little

# Joint limits in degrees, per digit: [flex 0, flex 1, flex 2, abduction] as (lower, upper).
_LIMITS_DEG = np.array([
    [[-15, 50], [-10, 60], [-15, 80], [-15, 60]],    # thumb: CMC flex, MCP flex, IP flex, CMC palmar abduction
    [[-20, 90], [-5, 110], [-5, 80], [-20, 20]],     # index: MCP flex, PIP flex, DIP flex, MCP abduction
    [[-20, 90], [-5, 110], [-5, 80], [-20, 20]],     # middle
    [[-20, 90], [-5, 110], [-5, 80], [-20, 20]],     # ring
    [[-20, 90], [-5, 110], [-5, 80], [-25, 20]],     # little
], dtype=float)

DISTAL_COUPLING = np.array([1.0, 2 / 3, 2 / 3, 2 / 3, 2 / 3])
"""Flexion of the last joint as a fraction of the joint before it (thumb IP per MCP, finger DIP per PIP).

The last two joints of a digit share tendons and normally bend together; 2/3 is the textbook DIP/PIP ratio.
The glove has no sensor beyond bone 1, so this coupling is how the last joint is reconstructed.
"""


def _right_hand_joint_frames() -> np.ndarray:
    """Rest orientation of each digit's first joint in the hand frame, right hand: (5, 3, 3)."""
    frames = np.empty((5, 3, 3))
    frames[0] = rot_z(_THUMB_AZIMUTH) @ rot_y(-_THUMB_ELEVATION) @ rot_x(-_THUMB_PRONATION)
    frames[1:] = rot_x(_FINGER_SUPINATION)
    return frames


class HandModel:
    """Skeleton of one hand: where the joints are, how far they bend, and forward kinematics.

    Attributes (all in the hand frame of ``side``, metres and radians):
        base_pos: (5, 3) first joint of each digit.
        base_rot: (5, 3, 3) rest orientation of each digit's first joint; +x along the straight digit,
            +z towards its nail.
        lengths:  (5, 3) bone lengths.
        lower, upper: (5, 4) joint limits in the ``HumanHand.joints`` layout.
        mirror:   +1.0 for a right hand, -1.0 for a left hand (the sign of everything lateral).
    """

    def __init__(self, side: str, hand_length: float = DEFAULT_HAND_LENGTH) -> None:
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        self.side = side
        self.hand_length = float(hand_length)
        self.mirror = 1.0 if side == "right" else -1.0
        # A left hand is the right hand reflected in the xz plane: y flips for points, and a rotation R
        # becomes M R M with M = diag(1, -1, 1).
        flip = np.array([1.0, self.mirror, 1.0])
        self.base_pos = _KNUCKLES * self.hand_length * flip
        self.base_rot = _right_hand_joint_frames() * np.outer(flip, flip)
        self.lengths = _BONES * self.hand_length
        self.lower = np.radians(_LIMITS_DEG[:, :, 0])
        self.upper = np.radians(_LIMITS_DEG[:, :, 1])
        self._base_rot_t = np.ascontiguousarray(self.base_rot.transpose(0, 2, 1))
        # Total flexion of a fully curled digit, with the last joint following the coupling.
        last = np.minimum(self.upper[:, 2], DISTAL_COUPLING * self.upper[:, 1])
        self._full_curl = self.upper[:, 0] + self.upper[:, 1] + last

    def clamp(self, joints: np.ndarray) -> np.ndarray:
        """Joint values limited to the anatomical range."""
        return np.clip(joints, self.lower, self.upper)

    def couple(self, joints: np.ndarray) -> np.ndarray:
        """Copy of ``joints`` with the last joint of each digit set from the one before it."""
        out = np.array(joints, dtype=float)
        out[:, 2] = np.clip(DISTAL_COUPLING * out[:, 1], self.lower[:, 2], self.upper[:, 2])
        return out

    def points(self, joints: np.ndarray) -> np.ndarray:
        """Joint centres and fingertip of every digit: (5, 4, 3) as [joint 0, joint 1, joint 2, tip].

        After the abduction turn a digit bends in a plane, so the chain is a closed form: each bone's pitch
        is the running sum of the flexions, its forward and downward extent are a cosine and a sine.
        """
        joints = np.asarray(joints, dtype=float)
        pitch = np.cumsum(joints[:, :3], axis=1)
        reach = np.cumsum(self.lengths * np.cos(pitch), axis=1)
        drop = np.cumsum(self.lengths * np.sin(pitch), axis=1)
        yaw = self.mirror * joints[:, 3:4]
        local = np.stack([reach * np.cos(yaw), reach * np.sin(yaw), -drop], axis=-1)
        out = np.empty((5, 4, 3))
        out[:, 0] = self.base_pos
        out[:, 1:] = self.base_pos[:, None, :] + local @ self._base_rot_t
        return out

    def tips(self, joints: np.ndarray) -> np.ndarray:
        """Fingertip positions, (5, 3)."""
        return self.points(joints)[:, 3]

    def bone_frames(self, joints: np.ndarray) -> np.ndarray:
        """Orientation of every bone: (5, 3, 3, 3) indexed [digit, bone]; +x along the bone, +z to its dorsal side."""
        joints = np.asarray(joints, dtype=float)
        pitch = np.cumsum(joints[:, :3], axis=1)
        abduct = rot_z(self.mirror * joints[:, 3])
        return (self.base_rot @ abduct)[:, None] @ rot_y(pitch)

    def curl(self, joints: np.ndarray) -> np.ndarray:
        """Per-digit curl in [0, 1]: summed flexion as a fraction of a fully curled digit. Shape (5,)."""
        return np.clip(np.sum(np.asarray(joints)[:, :3], axis=1) / self._full_curl, 0.0, 1.0)
