"""Inverse model: one raw ``GloveFrame`` in, one solved ``HumanHand`` out.

Fingers. Each digit carries one sensor, on bone 1 of its three-bone chain (see ``model.py``). Its pose
relative to the hub gives two things:

  * its position, minus the mounting offset, is the centre of joint 1 (PIP, or the thumb's MCP). The line
    from the knuckle to that point is bone 0, which fixes the first joint's flexion and abduction;
  * its orientation is the orientation of bone 1, which fixes the total flexion up to that bone. Joint 1 is
    the total minus the first joint's share.

The last joint has no sensor and follows joint 1 through the anatomical coupling. This is the solve Rokoko
describes in guides_hand-solving.txt (aim the parent bone at a target derived from the sensor, set the
driven bone from the sensor orientation, hinge, clamp), in closed form and for all five digits at once.

Wrist. With a Coil Pro the forearm sensor is known in both the coil frame and the hub frame, which chains to
an absolute hand pose. Without one, only the hub's orientation against gravity and magnetic north is known.
"""
from __future__ import annotations

import numpy as np

from kaiku.hand import glove_model as gm
from kaiku.hand.geometry import (Pose, mat_to_quat, pose_inv, pose_mul, quat_mul, quat_normalize,
                                 quat_to_mat)
from kaiku.hand.layout import glove_layout
from kaiku.hand.model import DEFAULT_HAND_LENGTH, HandModel
from kaiku.types import FINGERS, GloveFrame, HumanHand

ABDUCTION_ORIENTATION_WEIGHT = 0.25
"""How much the sensor's hinge axis counts against the knuckle-to-joint direction when estimating abduction.

Both measure the same angle. The direction (from EMF position) is the better one while the finger is
extended but says nothing once the first joint is bent 90 degrees, because bone 0 then points straight down
whatever the abduction. The hinge axis (from orientation) works at any flexion. Their direction vectors are
summed with this weight, so position leads when it is informative and orientation takes over smoothly as
it fades.
"""


class GloveSolver:
    """Solves one glove. Keep one instance per glove: it remembers the last good value of each digit.

    Args:
        side: "left" or "right"; frames from the other glove are rejected.
        hand_length: wrist joint to middle fingertip of the operator, metres.
        require_flip_decided: only trust the Coil Pro position once the glove has resolved its vertical
            ambiguity (``emf_flip_decided``). Until then the docs warn that z "may be inverted", so the hand
            is reported without a position rather than mirrored through the floor.
    """

    def __init__(self, side: str, hand_length: float = DEFAULT_HAND_LENGTH, *,
                 require_flip_decided: bool = True) -> None:
        self.model = HandModel(side, hand_length)
        self.require_flip_decided = require_flip_decided
        layout = glove_layout(self.model)
        axes = layout.hand_from_glove
        base_rot_t = self.model.base_rot.transpose(0, 2, 1)
        bone0 = self.model.lengths[:, 0, None, None]
        # A sensor's axes are the glove axes, so the bone it sits on is oriented G R_hub_sensor G^T in the
        # hand frame. On a quaternion that conjugation simply rotates the vector part by G.
        self._bone_quat_map = np.eye(4)
        self._bone_quat_map[1:, 1:] = axes.T
        # Affine map taking a sensor position in the hub frame into its digit's first-joint frame (+x along
        # the straight digit, +z dorsal), in units of that digit's bone-0 length.
        self._to_joint_frame = base_rot_t @ axes / bone0
        self._hub_in_joint_frame = base_rot_t @ (layout.hand_from_hub[0] - self.model.base_pos)[:, :, None] / bone0
        self._sensor_offset = layout.sensor_offset[:, :, None] / bone0
        self._base_rot_t = base_rot_t.copy()
        self._hub_from_hand = pose_inv(layout.hand_from_hub)
        self._tracking_from_coil: Pose = (np.zeros(3), mat_to_quat(gm.TRACKING_FROM_COIL))
        self._tracking_from_ned = mat_to_quat(gm.TRACKING_FROM_NED)
        # Fallbacks for sensors that have not delivered yet: a flat hand, level and unrotated.
        self._joints = np.zeros((5, 4))
        self._wrist_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def solve(self, frame: GloveFrame) -> HumanHand:
        """Solve one frame. A digit whose sensor is missing or invalid keeps its previous joint values."""
        if frame.side != self.model.side:
            raise ValueError(f"{self.model.side} solver got a {frame.side} glove frame")
        pos, quat, valid = _stack_fingers(frame)
        if valid.any():
            self._joints[valid] = self._finger_joints(pos, quat)[valid]
        joints = self._joints.copy()
        tips = self.model.tips(joints)
        wrist_pos, wrist_quat, has_position = self._wrist(frame)
        return HumanHand(
            side=frame.side,
            t=frame.t_recv,
            wrist_pos=wrist_pos,
            wrist_quat=wrist_quat,
            has_position=has_position,
            tips=tips,
            joints=joints,
            curl=self.model.curl(joints),
            pinch=np.linalg.norm(tips[1:] - tips[0], axis=1),
        )

    def _finger_joints(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """Joint values (5, 4) from the finger sensor positions (5, 3) and quaternions (5, 4) in the hub frame."""
        model = self.model
        # Bone 1 and the centre of joint 1, both in the digit's first-joint frame. There flexion is a pitch
        # of +x towards -z and abduction a yaw about z, so the angles can be read off directly.
        bone = self._base_rot_t @ quat_to_mat(quat @ self._bone_quat_map)
        sensor = self._to_joint_frame @ pos[:, :, None] + self._hub_in_joint_frame
        joint1 = (sensor - bone @ self._sensor_offset)[:, :, 0]
        along, hinge = bone[:, :, 0], bone[:, :, 1]

        w = ABDUCTION_ORIENTATION_WEIGHT
        yaw = np.arctan2(joint1[:, 1] - w * hinge[:, 0], joint1[:, 0] + w * hinge[:, 1])
        c, s = np.cos(yaw), np.sin(yaw)
        flex0 = np.arctan2(-joint1[:, 2], joint1[:, 0] * c + joint1[:, 1] * s)
        total = np.arctan2(-along[:, 2], along[:, 0] * c + along[:, 1] * s)
        # A curled finger's bone 1 can be pitched past 180 degrees in total; unwrap joint 1 into (-90, 270).
        flex1 = (total - flex0 + 0.5 * np.pi) % (2.0 * np.pi) - 0.5 * np.pi

        joints = np.zeros((5, 4))
        joints[:, 0], joints[:, 1], joints[:, 3] = flex0, flex1, model.mirror * yaw
        return model.couple(model.clamp(joints))

    def _wrist(self, frame: GloveFrame) -> tuple[np.ndarray, np.ndarray, bool]:
        """Hand pose in the tracking frame: ``(position, quaternion, has_position)``."""
        arm_hub, arm_coil = frame.arm_in_hub, frame.arm_in_coil
        flip_ok = frame.coil_flip_decided or not self.require_flip_decided
        if (frame.coil_present and flip_ok and arm_hub is not None and arm_coil is not None
                and arm_hub.valid and arm_coil.valid):
            # T_coil_hand = T_coil_arm * inverse(T_hub_arm) * T_hub_hand. The wrist may bend freely between
            # the arm sensor and the hub: both poses are measured in the same instant, so it cancels.
            coil_from_hub = pose_mul((arm_coil.pos, arm_coil.quat), pose_inv((arm_hub.pos, arm_hub.quat)))
            pos, quat = pose_mul(self._tracking_from_coil, pose_mul(coil_from_hub, self._hub_from_hand))
            if np.isfinite(pos).all() and _is_rotation(quat):
                self._wrist_quat = quat_normalize(quat)
                return pos, self._wrist_quat, True
        hub = np.asarray(frame.hub_quat_ned, dtype=float)
        if _is_rotation(hub):
            self._wrist_quat = quat_normalize(quat_mul(self._tracking_from_ned, quat_mul(hub, self._hub_from_hand[1])))
        return np.zeros(3), self._wrist_quat, False


def _is_rotation(quat: np.ndarray) -> bool:
    """True for a quaternion of roughly unit length; False for the zeros or NaNs of a sensor without data."""
    return 0.5 < float(quat @ quat) < 2.0


def _stack_fingers(frame: GloveFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finger sensor positions (5, 3), unit quaternions (5, 4) and a validity mask (5,), in FINGERS order.

    A sensor counts as valid when it is present, flagged valid and carries finite numbers with a real
    quaternion. Rows of the others hold an identity pose so the vectorised solve stays finite.
    """
    pos = np.zeros((5, 3))
    quat = np.zeros((5, 4))
    valid = np.zeros(5, dtype=bool)
    for i, name in enumerate(FINGERS):
        sensor = frame.fingers.get(name)
        if sensor is not None and sensor.valid:
            pos[i], quat[i], valid[i] = sensor.pos, sensor.quat, True
    norm = np.linalg.norm(quat, axis=1)
    valid &= np.isfinite(pos).all(axis=1) & (norm > 0.5) & (norm < 2.0)
    if valid.all():
        return pos, quat / norm[:, None], valid
    pos[~valid] = 0.0
    quat[~valid] = (1.0, 0.0, 0.0, 0.0)
    return pos, quat / np.linalg.norm(quat, axis=1, keepdims=True), valid
