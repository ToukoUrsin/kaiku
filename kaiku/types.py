"""Shared data types. Every module talks to the others through these.

Conventions (all modules):
  * SI units: metres, radians, seconds.
  * Quaternions are (w, x, y, z) everywhere in kaiku. RGMP on the wire is (x, y, z, w); kaiku.rgmp converts.
  * A pose "T_a_b" maps points in frame b to frame a (same as RGMP: target_frame -> reference_frame).
  * Robot / world frame = the G1 pelvis frame: +x forward, +y left, +z up.
  * Hand frame (same definition for the human hand and both sides): origin at the wrist joint,
    +x from the wrist towards the middle fingertip, +z out of the BACK of the hand (dorsal),
    +y = z cross x. So a flat, palm-down hand pointing forward has its axes parallel to the world axes.
    On a RIGHT hand +y points to the thumb side; on a LEFT hand the thumb is on -y.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "little")
SIDES = ("left", "right")


@dataclass
class SensorPose:
    """Pose of one glove sensor."""
    pos: np.ndarray            # (3,) m
    quat: np.ndarray           # (4,) wxyz
    valid: bool = True


@dataclass
class GloveFrame:
    """One decoded sample from one Smartglove (the RGMP `default` group), nothing solved yet.

    Frames are Rokoko's: the glove sensor frame is right-handed with +X right, +Y towards the wrist,
    +Z down when the glove lies flat palm-down (see ref/rokoko-docs).
    """
    side: str                                   # "left" | "right"
    device_id: int
    t_us: int                                   # device timestamp, microseconds
    t_recv: float                               # host monotonic seconds when the frame arrived
    hub_quat_ned: np.ndarray                    # (4,) wxyz, hub orientation in LTP_NED
    fingers: dict[str, SensorPose] = field(default_factory=dict)   # finger sensor pose in the hub frame
    arm_in_hub: SensorPose | None = None        # forearm sensor pose in the hub frame
    arm_in_coil: SensorPose | None = None       # forearm sensor pose in the Coil Pro frame
    coil_present: bool = False                  # is_coil_pro_present flag
    coil_flip_decided: bool = False             # emf_flip_decided flag
    heading_reliable: bool = False              # is_heading_reliable flag (Smartgloves II)


@dataclass
class HumanHand:
    """A solved human hand, in the hand frame defined at the top of this file."""
    side: str
    t: float                                    # host monotonic seconds
    # Wrist pose in the tracking frame (Coil Pro frame when a coil is tracking, else position is zeros
    # and only the orientation is meaningful).
    wrist_pos: np.ndarray                       # (3,)
    wrist_quat: np.ndarray                      # (4,) wxyz, hand frame in the tracking frame
    has_position: bool
    # Fingertip positions in the hand frame, metres, for FINGERS in order. Shape (5, 3).
    tips: np.ndarray
    # Flexion per finger, radians, 0 = straight, positive = curling towards the palm.
    # Columns: [mcp_flex, pip_flex, dip_flex, mcp_abduction]. For the thumb: [cmc_flex, mcp_flex, ip_flex, cmc_abduction].
    # Shape (5, 4).
    joints: np.ndarray
    # Convenience scalars in [0, 1]: 0 = open, 1 = fully curled. Shape (5,).
    curl: np.ndarray
    # Thumb tip to each other fingertip distance, metres. Shape (4,) for index, middle, ring, little.
    pinch: np.ndarray
    # The same distances as an outside solver measured them (Rokoko's rkk-hand-solver), when one is running.
    # They then take the place of the model's own all the way to the robot. None: there is no such solver.
    gaps: np.ndarray | None = None


@dataclass
class TeleopCommand:
    """What the operator is asking for at one instant, already calibrated into the robot frame."""
    t: float
    # Per side; None when that hand is not tracked.
    wrist_pos: dict[str, np.ndarray | None]     # (3,) in the pelvis frame
    wrist_quat: dict[str, np.ndarray | None]    # (4,) wxyz, hand frame in the pelvis frame
    hands: dict[str, HumanHand | None]
    clutch: bool = False                        # True while the operator is re-centring (robot holds)
