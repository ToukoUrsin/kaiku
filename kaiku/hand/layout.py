"""The glove assumptions of ``glove_model.py`` resolved for one hand: mirrored to its side, scaled to its size.

Both the solver and the forward model build on this one object, so a corrected constant reaches both.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kaiku.hand import glove_model as gm
from kaiku.hand.geometry import Pose, mat_to_quat
from kaiku.hand.model import HandModel


@dataclass(frozen=True)
class GloveLayout:
    """Where the glove's parts are on one hand. Positions in metres."""
    hand_from_glove: np.ndarray     # (3, 3) turns glove-axis vectors into hand-frame vectors (also bone from sensor)
    hand_from_hub: Pose             # hub frame in the hand frame
    sensor_offset: np.ndarray       # (5, 3) origin of each finger sensor in the frame of the bone it sits on
    arm_sensor_pos: np.ndarray      # (3,) arm sensor origin in the forearm frame


def glove_layout(model: HandModel) -> GloveLayout:
    """Layout of the glove worn on the hand described by ``model``."""
    flip = np.array([1.0, model.mirror, 1.0])
    rotation = gm.HAND_FROM_GLOVE[model.side]
    hub_pos = np.array([gm.HUB_ALONG * model.hand_length, gm.HUB_LATERAL, gm.HUB_HEIGHT]) * flip
    along = gm.SENSOR_ALONG * model.lengths[:, gm.SENSOR_BONE]
    return GloveLayout(
        hand_from_glove=rotation,
        hand_from_hub=(hub_pos, mat_to_quat(rotation)),
        sensor_offset=np.stack([along, gm.SENSOR_LATERAL, gm.SENSOR_HEIGHT], axis=1) * flip,
        arm_sensor_pos=gm.ARM_SENSOR_POSITION * flip,
    )
