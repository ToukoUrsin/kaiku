"""Hands: raw glove frames in, solved human hands and calibrated robot commands out.

    GloveFrame --GloveSolver.solve--> HumanHand --Teleop.update--> TeleopCommand

``synth`` provides the same glove frames without hardware: a forward model of the glove and a scripted
operator. Every physical assumption about the glove lives in ``glove_model.py``.
"""
from kaiku.hand.calibrate import (DEFAULT_REACH_SCALE, DEFAULT_ROBOT_HOME, Calibration, FilterParams, Teleop,
                                  TeleopFilters)
from kaiku.hand.model import DEFAULT_HAND_LENGTH, HandModel
from kaiku.hand.solve import GloveSolver
from kaiku.hand.synth import (DEFAULT_HOME, SCRIPTS, GloveNoise, Grip, Open, Pinch, ReachWaypoints, Relax,
                              ScriptedOperator, Step, hand_to_glove_frame, idle_script, open_close_script,
                              pinch_script, reach_script)

__all__ = [
    "DEFAULT_HAND_LENGTH", "DEFAULT_HOME", "DEFAULT_REACH_SCALE", "DEFAULT_ROBOT_HOME", "SCRIPTS",
    "Calibration", "FilterParams", "GloveNoise", "GloveSolver", "Grip", "HandModel", "Open", "Pinch",
    "ReachWaypoints", "Relax", "ScriptedOperator", "Step", "Teleop", "TeleopFilters",
    "hand_to_glove_frame", "idle_script", "open_close_script", "pinch_script", "reach_script",
]
