"""THE ASSUMPTIONS FILE: how a Rokoko Smartglove sits on the hand skeleton of ``kaiku.hand.model``.

Everything physical that we could not measure (we had no glove when this was written) is a constant in this
file and nowhere else. The solver (``solve.py``) and the forward model (``synth.py``) both read it through
``layout.py``, so they always agree with each other; whether they agree with a real glove depends on these
numbers. To correct them from a real recording, see "How to check" at each constant.

Each constant is tagged:
    [DOC]   stated in the Rokoko Device SDK docs (ref/rokoko-docs), quoted.
    [GUESS] our estimate; check against a recording.

Constants only. Values are for a RIGHT hand; ``layout.py`` mirrors them for a left hand, following
guides_hand-solving.txt: "A left hand is the mirror image of a right hand. Mirror the mounting offsets and
rest-pose angles across the hand's midline to switch sides."
"""
from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------------------------------------
# 1. Axes: the glove (hub) frame versus the kaiku hand frame
# ----------------------------------------------------------------------------------------------------------

# [DOC] guides_hand-solving.txt: "Values arrive in the glove's own right-handed sensor frame (+X right,
#       +Y toward the wrist, +Z down)."
# The kaiku hand frame is +x to the fingertips, +z out of the back of the hand, +y = z cross x. For a hand
# lying flat, palm down, that puts glove +X (right) on hand -y, glove +Y (to the wrist) on hand -x and glove
# +Z (down) on hand -z. The columns below are the glove axes written in the hand frame; the matrix turns
# glove-frame vectors into hand-frame vectors and is a proper rotation (half a turn about (1, -1, 0)).
#
# [GUESS] The same matrix is used for both gloves: the docs describe one frame ("+X right") and never a
#         per-side variant, and they insist the frame conversion is "a single proper rotation", never a
#         mirror. If the left glove turns out to use +X = left, only the "left" entry changes.
# [GUESS] The hub is mounted square on the back of the hand (no tilt between hub axes and hand axes).
#         Residual tilt is absorbed by the operator calibration in calibrate.py.
# How to check: lay a glove flat, palm down, fingers pointing north. The hub's NED quaternion should then
# solve to a hand whose +x points north and +z up (GloveSolver wrist_quat close to identity).
_GLOVE_AXES_IN_HAND = np.array([
    [0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
])
HAND_FROM_GLOVE = {"left": _GLOVE_AXES_IN_HAND, "right": _GLOVE_AXES_IN_HAND}

# [DOC] reference_products_smartgloves.txt: "The reference frames are defined such that when the glove is
#       lying flat, all coordinate axes are aligned."
# So a finger sensor's axes relate to the bone it is strapped to exactly as the hub's axes relate to the
# hand: the sensor's +Y points back along the bone, its +Z into the finger. layout.py therefore reuses
# HAND_FROM_GLOVE as "bone from sensor". (The solver reads bone direction and hinge axis separately, so a
# sensor that is rolled around its finger does not disturb the flexion it reports.)

# ----------------------------------------------------------------------------------------------------------
# 2. Where the hub sits
# ----------------------------------------------------------------------------------------------------------

# [DOC] reference_products_smartgloves.txt: "The hub is the central unit on the back of the hand (centered
#       at the circular Rokoko logo)". Finger and arm sensor positions are reported relative to this point.
# [GUESS] The numbers: the middle of the back of the hand, a little above the skin.
# How to check: with the hand flat, the solved knuckle-to-PIP direction of every finger should be level
# (mcp_flex near 0) and straight (abduction near 0). A constant flexion offset on all fingers means
# HUB_HEIGHT is off; fingers splaying in or out together means HUB_ALONG is off.
HUB_ALONG = 0.26
"""Hub centre along hand +x, as a fraction of hand length from the wrist joint."""
HUB_LATERAL = 0.0
"""Hub centre towards the thumb side, metres."""
HUB_HEIGHT = 0.016
"""Hub centre above the plane of the knuckles (hand +z), metres."""

# ----------------------------------------------------------------------------------------------------------
# 3. Where the finger sensors sit
# ----------------------------------------------------------------------------------------------------------

# [DOC] reference_products_smartgloves.txt: "The finger sensors sit on the back of the fingers, on the
#       middle segment of each of the four fingers and on the proximal segment of the thumb."
# In the three-bone chain of model.py that is bone 1 for every digit (thumb: metacarpal, PROXIMAL, distal;
# fingers: proximal, MIDDLE, distal), which is why one index covers all five.
SENSOR_BONE = 1

# [GUESS] Position of each sensor's origin on its bone, rows in FINGERS order (thumb, index, middle, ring,
#         little). Halfway along the bone, centred, on top of the finger: half a finger thickness plus half
#         a sensor housing above the bone axis.
# How to check: the static_data OBB entries give the housing size; the solved PIP joint is the sensor
# position minus this offset, so with a straight finger the solved knuckle-to-PIP distance should equal the
# model's proximal bone length. Too long or too short means SENSOR_ALONG is off.
SENSOR_ALONG = np.array([0.50, 0.50, 0.50, 0.50, 0.50])
"""Sensor origin along its bone, as a fraction of the bone length from the bone's proximal joint."""
SENSOR_LATERAL = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
"""Sensor origin off the bone axis towards the thumb side, metres."""
SENSOR_HEIGHT = np.array([0.013, 0.010, 0.010, 0.010, 0.009])
"""Sensor origin above the bone axis (dorsal), metres."""

# ----------------------------------------------------------------------------------------------------------
# 4. Where the arm sensor sits (forward model only)
# ----------------------------------------------------------------------------------------------------------

# [DOC] reference_products_smartgloves.txt: "The arm sensor is located on the forearm near the wrist."
# The solver never needs this: it gets the arm sensor's pose relative to the hub from the glove itself
# every frame (the wrist bends between them). Only synth.py uses it, to invent that measurement.
# [GUESS] On top of the forearm, a few centimetres behind the wrist joint, axes aligned with the hub's when
#         the wrist is straight. Given in the forearm frame, which coincides with the hand frame for a
#         straight wrist.
ARM_SENSOR_POSITION = np.array([-0.060, 0.0, 0.022])

# ----------------------------------------------------------------------------------------------------------
# 5. World frames
# ----------------------------------------------------------------------------------------------------------

# kaiku wants a tracking frame with +z up, so that the operator calibration is a pure yaw.
#
# [GUESS] The Coil Pro frame has +Z up when the coil stands as intended. The docs only say Z is the coil's
#         vertical axis (reference_products_smartgloves.txt: "an ambiguity in the sign of the z-axis
#         coordinate ... the vertical (z-axis) position may be inverted" before emf_flip_decided) and show
#         the frame in two images we do not have.
# How to check: raise a gloved hand. If the solved wrist_pos z goes DOWN, the coil frame is Z-down; replace
# the identity below by np.diag([1.0, -1.0, -1.0]) (half a turn about X, a proper rotation).
TRACKING_FROM_COIL = np.eye(3)

# [DOC] reference_products_smartgloves.txt: hub "ORIENTATION | hub | LTP_NED"; reference_rgmp.txt:
#       "LTP_NED: Local Tangent Plane (North, East, Down). Right-handed."
# Half a turn about North makes it North-West-Up, a z-up frame. Its heading is magnetic north, which is
# unrelated to the coil's heading, so orientations from the two sources differ by an unknown yaw.
TRACKING_FROM_NED = np.diag([1.0, -1.0, -1.0])
