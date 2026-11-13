"""Top-down pinch grasps for the Dex3 and the pick-and-place motion built from them.

The Dex3 has a thumb opposing two fingers 57 mm apart. A 45 mm part is pinched between the
thumb and ONE finger (the index), with the palm facing the counter: centred between both
fingers it wedges and is never released (noted by the challenge organisers; reproduced here).

Geometry is expressed for the palm frame of the challenge URDF: +x along the fingers, the
palm normal along -y on the left hand and +y on the right, so a top-down grasp is fully
described by one angle, the direction the fingers point in the horizontal plane.

This module is pure geometry. It is used by the scripted operator (which turns the palm path
into human wrist motion), by the fan-out (which needs to know the equivalent grasps of a
rotated part) and by tests that check a grasp plan against physics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kaiku.sim.scene import SHAPE_SYMMETRY, Scene
from kaiku.sim.twin import GRASP_IN_PALM

# Dex3 postures, LEFT hand order (thumb 0-2, middle 0-1, index 0-1), radians.
HAND_OPEN = np.array([0.0, -0.5, 1.5, -0.4, -0.4, -0.4, -0.4])       # wide claw, ready to grasp
HAND_CLOSED = np.array([0.0, -0.1, 1.2, -1.0, -0.6, -1.0, -0.6])     # pinch on a 45 mm part
HAND_RELEASE = np.array([0.0, -0.5, 1.5, -0.05, -0.05, -0.05, -0.05])  # fingers straight: lets go cleanly
# Natural direction of the fingers for each arm (forward, slightly inward), radians about +z.
NATURAL_DIR = {"left": np.radians(-25.0), "right": np.radians(25.0)}
# Finger directions relative to a triangle's apex that pinch it best: the index pad lies flat on
# one face while the thumb presses the opposite edge slightly off-centre, which locks the part
# against rotation. Measured in this twin: 98 % held at +-75 deg, 74 % at 60, 84 % at 0.
TRIANGLE_PHI = np.radians(75.0)


def left_to_right(hand_left: np.ndarray) -> np.ndarray:
    """A left-hand posture as the mirrored right-hand command (sign flip, index and middle swapped)."""
    return -np.asarray(hand_left, float)[[0, 1, 2, 5, 6, 3, 4]]


def hand14(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Two left-order postures -> the challenge's 14-vector (left hand, then mirrored right hand)."""
    return np.r_[left, left_to_right(right)]


def palm_rotation(side: str, finger_dir: float, tilt: float = 0.0) -> np.ndarray:
    """Palm orientation for a top-down grasp: fingers point along ``finger_dir`` (radians about
    +z), palm normal straight down. ``tilt`` pitches the fingertips down (radians)."""
    x = np.array([np.cos(finger_dir) * np.cos(tilt), np.sin(finger_dir) * np.cos(tilt), -np.sin(tilt)])
    back = np.array([0.0, 0.0, 1.0]) - x[2] * x          # back of the hand: up, perpendicular to the fingers
    back /= np.linalg.norm(back)
    y = back if side == "left" else -back                # palm normal is -y on the left hand, +y on the right
    return np.stack([x, y, np.cross(x, y)], axis=1)


def grasp_offset(side: str) -> np.ndarray:
    """Pinch point in the palm frame of ``side``."""
    return GRASP_IN_PALM * [1.0, 1.0 if side == "left" else -1.0, 1.0]


def palm_position(side: str, rot: np.ndarray, pinch_point: np.ndarray) -> np.ndarray:
    """Palm origin that puts the pinch point at ``pinch_point``."""
    return np.asarray(pinch_point, float) - rot @ grasp_offset(side)


def grasp_directions(shape: str, yaw: float) -> np.ndarray:
    """Finger directions (radians about +z) that pinch a part of this shape at this yaw across
    its 45 mm grasp width. Cylinder: any direction (returns an empty array)."""
    if shape == "cube":
        return yaw + np.arange(4) * np.pi / 2
    if shape == "triangle":
        apex = yaw + np.pi / 2
        turns = np.arange(3) * 2 * np.pi / 3
        return np.r_[apex + TRIANGLE_PHI + turns, apex - TRIANGLE_PHI + turns]
    return np.zeros(0)


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def best_direction(shape: str, yaw: float, side: str, prefer: float | None = None) -> float:
    """The valid finger direction closest to ``prefer`` (default: the arm's natural direction)."""
    prefer = NATURAL_DIR[side] if prefer is None else prefer
    cands = grasp_directions(shape, yaw)
    if len(cands) == 0:
        return float(prefer)
    return float(prefer + wrap(cands - prefer)[np.argmin(np.abs(wrap(cands - prefer)))])


def pinch_shift(shape: str, yaw: float, finger_dir: float, size: float) -> np.ndarray:
    """Offset (xy) from a part's centre to the midpoint of its extent along the finger direction.
    Zero for cube and cylinder; a triangle's centroid is not midway between a face and its apex."""
    if shape != "triangle":
        return np.zeros(2)
    r = 2.0 * size / 3.0                         # centroid -> vertex (altitude = size)
    v = yaw + np.pi / 2 + np.arange(3) * 2 * np.pi / 3
    mid = 0.5 * (r * np.cos(finger_dir - v).max() - r * np.cos(finger_dir + np.pi - v).max())
    return mid * np.array([np.cos(finger_dir), np.sin(finger_dir)])


def symmetry_delta(shape: str, yaw_from: float, yaw_to: float) -> float:
    """Smallest rotation about +z that maps a part at ``yaw_from`` onto the same part at
    ``yaw_to``, using the shape's rotational symmetry (0 for a cylinder)."""
    n = SHAPE_SYMMETRY[shape]
    if n == 0:
        return 0.0
    period = 2 * np.pi / n
    return float((yaw_to - yaw_from + period / 2) % period - period / 2)


@dataclass
class Waypoint:
    """Where the pinch point and palm should be at the end of a segment, and the hand posture."""

    label: str
    point: np.ndarray        # (3,) pinch point, pelvis frame
    rot: np.ndarray          # (3, 3) palm orientation
    hand: np.ndarray         # (7,) left-order posture for the working hand
    duration: float          # seconds to get there from the previous waypoint
    hold: float = 0.0        # seconds to stay after arriving


# Finger direction (left arm; mirrored for the right) in which each bin is easiest to reach.
DROP_DIR = {"side": np.radians(15.0), "center": np.radians(-10.0)}


def drop_direction(side: str, bin_name: str) -> float:
    sy = 1.0 if side == "left" else -1.0
    return sy * DROP_DIR["center" if bin_name == "center" else "side"]


def drop_point(scene: Scene, side: str, bin_name: str, slot: int = 0) -> np.ndarray:
    """Release point above a bin: over its near part, on the side of the arm that carries (the
    far corners are beyond arm's reach), cycling through a few spots so parts do not stack
    above the rim."""
    lo, hi = scene.bin_inner(bin_name)
    sy = 1.0 if side == "left" else -1.0
    c = scene.bin_centers[bin_name]
    x0 = lo[0] + 0.045
    y0 = c[1] + (sy * 0.03 if bin_name == "center" else -np.sign(c[1]) * 0.045)
    dx, dy = [(0.0, 0.0), (0.045, 0.0), (0.0, 0.05), (0.045, 0.05), (0.02, -0.03)][slot % 5]
    y = np.clip(y0 + dy * np.sign(c[1] if bin_name != "center" else -sy), lo[1] + 0.04, hi[1] - 0.04)
    z = scene.counter_z + scene.bin_height + scene.height / 2 + 0.045
    return np.array([x0 + dx, y, z])


def reachable_direction(ik, scene: Scene, side: str, shape: str, pos: np.ndarray, yaw: float,
                        tol: float = 0.004) -> tuple[float, float]:
    """Among the finger directions that pinch this part, the one nearest the arm's natural
    direction that the arm can actually reach (palm level, at grasp height and 8 cm above).
    Returns (direction, worst position error); the error exceeds ``tol`` if nothing is reachable."""
    nat = NATURAL_DIR[side]
    cands = grasp_directions(shape, yaw)
    cands = nat + wrap(cands - nat) if len(cands) else nat + np.radians([0.0, -20.0, 20.0, -40.0, 40.0])
    best = (float(cands[0]), np.inf)
    for f in sorted(cands, key=lambda a: abs(a - nat)):
        rot = palm_rotation(side, f)
        grasp = np.r_[pos[:2] + pinch_shift(shape, yaw, f, scene.size), scene.rest_z + 0.004]
        q, e_hi, r_hi = ik.solve_full(side, palm_position(side, rot, grasp + [0, 0, 0.085]), rot)
        q, e_lo, r_lo = ik.solve_full(side, palm_position(side, rot, grasp), rot, seed=q, iters=40)
        err = max(e_hi, e_lo) + 0.02 * max(r_hi, r_lo)
        if err < tol:
            return float(f), err
        if err < best[1]:
            best = (float(f), err)
    return best


def plan_pick_place(scene: Scene, side: str, shape: str, pos: np.ndarray, yaw: float, bin_name: str,
                    slot: int = 0, speed: float = 1.0, finger_dir: float | None = None) -> list[Waypoint]:
    """Palm path for one part: above -> down -> pinch -> lift -> over the bin -> release -> clear.
    ``speed`` scales every duration (1.0 = unhurried human pace)."""
    f = best_direction(shape, yaw, side) if finger_dir is None else finger_dir
    rot = palm_rotation(side, f)
    grasp = np.r_[pos[:2] + pinch_shift(shape, yaw, f, scene.size), scene.rest_z + 0.004]
    above = grasp + [0.0, 0.0, 0.085]
    drop = drop_point(scene, side, bin_name, slot)
    carry_z = max(above[2], drop[2]) + 0.03
    rot_drop = palm_rotation(side, drop_direction(side, bin_name))
    s = 1.0 / speed
    return [
        Waypoint("approach", above, rot, HAND_OPEN, 1.5 * s),
        Waypoint("descend", grasp, rot, HAND_OPEN, 0.9 * s, hold=0.1 * s),
        Waypoint("grasp", grasp, rot, HAND_CLOSED, 0.45 * s, hold=0.25 * s),
        Waypoint("lift", np.r_[grasp[:2], carry_z], rot, HAND_CLOSED, 0.9 * s),
        Waypoint("carry", np.r_[drop[:2], carry_z], rot_drop, HAND_CLOSED, 1.6 * s),
        Waypoint("lower", drop, rot_drop, HAND_CLOSED, 0.5 * s, hold=0.1 * s),
        Waypoint("release", drop, rot_drop, HAND_RELEASE, 0.4 * s, hold=0.3 * s),
        Waypoint("retreat", np.r_[drop[:2], carry_z + 0.02], rot_drop, HAND_OPEN, 0.6 * s),
    ]
