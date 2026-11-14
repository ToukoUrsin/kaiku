"""Rotation helpers for the hand module.

Quaternions are unit, Hamilton, (w, x, y, z), as everywhere in kaiku. A quaternion ``q_a_b`` rotates vectors
expressed in frame b into frame a, matching the pose convention of ``kaiku.types`` (``T_a_b`` maps points in b
to a). A pose is a ``(position, quaternion)`` tuple.

The scalar functions take one quaternion each and are written out component by component: at 83 Hz with two
hands the call overhead of general array code would dominate. Only ``quat_to_mat`` and the ``rot_*`` builders
are batched, because the solver and the forward kinematics process all five digits at once.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

Pose = tuple[np.ndarray, np.ndarray]

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def _floats(a: ArrayLike) -> list[float]:
    """Components as Python floats, whose arithmetic is several times faster than NumPy scalars'."""
    return np.asarray(a, dtype=float).tolist()


def _quat_to_mat_table() -> np.ndarray:
    """Table K, (16, 9), such that ``outer(q, q).ravel() @ K`` is the rotation matrix of unit q, row by row.

    Every entry of the matrix is a sum of products of two quaternion components (using w2+x2+y2+z2 = 1 on
    the diagonal), so the whole conversion is one outer product and one matrix product, for any batch.
    """
    w, x, y, z = range(4)
    entries = [
        [(1, w, w), (1, x, x), (-1, y, y), (-1, z, z)], [(2, x, y), (-2, w, z)], [(2, x, z), (2, w, y)],
        [(2, x, y), (2, w, z)], [(1, w, w), (-1, x, x), (1, y, y), (-1, z, z)], [(2, y, z), (-2, w, x)],
        [(2, x, z), (-2, w, y)], [(2, y, z), (2, w, x)], [(1, w, w), (-1, x, x), (-1, y, y), (1, z, z)],
    ]
    table = np.zeros((16, 9))
    for entry, products in enumerate(entries):
        for coefficient, i, j in products:
            table[4 * i + j, entry] = coefficient
    return table


_QUAT_TO_MAT = _quat_to_mat_table()


def quat_normalize(q: ArrayLike) -> np.ndarray:
    """Scale a quaternion to unit length."""
    q = np.asarray(q, dtype=float)
    return q / math.sqrt(float(q @ q))


def quat_conj(q: ArrayLike) -> np.ndarray:
    """Conjugate, which is the inverse rotation for a unit quaternion."""
    w, x, y, z = _floats(q)
    return np.array([w, -x, -y, -z])


def quat_mul(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Hamilton product ``a * b``: the rotation b followed by the rotation a."""
    aw, ax, ay, az = _floats(a)
    bw, bx, by, bz = _floats(b)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_rotate(q: ArrayLike, v: ArrayLike) -> np.ndarray:
    """Rotate vector v by quaternion q."""
    w, x, y, z = _floats(q)
    vx, vy, vz = _floats(v)
    # v + 2 w (u x v) + 2 u x (u x v), with u the vector part of q.
    cx, cy, cz = y * vz - z * vy, z * vx - x * vz, x * vy - y * vx
    return np.array([
        vx + 2.0 * (w * cx + y * cz - z * cy),
        vy + 2.0 * (w * cy + z * cx - x * cz),
        vz + 2.0 * (w * cz + x * cy - y * cx),
    ])


def quat_to_mat(q: ArrayLike) -> np.ndarray:
    """Rotation matrices of unit quaternions: ``(..., 4) -> (..., 3, 3)``."""
    q = np.asarray(q, dtype=float)
    outer = q[..., :, None] * q[..., None, :]
    return (outer.reshape(q.shape[:-1] + (16,)) @ _QUAT_TO_MAT).reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(m: ArrayLike) -> np.ndarray:
    """Unit quaternion (w >= 0) of one proper 3x3 rotation matrix.

    Uses Shepperd's method: the largest of w, x, y, z is taken from the diagonal so the division is always by a
    number of magnitude >= 1.
    """
    (m00, m01, m02), (m10, m11, m12), (m20, m21, m22) = np.asarray(m, dtype=float).tolist()
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 2.0 * math.sqrt(1.0 + trace)
        q = (0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
    elif m00 >= m11 and m00 >= m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        q = ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    elif m11 >= m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        q = ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        q = ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
    out = np.array(q)
    return -out if out[0] < 0.0 else out


def quat_from_rotvec(rotvec: ArrayLike) -> np.ndarray:
    """Quaternion of a rotation vector (axis times angle, radians)."""
    rx, ry, rz = _floats(rotvec)
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    # sin(a/2)/a tends to 1/2; the series keeps tiny tremor rotations exact.
    k = 0.5 - angle * angle / 48.0 if angle < 1e-4 else math.sin(0.5 * angle) / angle
    return np.array([math.cos(0.5 * angle), k * rx, k * ry, k * rz])


def quat_from_yaw(yaw: float) -> np.ndarray:
    """Rotation by ``yaw`` radians about +z."""
    return np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])


def quat_yaw(q: ArrayLike) -> float:
    """Heading of the rotated +x axis: its angle about +z, measured from +x, in radians."""
    w, x, y, z = _floats(q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_to_rotvec(q: ArrayLike) -> np.ndarray:
    """Rotation vector (axis times angle in [0, pi], radians) of a unit quaternion."""
    w, x, y, z = _floats(q)
    if w < 0.0:  # q and -q are the same rotation; take the short way round
        w, x, y, z = -w, -x, -y, -z
    n = math.sqrt(x * x + y * y + z * z)
    k = 2.0 * math.atan2(n, w) / n if n > 1e-12 else 2.0
    return np.array([k * x, k * y, k * z])


def quat_slerp(a: ArrayLike, b: ArrayLike, s: float) -> np.ndarray:
    """Constant-speed interpolation from a (s=0) to b (s=1) along the shorter arc."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dot = float(a @ b)
    if dot < 0.0:  # q and -q are the same rotation; pick the nearer one
        b, dot = -b, -dot
    if dot > 0.9995:  # nearly parallel: the sine formula loses precision, a chord is exact enough
        return quat_normalize(a + s * (b - a))
    theta = math.acos(dot)
    return (math.sin((1.0 - s) * theta) * a + math.sin(s * theta) * b) / math.sin(theta)


def rot_x(angle: ArrayLike) -> np.ndarray:
    """Rotation matrices about +x: ``(...) -> (..., 3, 3)``. A positive angle swings +y towards +z."""
    angle = np.asarray(angle, dtype=float)
    c, s = np.cos(angle), np.sin(angle)
    m = np.zeros(angle.shape + (3, 3))
    m[..., 0, 0] = 1.0
    m[..., 1, 1], m[..., 1, 2] = c, -s
    m[..., 2, 1], m[..., 2, 2] = s, c
    return m


def rot_y(angle: ArrayLike) -> np.ndarray:
    """Rotation matrices about +y: ``(...) -> (..., 3, 3)``. A positive angle swings +x towards -z."""
    angle = np.asarray(angle, dtype=float)
    c, s = np.cos(angle), np.sin(angle)
    m = np.zeros(angle.shape + (3, 3))
    m[..., 0, 0], m[..., 0, 2] = c, s
    m[..., 1, 1] = 1.0
    m[..., 2, 0], m[..., 2, 2] = -s, c
    return m


def rot_z(angle: ArrayLike) -> np.ndarray:
    """Rotation matrices about +z: ``(...) -> (..., 3, 3)``. A positive angle swings +x towards +y."""
    angle = np.asarray(angle, dtype=float)
    c, s = np.cos(angle), np.sin(angle)
    m = np.zeros(angle.shape + (3, 3))
    m[..., 0, 0], m[..., 0, 1] = c, -s
    m[..., 1, 0], m[..., 1, 1] = s, c
    m[..., 2, 2] = 1.0
    return m


def pose_mul(a: Pose, b: Pose) -> Pose:
    """Compose poses: ``T_a_c = T_a_b * T_b_c``."""
    return a[0] + quat_rotate(a[1], b[0]), quat_mul(a[1], b[1])


def pose_inv(a: Pose) -> Pose:
    """Invert a pose: ``T_b_a`` from ``T_a_b``."""
    inv = quat_conj(a[1])
    return -quat_rotate(inv, a[0]), inv
