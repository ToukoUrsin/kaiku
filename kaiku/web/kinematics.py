"""Reference forward kinematics over the exported kinematic tree.

This is the same algorithm the browser runs (``static/js/fk.js``), written with numpy so the tests
can hold it against MuJoCo's ``xpos`` / ``xquat``. Quaternions are (w, x, y, z).
"""
from __future__ import annotations

import numpy as np

JNT_FREE, JNT_BALL, JNT_SLIDE, JNT_HINGE = 0, 1, 2, 3


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, u = q[0], q[1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _unit(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def fk(tree: dict, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Body frames in the world for one environment: ``(xpos (nbody, 3), xquat (nbody, 4))``."""
    nbody = tree["nbody"]
    parent = tree["parent"]
    bpos = np.asarray(tree["pos"], dtype=np.float64).reshape(nbody, 3)
    bquat = np.asarray(tree["quat"], dtype=np.float64).reshape(nbody, 4)
    jpos = np.asarray(tree["jpos"], dtype=np.float64).reshape(-1, 3)
    jaxis = np.asarray(tree["jaxis"], dtype=np.float64).reshape(-1, 3)
    jtype, jqadr = tree["jtype"], tree["jqadr"]
    qpos0 = np.asarray(tree["qpos0"], dtype=np.float64)
    q = np.asarray(qpos, dtype=np.float64)

    xpos = np.zeros((nbody, 3))
    xquat = np.zeros((nbody, 4))
    xquat[0, 0] = 1.0
    for b in range(1, nbody):
        j0, nj = tree["jntadr"][b], tree["jntnum"][b]
        if nj == 1 and jtype[j0] == JNT_FREE:
            a = jqadr[j0]
            pos, quat = q[a:a + 3].copy(), _unit(q[a + 3:a + 7])
        else:
            p = parent[b]
            pos = xpos[p] + quat_rot(xquat[p], bpos[b])
            quat = quat_mul(xquat[p], bquat[b])
            for j in range(j0, j0 + nj):
                a, t = jqadr[j], jtype[j]
                if t == JNT_SLIDE:
                    pos = pos + quat_rot(quat, jaxis[j]) * (q[a] - qpos0[a])
                    continue
                if t == JNT_HINGE:
                    half = 0.5 * (q[a] - qpos0[a])
                    qloc = np.concatenate([[np.cos(half)], np.sin(half) * jaxis[j]])
                else:  # ball
                    qloc = _unit(q[a:a + 4])
                anchor = pos + quat_rot(quat, jpos[j])
                quat = quat_mul(quat, qloc)
                pos = anchor - quat_rot(quat, jpos[j])
            quat = _unit(quat)
        xpos[b], xquat[b] = pos, quat
    return xpos, xquat
