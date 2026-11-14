"""Wire format of the live state stream (server -> browser).

Mirrored by ``static/js/protocol.js``; ``tests/test_web_protocol.py`` keeps the two in step.

Every joint coordinate is sent as one int16 (``qpos ~= int16 * scale[j]``), see :class:`Quantizer`.
A frame is either a KEY frame (the whole int16 table) or a DELTA against the last frame *that
client* was sent, so a client that is skipped for a while simply gets one bigger delta.

Frame layout, little endian::

    u8  magic (0x4B)      u8  kind (1 key, 2 delta)     u8  flags (bit0: env_state present)    u8  0
    u32 seq               u32 base_seq (delta only)     u32 t_ms (server clock, wraps)
    u16 n_envs            u16 nq
    [u8 env_state[n_envs]]                      if flags & 1
    key:    i16 q[n_envs * nq]                  (preceded by one pad byte if needed for alignment)
    delta:  u8 rowmask[ceil(n_envs / 8)]        bit e set: environment e has changes
            u8 colmask[ceil(nq / 8)]            one per changed environment, bit j: coordinate j changed
            zigzag LEB128 varints               one per set bit, row major: new - base
"""
from __future__ import annotations

import struct

import numpy as np

MAGIC = 0x4B
KIND_KEY = 1
KIND_DELTA = 2
FLAG_STATE = 1
HEADER = struct.Struct("<BBBBIIIHH")
HEADER_SIZE = HEADER.size  # 20

# Client -> server binary messages.
ACK = struct.Struct("<BI")  # (1, seq)
MSG_ACK = 1

# Joint types, same numbering as MuJoCo's mjtJoint.
JNT_FREE, JNT_BALL, JNT_SLIDE, JNT_HINGE = 0, 1, 2, 3

# Resolution of one int16 step. Fine enough to be invisible, coarse enough that physics noise on a
# resting joint does not change the value every frame.
ANGLE_STEP = 2.5e-4   # rad   (+-8.19 rad)
SLIDE_STEP = 1.0e-4   # m     (+-3.27 m)
POS_STEP = 2.5e-4     # m     (+-8.19 m, free joint position)
QUAT_STEP = 1.0 / 32767.0
I16_MAX = 32767


class Quantizer:
    """Maps float qpos to int16 and back. ``scale`` is exported to the browser."""

    def __init__(self, scale: np.ndarray, wrap: np.ndarray | None = None):
        self.scale = np.asarray(scale, dtype=np.float64).copy()
        self.inv = 1.0 / self.scale
        self.nq = int(self.scale.shape[0])
        wrap = np.zeros(self.nq, bool) if wrap is None else np.asarray(wrap, dtype=bool)
        self._wrap_idx = np.flatnonzero(wrap)

    @classmethod
    def from_joints(cls, nq: int, jnt_type, jnt_qposadr, jnt_limited, jnt_range) -> "Quantizer":
        scale = np.full(nq, ANGLE_STEP)
        wrap = np.zeros(nq, bool)
        for j in range(len(jnt_type)):
            a, t = int(jnt_qposadr[j]), int(jnt_type[j])
            if t == JNT_FREE:
                scale[a:a + 3] = POS_STEP
                scale[a + 3:a + 7] = QUAT_STEP
            elif t == JNT_BALL:
                scale[a:a + 4] = QUAT_STEP
            else:
                step = SLIDE_STEP if t == JNT_SLIDE else ANGLE_STEP
                if jnt_limited[j]:
                    reach = float(np.max(np.abs(jnt_range[j]))) * 1.05
                    step = max(step, reach / (I16_MAX - 1))
                elif t == JNT_HINGE:
                    wrap[a] = True      # unlimited hinge: send the angle modulo 2 pi
                scale[a] = step
        return cls(scale, wrap)

    def quantize(self, qpos: np.ndarray) -> np.ndarray:
        """(n_envs, nq) float -> (n_envs, nq) int16."""
        q = np.asarray(qpos, dtype=np.float64)
        if self._wrap_idx.size:
            q = q.copy()
            w = q[..., self._wrap_idx]
            q[..., self._wrap_idx] = (w + np.pi) % (2.0 * np.pi) - np.pi
        out = np.rint(q * self.inv)
        np.clip(out, -I16_MAX, I16_MAX, out=out)
        np.nan_to_num(out, copy=False)
        return out.astype(np.int16)

    def dequantize(self, q16: np.ndarray) -> np.ndarray:
        return q16.astype(np.float64) * self.scale


def _varints(delta: np.ndarray) -> bytes:
    """Zigzag LEB128 for an int32 vector whose values fit in 17 bits (so at most 3 bytes each)."""
    if delta.size == 0:
        return b""
    zz = ((delta << 1) ^ (delta >> 31)).astype(np.uint32)
    two = zz >= 0x80
    three = zz >= 0x4000
    lens = 1 + two.astype(np.intp) + three
    ends = np.cumsum(lens)
    start = ends - lens
    out = np.empty(int(ends[-1]), dtype=np.uint8)
    out[start] = (zz & 0x7F) | (two.astype(np.uint32) << 7)
    if two.any():
        s2 = start[two]
        z2 = zz[two]
        out[s2 + 1] = ((z2 >> 7) & 0x7F) | (three[two].astype(np.uint32) << 7)
        if three.any():
            out[start[three] + 2] = (zz[three] >> 14) & 0x7F
    return out.tobytes()


def encode_key(seq: int, t_ms: int, q16: np.ndarray, state: np.ndarray | None) -> bytes:
    n_envs, nq = q16.shape
    flags = FLAG_STATE if state is not None else 0
    parts = [HEADER.pack(MAGIC, KIND_KEY, flags, 0, seq & 0xFFFFFFFF, 0, t_ms & 0xFFFFFFFF, n_envs, nq)]
    if state is not None:
        parts.append(np.ascontiguousarray(state, dtype=np.uint8).tobytes())
        if n_envs & 1:
            parts.append(b"\x00")
    parts.append(np.ascontiguousarray(q16, dtype="<i2").tobytes())
    return b"".join(parts)


def encode_delta(seq: int, base_seq: int, t_ms: int, q16: np.ndarray, base: np.ndarray,
                 state: np.ndarray | None) -> bytes:
    """Delta of ``q16`` against ``base``. Pass ``state`` only when it differs from the base frame's."""
    n_envs, nq = q16.shape
    flags = FLAG_STATE if state is not None else 0
    parts = [HEADER.pack(MAGIC, KIND_DELTA, flags, 0, seq & 0xFFFFFFFF, base_seq & 0xFFFFFFFF,
                         t_ms & 0xFFFFFFFF, n_envs, nq)]
    if state is not None:
        parts.append(np.ascontiguousarray(state, dtype=np.uint8).tobytes())
    changed = q16 != base
    rows = changed.any(axis=1)
    parts.append(np.packbits(rows, bitorder="little").tobytes())
    if rows.any():
        sub = changed[rows]
        parts.append(np.packbits(sub, axis=1, bitorder="little").tobytes())
        delta = q16[rows][sub].astype(np.int32) - base[rows][sub].astype(np.int32)
        parts.append(_varints(delta))
    return b"".join(parts)


def decode(buf: bytes, q16: np.ndarray | None = None, state: np.ndarray | None = None) -> dict:
    """Reference decoder (the browser's is ``decodeFrame`` in protocol.js).

    ``q16`` / ``state`` are the client's current tables; they are updated in place for a delta and
    replaced for a key frame. Returns ``{kind, seq, base_seq, t_ms, q16, state}``.
    """
    magic, kind, flags, _, seq, base_seq, t_ms, n_envs, nq = HEADER.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ValueError("not a kaiku frame")
    o = HEADER_SIZE
    mv = memoryview(buf)
    if flags & FLAG_STATE:
        state = np.frombuffer(mv[o:o + n_envs], dtype=np.uint8).copy()
        o += n_envs
    if kind == KIND_KEY:
        o += o & 1
        q16 = np.frombuffer(mv[o:o + 2 * n_envs * nq], dtype="<i2").reshape(n_envs, nq).copy()
    elif kind == KIND_DELTA:
        if q16 is None:
            raise ValueError("delta frame without a base")
        rb = (n_envs + 7) // 8
        rows = np.unpackbits(np.frombuffer(mv[o:o + rb], dtype=np.uint8), bitorder="little")[:n_envs].astype(bool)
        o += rb
        cb = (nq + 7) // 8
        n_rows = int(rows.sum())
        raw = np.frombuffer(mv[o:o + n_rows * cb], dtype=np.uint8).reshape(n_rows, cb)
        masks = np.unpackbits(raw, axis=1, bitorder="little")[:, :nq].astype(bool)
        o += n_rows * cb
        deltas = np.empty(int(masks.sum()), dtype=np.int32)
        for i in range(deltas.size):
            v = shift = 0
            while True:
                b = buf[o]
                o += 1
                v |= (b & 0x7F) << shift
                shift += 7
                if not b & 0x80:
                    break
            deltas[i] = (v >> 1) ^ -(v & 1)
        sub = q16[rows]
        sub[masks] = (sub[masks].astype(np.int32) + deltas).astype(np.int16)
        q16[rows] = sub
    else:
        raise ValueError(f"unknown frame kind {kind}")
    return {"kind": kind, "seq": seq, "base_seq": base_seq, "t_ms": t_ms, "q16": q16, "state": state}

