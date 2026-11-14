"""Turn a MuJoCo ``MjModel`` into what the browser needs to draw it.

Two products:

* the **scene description** (:func:`export_scene`, JSON-able): the kinematic tree for in-browser
  forward kinematics, the visual geoms (groups 0 to 2) with their materials and textures, lights,
  bounds and the per-coordinate quantisation steps of the state stream;
* the **mesh pack** (:meth:`SceneExport.ensure_meshes`): one binary file per level of detail,
  cached on disk under a hash of the mesh data so only the first run pays for the simplification.

Levels of detail: 0 is the mesh exactly as MuJoCo compiled it (the lead robot, close up), 1 and 2
are simplified with fast-simplification. The triangle budget of a mesh depends only on its own
surface area, so the G1 with hands comes out at about 25k (lod 1) and 6.5k (lod 2) triangles per
robot whatever else is in the scene.

Pack file layout (little endian)::

    "KMSH"  u32 version  u32 json_len  json (padded to 4 bytes)  blobs (each padded to 4 bytes)

    json = {"lod": k, "meshes": [{"id", "name", "nv", "nf", "center": [3], "half": [3],
                                  "pos": off, "nrm": off, "idx": off, "idx_bytes": 2|4, "uv": off|null}]}
    pos  i16 x 3 per vertex, normalised: p = center + half * (v / 32767); the box is the same for
         every lod of a mesh, so the geom transform does not depend on the lod
    nrm  i8 x 3 per vertex, normalised
    idx  u16 or u32 x 3 per triangle
    uv   f32 x 2 per vertex (only meshes that have texture coordinates)

Offsets are relative to the start of the blob section.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import struct
import tempfile
import threading
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .protocol import Quantizer

EXPORT_VERSION = 3
PACK_MAGIC = b"KMSH"
PACK_VERSION = 1
VISUAL_GROUPS = (0, 1, 2)
N_LODS = 3
# Triangles for a simplified mesh = density * area[m^2] ** 0.6, at least MIN_TRIS, never more than it has.
LOD_DENSITY = (None, 3500.0, 900.0)
LOD_AREA_POWER = 0.6
MIN_TRIS = 40
CREASE_DEG = 38.0
# A plane on the world body at least this large (half size, metres; 0 means infinite) is the floor:
# the viewer draws one stage floor instead of a copy per environment.
GROUND_MIN_HALF = 2.5

_G = mujoco.mjtGeom
GEOM_TYPES = {
    int(_G.mjGEOM_PLANE): "plane",
    int(_G.mjGEOM_SPHERE): "sphere",
    int(_G.mjGEOM_CAPSULE): "capsule",
    int(_G.mjGEOM_ELLIPSOID): "ellipsoid",
    int(_G.mjGEOM_CYLINDER): "cylinder",
    int(_G.mjGEOM_BOX): "box",
    int(_G.mjGEOM_MESH): "mesh",
}
_TEX_RGB = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
_TEX_2D = int(mujoco.mjtTexture.mjTEXTURE_2D)

_build_lock = threading.Lock()


def default_cache_dir() -> Path:
    root = os.environ.get("KAIKU_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "kaiku")
    return Path(root) / "web"


# ----------------------------------------------------------------------------------------------
# kinematic tree


def export_tree(model: mujoco.MjModel) -> dict:
    """Everything forward kinematics needs, as plain lists (see ``kinematics.fk`` and fk.js)."""
    m = model
    return {
        "nbody": int(m.nbody),
        "njnt": int(m.njnt),
        "nq": int(m.nq),
        "parent": m.body_parentid.tolist(),
        "pos": m.body_pos.ravel().tolist(),
        "quat": m.body_quat.ravel().tolist(),
        "jntadr": m.body_jntadr.tolist(),
        "jntnum": m.body_jntnum.tolist(),
        "jtype": m.jnt_type.tolist(),
        "jqadr": m.jnt_qposadr.tolist(),
        "jpos": m.jnt_pos.ravel().tolist(),
        "jaxis": m.jnt_axis.ravel().tolist(),
        "qpos0": m.qpos0.tolist(),
        "names": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)],
    }


def make_quantizer(model: mujoco.MjModel) -> Quantizer:
    return Quantizer.from_joints(model.nq, model.jnt_type, model.jnt_qposadr, model.jnt_limited, model.jnt_range)


# ----------------------------------------------------------------------------------------------
# geoms, materials, textures


def _png(img: np.ndarray) -> bytes:
    """Minimal PNG encoder for an (h, w, c) uint8 image, c in 1..4."""
    h, w, c = img.shape
    raw = np.zeros((h, 1 + w * c), dtype=np.uint8)
    raw[:, 1:] = img.reshape(h, w * c)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, {1: 0, 2: 4, 3: 2, 4: 6}[c], 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)) + chunk(b"IEND", b"")


def _texture_png(m: mujoco.MjModel, tid: int) -> tuple[str, bytes, int, int]:
    h, w, c = int(m.tex_height[tid]), int(m.tex_width[tid]), int(m.tex_nchannel[tid])
    a = int(m.tex_adr[tid])
    data = np.asarray(m.tex_data[a:a + h * w * c], dtype=np.uint8).reshape(h, w, c)
    png = _png(data)
    return hashlib.blake2b(png, digest_size=8).hexdigest() + ".png", png, w, h


def _material(m: mujoco.MjModel, g: int) -> dict:
    """Resolve colour and surface the way MuJoCo's own renderer does (material, then geom rgba)."""
    matid = int(m.geom_matid[g])
    rgba = m.geom_rgba[g].astype(float)
    out = {"rgba": rgba.tolist(), "emission": 0.0, "specular": 0.5, "shininess": 0.5,
           "metallic": -1.0, "roughness": -1.0, "tex": -1, "texrepeat": [1.0, 1.0], "texuniform": False}
    if matid >= 0:
        out.update(
            emission=float(m.mat_emission[matid]), specular=float(m.mat_specular[matid]),
            shininess=float(m.mat_shininess[matid]), metallic=float(m.mat_metallic[matid]),
            roughness=float(m.mat_roughness[matid]), tex=int(m.mat_texid[matid, _TEX_RGB]),
            texrepeat=m.mat_texrepeat[matid].astype(float).tolist(), texuniform=bool(m.mat_texuniform[matid]))
        if np.allclose(rgba, (0.5, 0.5, 0.5, 1.0)):
            out["rgba"] = m.mat_rgba[matid].astype(float).tolist()
    return out


def _is_ground(m: mujoco.MjModel, g: int) -> bool:
    if int(m.geom_type[g]) != int(_G.mjGEOM_PLANE) or int(m.geom_bodyid[g]) != 0:
        return False
    sx, sy = float(m.geom_size[g, 0]), float(m.geom_size[g, 1])
    return sx <= 0 or sy <= 0 or max(sx, sy) >= GROUND_MIN_HALF


def _geom_corners(m: mujoco.MjModel, g: int) -> np.ndarray:
    """The 8 corners of a geom's bounding box in its own frame."""
    if int(m.geom_type[g]) == int(_G.mjGEOM_PLANE):
        c, h = np.zeros(3), np.array([m.geom_size[g, 0], m.geom_size[g, 1], 0.0])
    else:
        c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=float)
    return c + signs * h


def _bounds(m: mujoco.MjModel, geom_ids: list[int], samples: int = 96) -> dict:
    """Rest-pose box of the visual geoms, and the box they sweep over random joint positions."""
    d = mujoco.MjData(m)
    corners = {g: _geom_corners(m, g) for g in geom_ids}

    def box(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d.qpos[:] = qpos
        mujoco.mj_kinematics(m, d)
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for g, c in corners.items():
            w = d.geom_xpos[g] + c @ d.geom_xmat[g].reshape(3, 3).T
            lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
        return lo, hi

    if not geom_ids:
        z = np.zeros(3)
        return {"min": (z - 0.5).tolist(), "max": (z + 0.5).tolist(), "reach_min": (z - 0.5).tolist(),
                "reach_max": (z + 0.5).tolist()}
    lo, hi = box(m.qpos0)
    rlo, rhi = lo.copy(), hi.copy()
    rng = np.random.default_rng(7)
    movable = [j for j in range(m.njnt) if int(m.jnt_type[j]) in (2, 3)]
    for _ in range(samples if movable else 0):
        q = m.qpos0.copy()
        for j in movable:
            a = int(m.jnt_qposadr[j])
            if m.jnt_limited[j]:
                q[a] = rng.uniform(*m.jnt_range[j])
            elif int(m.jnt_type[j]) == 3:
                q[a] = rng.uniform(-np.pi, np.pi)
        a, b = box(q)
        rlo, rhi = np.minimum(rlo, a), np.maximum(rhi, b)
    return {"min": lo.tolist(), "max": hi.tolist(), "reach_min": rlo.tolist(), "reach_max": rhi.tolist()}


def _lights(m: mujoco.MjModel) -> list[dict]:
    if m.nlight == 0:
        return []
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    mujoco.mj_camlight(m, d)
    out = []
    for i in range(m.nlight):
        if hasattr(m, "light_type"):
            kind = {int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL): "directional",
                    int(mujoco.mjtLightType.mjLIGHT_POINT): "point"}.get(int(m.light_type[i]), "spot")
        else:  # MuJoCo < 3.3
            kind = "directional" if m.light_directional[i] else "spot"
        out.append({
            "type": kind,
            "pos": d.light_xpos[i].tolist(),
            "dir": d.light_xdir[i].tolist(),
            "diffuse": m.light_diffuse[i].astype(float).tolist(),
            "castshadow": bool(m.light_castshadow[i]),
            "active": bool(m.light_active[i]) if hasattr(m, "light_active") else True,
        })
    return out


# ----------------------------------------------------------------------------------------------
# meshes


def _mesh_arrays(m: mujoco.MjModel, i: int) -> dict:
    """Raw compiled mesh i with one normal (and uv) per vertex, splitting vertices where needed."""
    va, vn = int(m.mesh_vertadr[i]), int(m.mesh_vertnum[i])
    fa, fn = int(m.mesh_faceadr[i]), int(m.mesh_facenum[i])
    V = np.asarray(m.mesh_vert[va:va + vn], dtype=np.float64)
    F = np.asarray(m.mesh_face[fa:fa + fn], dtype=np.int64)
    na, nn = int(m.mesh_normaladr[i]), int(m.mesh_normalnum[i])
    N = np.asarray(m.mesh_normal[na:na + nn], dtype=np.float64)
    FN = np.asarray(m.mesh_facenormal[fa:fa + fn], dtype=np.int64)
    ta, tn = int(m.mesh_texcoordadr[i]), int(m.mesh_texcoordnum[i])
    has_uv = ta >= 0 and tn > 0
    keys = [F.ravel(), FN.ravel()]
    if has_uv:
        T = np.asarray(m.mesh_texcoord[ta:ta + tn], dtype=np.float64)
        keys.append(np.asarray(m.mesh_facetexcoord[fa:fa + fn], dtype=np.int64).ravel())
    if not has_uv and nn == vn and np.array_equal(F, FN):
        return {"pos": V, "nrm": N, "idx": F, "uv": None, "V": V, "F": F}
    uniq, inv = np.unique(np.stack(keys, axis=1), axis=0, return_inverse=True)
    uv = None
    if has_uv:
        uv = T[uniq[:, 2]].copy()
        uv[:, 1] = 1.0 - uv[:, 1]        # MuJoCo measures v from the top row, three.js from the bottom
    return {"pos": V[uniq[:, 0]], "nrm": N[uniq[:, 1]], "idx": inv.reshape(-1, 3), "uv": uv, "V": V, "F": F}


def _area(V: np.ndarray, F: np.ndarray) -> float:
    return float(0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum())


def lod_target(area: float, n_faces: int, lod: int) -> int:
    if LOD_DENSITY[lod] is None:
        return n_faces
    return int(min(n_faces, max(MIN_TRIS, round(LOD_DENSITY[lod] * area ** LOD_AREA_POWER))))


def crease_normals(V: np.ndarray, F: np.ndarray, angle_deg: float = CREASE_DEG):
    """Vertex normals that stay sharp across folds steeper than ``angle_deg``.

    Returns ``(pos, nrm, idx)`` with vertices duplicated along the creases.
    """
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])     # length = 2 * area
    a2 = np.linalg.norm(fn, axis=1)
    keep = a2 > 1e-14
    F, fn, a2 = F[keep], fn[keep], a2[keep]
    if len(F) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    fu = fn / a2[:, None]
    corner_v = F.ravel()
    corner_f = np.repeat(np.arange(len(F)), 3)
    order = np.argsort(corner_v, kind="stable")
    sv, sf = corner_v[order], corner_f[order]
    starts = np.flatnonzero(np.r_[True, sv[1:] != sv[:-1]])
    sizes = np.diff(np.r_[starts, len(sv)])
    gstart, gsize = np.repeat(starts, sizes), np.repeat(sizes, sizes)
    # every (corner, corner) pair around the same vertex
    me = np.repeat(np.arange(len(sv)), gsize)
    first = np.cumsum(gsize) - gsize
    other = np.repeat(gstart, gsize) + (np.arange(len(me)) - np.repeat(first, gsize))
    fi, fj = sf[me], sf[other]
    ok = np.einsum("ij,ij->i", fu[fi], fu[fj]) >= np.cos(np.radians(angle_deg))
    me, fj = me[ok], fj[ok]
    acc = np.stack([np.bincount(me, weights=fn[fj, k], minlength=len(sv)) for k in range(3)], axis=1)
    acc /= np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-20)
    nrm_c = np.empty_like(acc)
    nrm_c[order] = acc
    key = np.concatenate([corner_v[:, None], np.rint(nrm_c * 127).astype(np.int64)], axis=1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    nrm = uniq[:, 1:].astype(np.float64)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-20)
    return V[uniq[:, 0]], nrm, inv.reshape(-1, 3)


def _cluster(V: np.ndarray, F: np.ndarray, cell: float, lo: np.ndarray):
    """Merge all vertices that fall in the same grid cell; place each merged vertex where it best
    fits the planes of the triangles around it (so flat faces stay flat and edges stay crisp)."""
    ijk = np.floor((V - lo) / cell).astype(np.int64)
    key = (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]
    uniq, inv = np.unique(key, return_inverse=True)
    nc = len(uniq)
    f = inv[F]
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[ok]
    # same triangle, same winding: keep one (opposite windings both stay, they face different ways)
    first = np.argmin(f, axis=1)
    f = np.take_along_axis(f, (first[:, None] + np.arange(3)) % 3, axis=1)
    f = np.unique(f, axis=0)

    count = np.bincount(inv, minlength=nc).astype(np.float64)
    mean = np.stack([np.bincount(inv, weights=V[:, k], minlength=nc) for k in range(3)], axis=1) / count[:, None]
    # plane quadrics of the original triangles, accumulated on the clusters they touch
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    w = np.linalg.norm(n, axis=1)
    good = w > 1e-18
    n, w, Fg = n[good] / w[good, None], w[good], F[good]
    d = -np.einsum("ij,ij->i", n, V[Fg[:, 0]])
    A = np.zeros((nc, 3, 3))
    b = np.zeros((nc, 3))
    for c in range(3):
        cl = inv[Fg[:, c]]
        for i in range(3):
            b[:, i] += np.bincount(cl, weights=-w * d * n[:, i], minlength=nc)
            for j in range(i, 3):
                A[:, i, j] += np.bincount(cl, weights=w * n[:, i] * n[:, j], minlength=nc)
    A[:, 1, 0], A[:, 2, 0], A[:, 2, 1] = A[:, 0, 1], A[:, 0, 2], A[:, 1, 2]
    # pull towards the mean so flat or straight neighbourhoods stay well defined
    reg = 1e-3 * (np.trace(A, axis1=1, axis2=2) + 1e-30)
    A += reg[:, None, None] * np.eye(3)
    b += reg[:, None] * mean
    pos = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    pos = np.clip(pos, mean - cell, mean + cell)
    return pos, f


def cluster_simplify(V: np.ndarray, F: np.ndarray, target: int):
    """Vertex clustering down to about ``target`` triangles. Unlike edge collapse this may change the
    topology, so it gets through meshes full of screw holes and inner walls."""
    lo, hi = V.min(0), V.max(0)
    diag = float(np.linalg.norm(hi - lo))
    a, b = diag / 2000.0, diag / 1.5
    best = None
    for _ in range(16):
        cell = float(np.sqrt(a * b))
        pos, f = _cluster(V, F, cell, lo - 0.5 * cell)
        if len(f) > target:
            a = cell
        else:
            b = cell
            if best is None or len(f) > len(best[1]):
                best = (pos, f)
    if best is None:
        best = _cluster(V, F, b, lo - 0.5 * b)
    pos, f = best
    used = np.unique(f)
    remap = np.full(len(pos), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return pos[used], remap[f]


def simplify(V: np.ndarray, F: np.ndarray, target: int):
    """Edge collapse (fast-simplification) first; clustering finishes what it could not reach."""
    try:
        import fast_simplification
    except ImportError:                                    # pragma: no cover - optional dependency
        fast_simplification = None
    v, f = V, F
    if fast_simplification is not None:
        # the simplifier's error thresholds are absolute, so work at a fixed size
        scale = 100.0 / max(float(np.linalg.norm(V.max(0) - V.min(0))), 1e-12)
        v, f = fast_simplification.simplify(V * scale, F.astype(np.int32), target_count=target, agg=7.0)
        v, f = np.asarray(v, dtype=np.float64) / scale, np.asarray(f, dtype=np.int64)
    if len(f) > 1.2 * target:
        v, f = cluster_simplify(v, f, target)
    return v, f


def build_lods(mesh: dict) -> list[dict]:
    """Levels of detail 0..2 of one mesh as dicts with ``pos``, ``nrm``, ``idx`` and ``uv``."""
    full = {k: mesh[k] for k in ("pos", "nrm", "idx", "uv")}
    lods = [full]
    V, F = mesh["V"], mesh["F"]
    area = _area(V, F)
    for lod in range(1, N_LODS):
        target = lod_target(area, len(F), lod)
        if mesh["uv"] is not None or target >= 0.9 * len(F):
            lods.append(full)
            continue
        v, f = simplify(V, F, target)
        pos, nrm, idx = crease_normals(v, f)
        if len(idx) == 0:
            lods.append(lods[-1])
            continue
        lods.append({"pos": pos, "nrm": nrm, "idx": idx, "uv": None})
    return lods


def _pad4(b: bytes) -> bytes:
    return b + b"\x00" * (-len(b) % 4)


def write_pack(path: Path, lod: int, meshes: list[dict]) -> int:
    """``meshes``: dicts with id, name, pos, nrm, idx, uv, center, half. Returns the triangle count."""
    blobs: list[bytes] = []
    off = 0
    entries = []
    tris = 0
    for me in meshes:
        half = np.maximum(me["half"], 1e-9)
        p16 = np.clip(np.rint((me["pos"] - me["center"]) / half * 32767.0), -32767, 32767).astype("<i2")
        n8 = np.clip(np.rint(me["nrm"] * 127.0), -127, 127).astype(np.int8)
        nv = len(p16)
        idx = np.ascontiguousarray(me["idx"], dtype="<u2" if nv <= 65535 else "<u4")
        entry = {"id": me["id"], "name": me["name"], "nv": nv, "nf": int(len(idx)),
                 "center": [float(x) for x in me["center"]], "half": [float(x) for x in half],
                 "idx_bytes": idx.dtype.itemsize, "uv": None}
        parts = [("pos", p16.tobytes()), ("nrm", n8.tobytes()), ("idx", idx.tobytes())]
        if me.get("uv") is not None:
            parts.append(("uv", np.ascontiguousarray(me["uv"], dtype="<f4").tobytes()))
        for name, b in parts:
            b = _pad4(b)
            entry[name] = off
            blobs.append(b)
            off += len(b)
        entries.append(entry)
        tris += len(idx)
    head = json.dumps({"lod": lod, "meshes": entries}, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 4)
    data = PACK_MAGIC + struct.pack("<II", PACK_VERSION, len(head)) + head + b"".join(blobs)
    _atomic_write(path, data)
    _atomic_write(path.with_name(path.name + ".gz"), gzip.compress(data, compresslevel=6, mtime=0))
    return tris


def read_pack(data: bytes | Path) -> dict:
    """Decode a pack back to float arrays (tests, tools). ``{"lod": k, "meshes": {id: {...}}}``."""
    if isinstance(data, (str, Path)):
        data = Path(data).read_bytes()
    if data[:4] != PACK_MAGIC:
        raise ValueError("not a kaiku mesh pack")
    version, jlen = struct.unpack_from("<II", data, 4)
    if version != PACK_VERSION:
        raise ValueError(f"mesh pack version {version}")
    head = json.loads(data[12:12 + jlen].decode())
    base = 12 + jlen
    out = {}
    for e in head["meshes"]:
        nv, nf = e["nv"], e["nf"]
        p16 = np.frombuffer(data, dtype="<i2", count=nv * 3, offset=base + e["pos"]).reshape(nv, 3)
        n8 = np.frombuffer(data, dtype=np.int8, count=nv * 3, offset=base + e["nrm"]).reshape(nv, 3)
        idx = np.frombuffer(data, dtype="<u2" if e["idx_bytes"] == 2 else "<u4", count=nf * 3,
                            offset=base + e["idx"]).reshape(nf, 3)
        uv = None
        if e["uv"] is not None:
            uv = np.frombuffer(data, dtype="<f4", count=nv * 2, offset=base + e["uv"]).reshape(nv, 2)
        center, half = np.array(e["center"]), np.array(e["half"])
        out[e["id"]] = {"name": e["name"], "pos": center + half * (p16 / 32767.0), "nrm": n8 / 127.0,
                        "idx": idx.astype(np.int64), "uv": uv, "center": center, "half": half}
    return {"lod": head["lod"], "meshes": out}


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _mesh_hash(m: mujoco.MjModel, mesh_ids: list[int]) -> str:
    h = hashlib.blake2b(digest_size=10)
    h.update(repr((EXPORT_VERSION, PACK_VERSION, LOD_DENSITY, LOD_AREA_POWER, MIN_TRIS, CREASE_DEG)).encode())
    for i in mesh_ids:
        va, vn = int(m.mesh_vertadr[i]), int(m.mesh_vertnum[i])
        fa, fn = int(m.mesh_faceadr[i]), int(m.mesh_facenum[i])
        na, nn = int(m.mesh_normaladr[i]), int(m.mesh_normalnum[i])
        ta, tn = int(m.mesh_texcoordadr[i]), int(m.mesh_texcoordnum[i])
        h.update(struct.pack("<5i", i, vn, fn, nn, tn))
        h.update(np.ascontiguousarray(m.mesh_vert[va:va + vn]).tobytes())
        h.update(np.ascontiguousarray(m.mesh_face[fa:fa + fn]).tobytes())
        h.update(np.ascontiguousarray(m.mesh_normal[na:na + nn]).tobytes())
        h.update(np.ascontiguousarray(m.mesh_facenormal[fa:fa + fn]).tobytes())
        if ta >= 0 and tn > 0:
            h.update(np.ascontiguousarray(m.mesh_texcoord[ta:ta + tn]).tobytes())
            h.update(np.ascontiguousarray(m.mesh_facetexcoord[fa:fa + fn]).tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------
# public API


@dataclass
class SceneExport:
    """Result of :func:`export_scene`."""

    scene: dict                       # JSON-able scene description
    model_hash: str                   # changes whenever anything the browser draws changes
    mesh_hash: str                    # key of the mesh pack cache
    textures: dict[str, bytes]        # file name -> PNG bytes
    quantizer: Quantizer
    cache_dir: Path
    _model: mujoco.MjModel = field(repr=False, default=None)
    _mesh_ids: list[int] = field(repr=False, default_factory=list)

    @property
    def pack_dir(self) -> Path:
        return self.cache_dir / f"meshes-{self.mesh_hash}"

    def pack_path(self, lod: int, gz: bool = False) -> Path:
        return self.pack_dir / (f"lod{lod}.bin" + (".gz" if gz else ""))

    def meshes_ready(self) -> bool:
        return (self.pack_dir / "meta.json").exists()

    def ensure_meshes(self) -> dict:
        """Build the mesh pack unless it is cached. Blocking (a few seconds for the G1, once)."""
        meta_path = self.pack_dir / "meta.json"
        with _build_lock:
            if meta_path.exists():
                try:
                    return json.loads(meta_path.read_text())
                except (OSError, ValueError):
                    pass
            m = self._model
            per_lod: list[list[dict]] = [[] for _ in range(N_LODS)]
            for i in self._mesh_ids:
                lods = build_lods(_mesh_arrays(m, i))
                lo = np.min([l["pos"].min(0) for l in lods], axis=0)
                hi = np.max([l["pos"].max(0) for l in lods], axis=0)
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, i) or f"mesh{i}"
                for k, l in enumerate(lods):
                    per_lod[k].append({**l, "id": i, "name": name, "center": (lo + hi) / 2, "half": (hi - lo) / 2})
            meta = {"version": PACK_VERSION, "hash": self.mesh_hash, "lods": []}
            for k in range(N_LODS):
                tris = write_pack(self.pack_path(k), k, per_lod[k])
                meta["lods"].append({"lod": k, "tris": tris, "bytes": self.pack_path(k).stat().st_size,
                                     "gz_bytes": self.pack_path(k, gz=True).stat().st_size})
            _atomic_write(meta_path, json.dumps(meta).encode())
            return meta


def export_scene(model: mujoco.MjModel, cache_dir: str | os.PathLike | None = None) -> SceneExport:
    """Describe ``model`` for the browser. Cheap (milliseconds); the mesh pack is built separately."""
    m = model
    geoms, geom_ids, mesh_ids, textures, tex_meta = [], [], [], {}, {}
    # Resolve the stage height before filtering parked free bodies. Twin scenes use a large
    # world box named ``floor`` rather than a MuJoCo plane.
    ground_z = None
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) != 0:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        typ = int(m.geom_type[g])
        if typ == int(_G.mjGEOM_PLANE):
            ground_z = float(m.geom_pos[g, 2])
        elif name == "floor" and typ == int(_G.mjGEOM_BOX):
            ground_z = float(m.geom_pos[g, 2] + m.geom_size[g, 2])
    parked_bodies = set()
    if ground_z is not None:
        for j in range(m.njnt):
            if int(m.jnt_type[j]) == 0 and float(m.qpos0[int(m.jnt_qposadr[j]) + 2]) < ground_z - 0.35:
                parked_bodies.add(int(m.jnt_bodyid[j]))
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) in parked_bodies:
            continue
        gtype = GEOM_TYPES.get(int(m.geom_type[g]))
        if gtype is None or int(m.geom_group[g]) not in VISUAL_GROUPS:
            continue
        mat = _material(m, g)
        if mat["rgba"][3] <= 0:
            continue
        if _is_ground(m, g):
            z = float(m.geom_pos[g, 2])
            ground_z = z if ground_z is None else max(ground_z, z)
            continue
        tex = mat["tex"]
        if tex >= 0 and int(m.tex_type[tex]) == _TEX_2D:
            if tex not in tex_meta:
                name, png, w, h = _texture_png(m, tex)
                textures[name] = png
                tex_meta[tex] = {"id": tex, "file": name, "w": w, "h": h}
        else:
            mat["tex"] = -1
        entry = {
            "id": g,
            "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "",
            "body": int(m.geom_bodyid[g]),
            "type": gtype,
            "group": int(m.geom_group[g]),
            "size": m.geom_size[g].astype(float).tolist(),
            "pos": m.geom_pos[g].astype(float).tolist(),
            "quat": m.geom_quat[g].astype(float).tolist(),
            **mat,
        }
        if gtype == "mesh":
            mid = int(m.geom_dataid[g])
            entry["mesh"] = mid
            if mid not in mesh_ids:
                mesh_ids.append(mid)
        geoms.append(entry)
        geom_ids.append(g)
    mesh_ids.sort()

    bounds = _bounds(m, geom_ids)
    if ground_z is None:
        ground_z = float(bounds["min"][2])
    quant = make_quantizer(m)
    mesh_hash = _mesh_hash(m, mesh_ids)
    scene = {
        "version": EXPORT_VERSION,
        "nq": int(m.nq),
        "tree": export_tree(m),
        "quant": {"scale": quant.scale.tolist()},
        "geoms": geoms,
        "textures": list(tex_meta.values()),
        "lights": _lights(m),
        "bounds": bounds,
        "ground_z": ground_z,
        "meshes": {"hash": mesh_hash, "ids": mesh_ids, "lods": N_LODS},
        "states": ["idle", "running", "success", "failed", "lead"],
    }
    digest = hashlib.blake2b(json.dumps(scene, sort_keys=True).encode(), digest_size=10).hexdigest()
    scene["model_hash"] = digest
    return SceneExport(scene=scene, model_hash=digest, mesh_hash=mesh_hash, textures=textures, quantizer=quant,
                       cache_dir=Path(cache_dir) if cache_dir else default_cache_dir(), _model=m, _mesh_ids=mesh_ids)

