"""Build the MuJoCo digital twin: a fixed-base Unitree G1 with Dex3 hands at the challenge counter.

The robot comes from MuJoCo Menagerie (``unitree_g1/g1_with_hands.xml``, BSD-3) and is edited
in memory, never on disk:

* the pelvis is welded at the world origin, so world == pelvis frame as in the challenge;
* legs and waist are locked (their joints and motors removed), leaving the challenge's action
  space: 14 arm + 14 hand position targets;
* hand collision meshes become boxes. Mesh-vs-box contact gives one contact point, which lets a
  pinched cube spin out of the fingers; box-vs-box gives a contact patch and a stable grasp;
* palm, fingertip and grasp sites are added for IK, retargeting and the safety filter.

Counter, mat, bins and the 18 printed objects are generated from ``scene.yaml``. Every
environment shares this one model; objects that are not part of an episode are parked under
the floor.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from kaiku.sim.scene import ASSETS, Scene

REPO = Path(__file__).resolve().parents[2]
G1_XML = Path(os.environ.get("KAIKU_G1_XML", REPO / "third_party/mujoco_menagerie/unitree_g1/g1_with_hands.xml"))

PELVIS_HEIGHT = 0.79            # m above the floor (scene.yaml assumption for a standing G1)
CONTROL_HZ = 30                 # the challenge's action rate
SUBSTEPS = 16                   # physics steps per control tick (8 halves the cost but drops triangle grasps to 84 %)
TIMESTEP = 1.0 / (CONTROL_HZ * SUBSTEPS)
OBJECT_MASS = 0.050             # kg, printed PLA part
FINGER_FRICTION = float(os.environ.get("KAIKU_FINGER_FRICTION", 1.0))
# Palm frame of the challenge URDF (left_hand_palm_link) relative to wrist_yaw_link; the right is mirrored in y.
PALM_IN_WRIST = np.array([0.0415, 0.003, 0.0])
# Where a pinched object's centre sits in the LEFT palm frame (challenge classical baseline, thumb vs index).
GRASP_IN_PALM = np.array([0.099, -0.084, 0.015])

# Collision bits. Robot geoms collide with objects and the static scene but not with each other.
_ROBOT, _OBJECT = 1, 2
_HAND_LINKS = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
_TIP_LINKS = {"thumb": "thumb_2", "index": "index_1", "middle": "middle_1"}


def _quat_yaw(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _body_frame_aabbs(model: mujoco.MjModel) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Axis-aligned bounds, in the owning body's frame, of every collision mesh geom (by geom id)."""
    out = {}
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_group[g] != 3:
            continue
        mid = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[mid]: model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[g])
        p = v @ R.reshape(3, 3).T + model.geom_pos[g]
        out[g] = (p.min(0), p.max(0))
    return out


def build_spec(scene: Scene | None = None, with_scene: bool = True) -> mujoco.MjSpec:
    """The twin as an editable spec. ``with_scene=False`` gives the bare robot (28 DoF), used
    for kinematics where the objects would only cost time."""
    scene = scene or Scene()
    aabbs = _body_frame_aabbs(mujoco.MjModel.from_xml_path(str(G1_XML)))
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    spec.modelname = "kaiku_twin" if with_scene else "kaiku_robot"
    for key in list(spec.keys):
        spec.delete(key)

    # -- fixed base, locked lower body -------------------------------------------------
    spec.body("pelvis").pos = [0.0, 0.0, 0.0]
    locked = [j for j in spec.joints
              if j.name == "floating_base_joint" or any(k in j.name for k in ("hip", "knee", "ankle", "waist"))]
    locked_names = {j.name for j in locked}
    for act in [a for a in spec.actuators if a.name in locked_names]:
        spec.delete(act)
    for j in locked:
        spec.delete(j)

    # The real arms run a gravity-compensated position controller; without it the position
    # motors sag by centimetres under the arm's own weight.
    for body in spec.bodies:
        if any(k in body.name for k in ("shoulder", "elbow", "wrist", "hand")):
            body.gravcomp = 1.0

    # -- collision geometry --------------------------------------------------------------
    for gid, g in enumerate(spec.geoms):
        if g.group != 3:
            continue
        body = g.parent.name
        if not any(k in body for k in ("shoulder", "elbow", "wrist", "hand", "torso")):
            g.contype, g.conaffinity = 0, 0          # legs and pelvis never touch the task
            continue
        g.contype, g.conaffinity = _ROBOT, 0
        if "hand" in body and gid in aabbs:
            lo, hi = aabbs[gid]
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.meshname = ""
            g.pos, g.quat, g.size = (lo + hi) / 2, [1, 0, 0, 0], (hi - lo) / 2
        if "hand" in body or "wrist_yaw" in body:
            g.friction = [FINGER_FRICTION, 0.02, 0.0005]
            g.condim = 4
            g.solref = [0.005, 1.0]
            g.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]

    # -- sites ----------------------------------------------------------------------------
    for side, sy in (("left", 1.0), ("right", -1.0)):
        wrist = spec.body(f"{side}_wrist_yaw_link")
        palm = PALM_IN_WRIST * [1, sy, 1]
        wrist.add_site(name=f"{side}_palm", pos=palm, size=[0.006, 0, 0], group=5)
        wrist.add_site(name=f"{side}_grasp", pos=palm + GRASP_IN_PALM * [1, sy, 1], size=[0.006, 0, 0], group=5)
        for finger, link in _TIP_LINKS.items():
            # Distal links are 59 mm long; the pad centre is ~46 mm out, on the palmar side.
            tip = [0.0, -0.046 * sy, 0.0] if finger == "thumb" else [0.046, 0.0, 0.0]
            spec.body(f"{side}_hand_{link}_link").add_site(
                name=f"{side}_{finger}_tip", pos=tip, size=[0.005, 0, 0], group=5)

    spec.option.timestep = TIMESTEP
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    if with_scene:
        _add_scene(spec, scene)
    return spec


def _add_scene(spec: mujoco.MjSpec, sc: Scene) -> None:
    world = spec.worldbody
    box, cyl, mesh = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_MESH
    static = dict(contype=0, conaffinity=_ROBOT | _OBJECT)
    visual = dict(contype=0, conaffinity=0, group=2)

    # Floor slab (finite, so parked objects can sit below it untouched).
    world.add_geom(name="floor", type=box, pos=[0.3, 0, -PELVIS_HEIGHT - 0.025], size=[2.0, 2.0, 0.025],
                   rgba=[0.13, 0.14, 0.16, 1], group=1, **static)

    # Counter top and legs.
    lo, hi = sc.counter_box
    cx, cy = (lo + hi) / 2
    hx, hy = (hi - lo) / 2
    top_t = 0.03
    world.add_geom(name="counter", type=box, pos=[cx, cy, sc.counter_z - top_t / 2], size=[hx, hy, top_t / 2],
                   rgba=[0.86, 0.84, 0.80, 1], group=1, friction=[0.8, 0.02, 0.0005], condim=4, **static)
    leg_h = (PELVIS_HEIGHT + sc.counter_z - top_t) / 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            world.add_geom(type=box, pos=[cx + sx * (hx - 0.04), cy + sy * (hy - 0.04), sc.counter_z - top_t - leg_h],
                           size=[0.02, 0.02, leg_h], rgba=[0.22, 0.23, 0.25, 1], **visual)

    # Printed mat: a textured sheet on the counter (visual only; objects rest on the counter top).
    spec.add_texture(name="mat", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(ASSETS / "mat_texture.png"))
    mat = spec.add_material(name="mat")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "mat"
    mlo, mhi = sc.mat_box
    world.add_geom(name="mat", type=box, pos=[*(mlo + mhi) / 2, sc.counter_z + 0.0001],
                   size=[*(mhi - mlo) / 2, 0.0003], material="mat", **visual)

    # Bins: five boxes each. Walls collide 3 mm thicker (outwards) than they look, so a dropped
    # part cannot tunnel through a 3 mm sheet between two physics steps.
    spec.add_mesh(name="bin", file=str(ASSETS / "bin.stl"), scale=[0.001] * 3)
    rgba = [*sc.bin_rgb, 1.0]
    half = sc.bin_outer[::-1] / 2
    t, h, extra = sc.bin_wall, sc.bin_height, 0.003
    for name, c in sc.bin_centers.items():
        world.add_geom(name=f"bin_{name}", type=mesh, meshname="bin", pos=[c[0], c[1], sc.counter_z],
                       rgba=rgba, **visual)
        world.add_geom(type=box, pos=[c[0], c[1], sc.counter_z + sc.bin_floor / 2],
                       size=[half[0], half[1], sc.bin_floor / 2], rgba=rgba, group=3,
                       friction=[0.6, 0.02, 0.0005], **static)
        for axis in (0, 1):
            for sgn in (-1, 1):
                pos = np.array([c[0], c[1], sc.counter_z + h / 2])
                size = np.array([half[0], half[1], h / 2])
                pos[axis] += sgn * (half[axis] - t / 2 + extra / 2)
                size[axis] = (t + extra) / 2
                world.add_geom(type=box, pos=pos, size=size, rgba=rgba, group=3, **static)

    # The 18 printed parts. Each starts parked; an episode moves the ones it uses onto the mat.
    for shape in ("cube", "cylinder", "triangle"):
        spec.add_mesh(name=f"object_{shape}", file=str(ASSETS / f"object_{shape}.stl"), scale=[0.001] * 3)
    r, hh = sc.size / 2, sc.height / 2
    contact = dict(contype=_OBJECT, conaffinity=_ROBOT | _OBJECT, group=3, friction=[0.8, 0.02, 0.0005],
                   condim=4, solref=[0.005, 1.0], solimp=[0.95, 0.99, 0.001, 0.5, 2.0], mass=OBJECT_MASS)
    for ob in sc.objects:
        body = world.add_body(name=ob.name, pos=sc.park_pos(ob.index))
        body.add_freejoint(name=ob.name)
        rgba = [*sc.colors[ob.color], 1.0]
        body.add_geom(type=mesh, meshname=f"object_{ob.shape}", pos=[0, 0, -hh], rgba=rgba, **visual)
        if ob.shape == "cube":
            body.add_geom(type=box, size=[r, r, hh], rgba=rgba, **contact)
        elif ob.shape == "cylinder":
            # condim 6 adds rolling resistance: a cylinder knocked onto its side comes to rest
            # instead of rolling about its bin for seconds.
            body.add_geom(type=cyl, size=[r, hh, 0], rgba=rgba, **{**contact, "condim": 6,
                                                                    "friction": [0.8, 0.02, 0.003]})
        else:
            body.add_geom(type=mesh, meshname="object_triangle", pos=[0, 0, -hh], rgba=rgba, **contact)

    world.add_light(pos=[0.2, 0.0, 2.0], dir=[0.1, 0.0, -1.0], diffuse=[0.8, 0.8, 0.8])
    world.add_light(pos=[-1.0, 1.5, 1.5], dir=[0.5, -0.7, -0.6], diffuse=[0.35, 0.35, 0.38])


@dataclass
class Twin:
    """A compiled model plus the index tables every other module needs.

    ``arm`` / ``hand`` arrays follow the challenge's 14-vectors: arms are left 7 then right 7;
    hands are left (thumb 0-2, middle 0-1, index 0-1) then right (thumb 0-2, index 0-1, middle 0-1).
    """

    model: mujoco.MjModel
    scene: Scene
    arm_names: list[str]
    hand_names: list[str]
    arm_qadr: np.ndarray       # (14,) qpos / dof address (robot joints are hinges, so they coincide per joint)
    hand_qadr: np.ndarray      # (14,)
    arm_dadr: np.ndarray
    hand_dadr: np.ndarray
    arm_act: np.ndarray        # (14,) actuator ids
    hand_act: np.ndarray
    arm_range: np.ndarray      # (14, 2)
    hand_range: np.ndarray
    obj_qadr: np.ndarray       # (18,) start of each object's free-joint qpos (7 values)
    obj_dadr: np.ndarray       # (18,) start of its 6 velocities
    obj_body: np.ndarray       # (18,) body ids
    site: dict[str, int]

    @property
    def n_objects(self) -> int:
        return len(self.obj_body)


def build_twin(scene: Scene | None = None, with_scene: bool = True) -> Twin:
    import yaml

    scene = scene or Scene()
    model = build_spec(scene, with_scene).compile()
    spec = yaml.safe_load((ASSETS / "joint_spec.yaml").read_text())
    arm_names = [j["name"] for j in spec["arm"]]
    hand_names = [j["name"] for j in spec["hand"]]

    def jid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)

    def aid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

    def table(names, fn):
        ids = np.array([fn(n) for n in names])
        assert (ids >= 0).all(), "model and challenge joint spec disagree"
        return ids

    arm_j, hand_j = table(arm_names, jid), table(hand_names, jid)
    obj_j = table([o.name for o in scene.objects], jid) if with_scene else np.zeros(0, int)
    sites = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i): i for i in range(model.nsite)}
    return Twin(
        model=model, scene=scene, arm_names=arm_names, hand_names=hand_names,
        arm_qadr=model.jnt_qposadr[arm_j], hand_qadr=model.jnt_qposadr[hand_j],
        arm_dadr=model.jnt_dofadr[arm_j], hand_dadr=model.jnt_dofadr[hand_j],
        arm_act=table(arm_names, aid), hand_act=table(hand_names, aid),
        arm_range=model.jnt_range[arm_j].copy(), hand_range=model.jnt_range[hand_j].copy(),
        obj_qadr=model.jnt_qposadr[obj_j], obj_dadr=model.jnt_dofadr[obj_j],
        obj_body=model.jnt_bodyid[obj_j] if with_scene else np.zeros(0, int),
        site={k: v for k, v in sites.items() if k},
    )
