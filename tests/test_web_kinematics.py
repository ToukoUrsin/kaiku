import mujoco, numpy as np
from kaiku.web.kinematics import fk
from kaiku.web.scene_export import export_tree

def test_tree_fk_matches_mujoco():
    m=mujoco.MjModel.from_xml_path('third_party/mujoco_menagerie/unitree_g1/g1_with_hands.xml'); d=mujoco.MjData(m); tree=export_tree(m); rng=np.random.default_rng(4)
    for _ in range(3):
        q=m.qpos0.copy()
        for j in range(m.njnt):
            if int(m.jnt_type[j])==3:q[int(m.jnt_qposadr[j])]=rng.uniform(-.4,.4)
        d.qpos[:]=q;mujoco.mj_forward(m,d);pos,quat=fk(tree,q)
        assert np.max(np.abs(pos-d.xpos))<1e-5
        assert np.max(np.abs(np.abs(quat)-np.abs(d.xquat)))<1e-5
