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


def test_browser_fk_matches_mujoco(tmp_path):
    """The page's own forward kinematics (static/js/fk.js) against MuJoCo, free joints included."""
    import json, shutil, subprocess
    from pathlib import Path
    import pytest
    if not shutil.which('node'):
        pytest.skip('node is not installed')
    root = Path(__file__).resolve().parents[1]
    m = mujoco.MjModel.from_xml_path('third_party/mujoco_menagerie/unitree_g1/g1_with_hands.xml')
    d = mujoco.MjData(m)
    rng = np.random.default_rng(7)
    q = m.qpos0.copy()
    for j in range(m.njnt):
        a = int(m.jnt_qposadr[j])
        if int(m.jnt_type[j]) == 3:
            q[a] = rng.uniform(-.4, .4)
        elif int(m.jnt_type[j]) == 0:
            q[a:a + 3] += rng.uniform(-.2, .2, 3)
            turn = rng.normal(size=4)
            q[a + 3:a + 7] = turn / np.linalg.norm(turn)
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    case = tmp_path / 'case.json'
    case.write_text(json.dumps({'tree': export_tree(m), 'q': q.tolist()}))
    js = ("import fs from 'fs'; import {Kinematics} from './kaiku/web/static/js/fk.js';"
          "const c = JSON.parse(fs.readFileSync(process.argv[1])); const k = new Kinematics(c.tree);"
          "k.solve(Float64Array.from(c.q)); console.log(JSON.stringify({pos: [...k.xpos], mat: [...k.xmat]}));")
    r = subprocess.run(['node', '--input-type=module', '-e', js, str(case)], cwd=root, text=True, capture_output=True, check=True)
    out = json.loads(r.stdout)
    assert np.max(np.abs(np.asarray(out['pos']).reshape(-1, 3) - d.xpos)) < 1e-5
    assert np.max(np.abs(np.asarray(out['mat']).reshape(-1, 9) - d.xmat)) < 1e-5
