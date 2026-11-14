# The dataset

Kaiku's recorder saves every clean run: the operator's and each echo's. `kaiku.data.export` turns a
recorder folder into the humanoid challenge's two layouts.

```bash
python -m kaiku.data.export DEMOS OUT --limit 20      # the first 20 demonstrations
```

- `OUT/lerobot/` is LeRobot v3.0, `OUT/groot/` is LeRobot v2.1 with GR00T's modality file.
- One demonstration is one episode: 28 joint targets and 28 joint states at 30 Hz in the pelvis frame.
  Arm velocity is the central difference of the measured arm angles.
- `observation.environment_state` is where the parts are in every frame: 5 parts times position and
  quaternion, in the order of the episode's `layout`, zeros for parts a scene does not have. The
  simulator gives it. With the robot state it is what the shipped policy reads, so a policy like
  ours can be trained from the download alone.
- `meta/aalto_episodes.jsonl` says who did each episode: `source` is `operator` or `echo`, and
  `driven_by` names the operator (the scripted stand-in, a visitor's pointer or camera hand, real
  Smartgloves). It also holds the scene's layout and which bin each colour goes to.
- The cameras (head, left and right wrist, 640x480) are rendered from the MuJoCo twin with the
  challenge's calibration only when asked for (`--render-cameras`; `tools/modal_export.py` does it
  on a GPU). The page's download has no video, and its `meta/info.json` lists none. There is no depth yet.
