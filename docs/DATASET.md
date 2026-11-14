# The dataset

Kaiku's recorder saves every clean run: the operator's and each echo's. `kaiku.data.export` turns a
recorder folder into the humanoid challenge's two layouts.

```bash
python -m kaiku.data.export DEMOS OUT --limit 20      # the first 20 demonstrations
```

- `OUT/lerobot/` is LeRobot v3.0, `OUT/groot/` is LeRobot v2.1 with GR00T's modality file.
- One demonstration is one episode: 28 joint targets and 28 joint states at 30 Hz in the pelvis frame.
  Arm velocity is the central difference of the measured arm angles.
- The cameras (head, left and right wrist, 640x480) are rendered from the MuJoCo twin with the
  challenge's calibration when asked for; `tools/modal_export.py` does it on a GPU. There is no depth yet.
