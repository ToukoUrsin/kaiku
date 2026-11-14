# Kaiku exported datasets

`kaiku.data.export` reads the immutable recorder snapshot at
`/bulk2/runs/robotteleop/policy/demos_snap/` and writes datasets under
`/bulk2/runs/robotteleop/export/<name>/`. Each recorder JSONL row becomes one episode;
the arrays are clipped to `span`, and arm velocity is a 30 Hz central difference of
measured arm angles. `groot/` is the LeRobot v2.1 + GR00T modality layout and
`lerobot/` is the direct LeRobot v3-compatible layout. Camera rendering uses the
MuJoCo twin and the challenge calibration; depth is intentionally omitted until a
render run is affordable.

For a local low-dimensional check:

```bash
.venv/bin/python -m kaiku.data.export /bulk2/runs/robotteleop/policy/demos_snap \
  /bulk2/runs/robotteleop/export/check20 --limit 20 --overwrite
```

For the camera export, run `tools/modal_export.py` through `/tmp/kd/modal.sh`; the
entrypoint uploads the code/model bundle, the Modal volume contains the snapshot at
`/data/demos`, and the output persists there.
