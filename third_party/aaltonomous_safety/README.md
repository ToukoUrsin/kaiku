# Aaltonomous official safety runtime

This directory vendors the challenge runtime from `aaltonomous/interface` in the
[Aaltonomous Humanoid challenge repository](https://github.com/zupermiez/aaltonomous-humanoid):
`safety.py` is copied unchanged (MIT), as are its generated `joint_spec.yaml` and
`LICENSE`. The checked-in challenge runtime has no separate safety YAML; its
`SafetyConfig` defaults are the active configuration, and match the checked-in
`aaltonomous/scene/scene.yaml` workspace/counter values.

`g1_body29_hand14.urdf` is the G1 29-DoF + Dex3 model named by the runtime's
`DEFAULT_URDF`, fetched from Unitree's `xr_teleoperate` assets. It is a 30 KB FK-only
copy: `<visual>` and `<collision>` mesh elements were removed because Pinocchio FK
needs links, joints, and inertials but no meshes. The source asset is revision
`f7bdf9ce4819a050010ac745c720062cde9a2f63` (the current upstream asset containing
this path); the challenge's generated joint spec records its original checkout
revision in `joint_spec.yaml`.
