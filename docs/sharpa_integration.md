# Sharpa FK and mesh validation

Place `HUGS-Sharpa-Assets` beside this repository. The standalone configs in
`dexlearn/config/robot/sharpa_{right,left}.yaml` define portable URDF/mesh paths,
flange root, canonical joint names, and fingertip link names.

```bash
.venv/bin/python scripts/check_sharpa_assets.py
```

The check uses the existing pytorch_kinematics and Visualizer on CPU, maps canonical
joint names explicitly, compares 19 FK poses per hand with independent URDF FK,
and loads all 30 visual links. It writes `docs/sharpa_validation.json`.
No MuJoCo install is needed in this environment. The parser's unknown `mujoco`
extension warning is expected; it does not affect the kinematic tree.

These are **not dataset/training presets**. Existing full visualization and metadata
consumers still assume Shadow-style `rh_`/`lh_` names and dummy arms in places.
Sharpa metadata, dataset contracts and those consumers must be adapted before the
full pipeline is enabled. Shadow checkpoints are not compatible merely because
Sharpa also has 22 DoFs per hand. No training was started.

See the sibling asset directory's `SHADOW_AUDIT.md` for the integration audit and
`provenance.json` for exact source hashes.
