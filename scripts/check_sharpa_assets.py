"""Validate standalone Sharpa assets using DexLearn's existing CPU FK and mesh loader."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytorch_kinematics as pk
import torch
import yaml
from mr_utils.robot.pk_visualizer import Visualizer

REPO = Path(__file__).resolve().parents[1]


def main():
    reports = []
    for side in ("right", "left"):
        config = yaml.safe_load((REPO / f"dexlearn/config/robot/sharpa_{side}.yaml").read_text())
        urdf = (REPO / config["urdf_path"]).resolve()
        sys.path.insert(0, str(urdf.parent.parent / "scripts"))
        from validate_models import urdf_fk, require
        original = ET.parse(urdf.parent.parent / "source/urdf" / urdf.name).getroot()
        chain = pk.build_chain_from_urdf(urdf.read_text()).to(dtype=torch.float64, device="cpu")
        names = chain.get_joint_parameter_names()
        require(set(names) == set(config["joint_names"]) and len(names) == 22, "PK joint names mismatch")
        canonical_to_pk = [config["joint_names"].index(name) for name in names]
        moving = {j.get("name"): j for j in original.findall("joint") if j.get("type") != "fixed"}
        limits = np.array([[float(moving[n].find('limit').get(k)) for k in ('lower', 'upper')]
                           for n in config['joint_names']])
        rng = np.random.default_rng(1030)
        samples = np.vstack([np.zeros(22), limits[:, 0], limits[:, 1], rng.uniform(*limits.T, (16, 22))])
        max_error = 0.0
        for q in samples:
            poses = chain.forward_kinematics(torch.tensor(q[canonical_to_pk], dtype=torch.float64))
            expected = urdf_fk(original, dict(zip(config["joint_names"], q)))
            for name, pose in expected.items():
                error = np.max(np.abs(poses[name].get_matrix()[0].numpy() - pose))
                max_error = max(max_error, float(error))
        require(max_error < 2e-6, f"PK FK mismatch: {max_error}")
        visualizer = Visualizer(str(urdf), str((REPO / config["mesh_dir_path"]).resolve()), device="cpu")
        require(len(visualizer.robot_mesh) == 30, "Visual geometry lost")
        pose = torch.zeros((1, 29))
        pose[:, 3] = 1
        pose[:, 7:] = torch.tensor(samples[-1], dtype=torch.float32)
        visualizer.set_robot_parameters(pose, joint_names=config["joint_names"])
        reports.append(dict(side=side, joints=22, visual_links=len(visualizer.robot_mesh),
                            fk_samples=len(samples), max_fk_matrix_error=max_error,
                            canonical_to_pk=canonical_to_pk))
        print(f"{side}: 22 joints, {len(samples)} FK poses and 30 mesh links passed")
    output = REPO / "docs/sharpa_validation.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
