"""Inspect one successful right_full grasp in world coordinates.

Run with HUGS-DexGraspBench/.venv/bin/python (MuJoCo, rtree and Viser).
No dataset/config/label changes; --headless writes numerical results only.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh
from omegaconf import OmegaConf
from transforms3d.quaternions import quat2mat

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
sys.path.insert(0, str(REPO))
from dexlearn.dataset.robot_multidex import RobotMultiDexDataset
from dexlearn.utils.rot import numpy_quaternion_to_matrix

STAGES = ('pregrasp', 'grasp', 'squeeze')
DEFAULT_SCENE = 'core_bottle_46910c75d17ef8554c6e2707eea5c15e/tabletop_ur10e/scale008_pose005_0'


def load(path):
    return np.load(path, allow_pickle=True).item()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def distance_stats(mesh, points):
    # Chunk exact point-to-triangle queries to bound memory; no nearest-vertex approximation.
    # Trimesh uses absolute tolerances for triangle queries. Condition both operands
    # identically, then convert distances back; displayed/input geometry is untouched.
    length = float(np.max(mesh.extents))
    origin = mesh.bounds.mean(0)
    query_mesh = mesh.copy()
    query_mesh.vertices = (mesh.vertices - origin) / length
    query_points = (points - origin) / length
    distances = length * np.concatenate([
        trimesh.proximity.closest_point(query_mesh, chunk)[1]
        for chunk in np.array_split(query_points, max(1, int(np.ceil(len(points) / 512))))
    ])
    return dict(mean=float(distances.mean()), rms=float(np.sqrt(np.mean(distances ** 2))),
                median=float(np.median(distances)), p95=float(np.quantile(distances, .95)),
                max=float(distances.max()), count=len(points))


def transform_mesh(mesh, scale, rotation, translation):
    result = mesh.copy()
    result.vertices = (np.asarray(mesh.vertices) * scale) @ rotation.T + translation
    return result


def geometry_from_mujoco(ho, hand):
    """Compiled mesh vertices already include MJCF mesh scale and compiler transforms."""
    import mujoco
    meshes = []
    for gid in range(ho.model.ngeom):
        if ho.model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        name = ho.model.geom(gid).name
        is_hand = ho.model.body(ho.model.geom_bodyid[gid]).name.startswith('child-')
        if hand:
            if not is_hand or ho.model.geom_group[gid] != 2:
                continue
        elif not name.startswith('object_collision_'):
            continue
        mid = ho.model.geom_dataid[gid]
        va, vn = ho.model.mesh_vertadr[mid], ho.model.mesh_vertnum[mid]
        fa, fn = ho.model.mesh_faceadr[mid], ho.model.mesh_facenum[mid]
        vertices = ho.model.mesh_vert[va:va + vn].copy()
        faces = ho.model.mesh_face[fa:fa + fn].copy()
        vertices = vertices @ ho.data.geom_xmat[gid].reshape(3, 3).T + ho.data.geom_xpos[gid]
        meshes.append(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))
    if not meshes:
        raise ValueError('No matching MuJoCo mesh geometry')
    return trimesh.util.concatenate(meshes)


def provenance(args, grouped, index, scene, obj_cfg):
    result = {}
    evaluation_dir = args.bench_run / 'evaluation' / args.scene_id
    matches = []
    for path in sorted(evaluation_dir.glob('*.npy')):
        record = load(path)
        if all(np.array_equal(record[f'{s}_qpos'], grouped[f'{s}_qpos'][index]) for s in STAGES):
            matches.append(path)
    result['matching_evaluation_files'] = matches
    if len(matches) != 1:
        result['limitation'] = 'Expected one exact evaluation match; source-candidate chain not established.'
        return result
    evaluation = load(matches[0])
    candidate = int(matches[0].name.split('_')[0])
    formatted_path = args.bench_run / 'graspdata' / args.scene_id / matches[0].name
    original_path = args.bodex_grasp_root / f'{args.scene_id}_grasp.npy'
    result.update(candidate_index=candidate, evaluation_success=bool(evaluation['succ_flag']),
                  formatted_path=formatted_path, bodex_path=original_path)
    result['evaluation_object_pose_max_abs_delta'] = float(np.max(np.abs(evaluation['obj_pose'] - obj_cfg['pose'])))
    result['evaluation_object_scale_delta'] = float(evaluation['obj_scale'] - obj_cfg['scale'][0])
    result['evaluation_scene_reference_matches'] = str(evaluation['scene_path']) == str(grouped['scene_path'])
    if not formatted_path.exists() or not original_path.exists():
        result['limitation'] = 'Original BODex or formatted artifact unavailable.'
        return result
    original, formatted = load(original_path), load(formatted_path)
    world_meshes = original['world_cfg'][0]['mesh']
    world_object = world_meshes[args.scene_id + scene['task']['obj_name']]
    result['saved_bodex_object_pose_max_abs_delta'] = float(np.max(np.abs(world_object['pose'] - obj_cfg['pose'])))
    result['saved_bodex_object_scale_max_abs_delta'] = float(np.max(np.abs(world_object['scale'] - obj_cfg['scale'])))
    result['saved_bodex_surface_path'] = world_object['file_path']
    result['bodex_scene_reference_matches'] = str(original['scene_path'][0]) == str(grouped['scene_path'])
    result['bodex_joint_names_match'] = list(original['joint_names']) == list(grouped['joint_names'][index])
    result['stage_deltas'] = {}
    for si, stage in enumerate(STAGES):
        source = np.asarray(original['robot_pose'][0, candidate, si])
        fmt = np.asarray(formatted[f'{stage}_qpos'])
        saved = np.asarray(grouped[f'{stage}_qpos'][index])
        result['stage_deltas'][stage] = {
            'bodex_to_formatted_qpos_max_abs': float(np.max(np.abs(source - fmt))),
            'formatted_to_success_qpos_max_abs': float(np.max(np.abs(fmt - saved))),
            'formatted_to_success_translation_m': (saved[:3] - fmt[:3]).tolist(),
        }
    return result


def training_validation(dataset, grouped, index, local_pc, scale, rotation, translation, angle):
    poses = np.stack([grouped[f'{s}_global_pose'][index] for s in STAGES])[None]
    joints = np.stack([grouped[f'{s}_joint_pos'][index] for s in STAGES])[None]
    targets = dict(right_hand_trans=poses[..., :3].copy(),
                   right_hand_rot=numpy_quaternion_to_matrix(poses[..., 3:7]),
                   right_hand_joint=joints.copy(), left_hand_fixed=True,
                   left_hand_trans=np.tile([0., 0., -.5], (1, 3, 1)),
                   left_hand_rot=np.tile(np.eye(3), (1, 3, 1, 1)),
                   left_hand_joint=np.zeros_like(joints))
    before = copy.deepcopy(targets)
    np.random.seed(0)
    sampled = dataset._transform_complete_pc(dataset._sample_point_cloud(local_pc), scale, rotation, translation)
    centroid = sampled.mean(0, keepdims=True)
    centered = dataset._apply_pc_centering(sampled.copy(), targets, grouped)
    center_pc_error = float(np.max(np.abs(centered - (sampled - centroid))))
    center_hand_error = float(np.max(np.abs(targets['right_hand_trans'] - (poses[..., :3] - centroid))))
    theta = np.deg2rad(angle)
    rz = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    with patch('numpy.random.uniform', return_value=theta):
        augmented = dataset._apply_rotation_aug(centered, targets)
    # Compare relative object positions in each stage's palm frame before/after.
    errors = []
    for si in range(3):
        original_local = (sampled - before['right_hand_trans'][0, si]) @ before['right_hand_rot'][0, si]
        augmented_local = (augmented - targets['right_hand_trans'][0, si]) @ targets['right_hand_rot'][0, si]
        errors.append(float(np.max(np.abs(original_local - augmented_local))))
    return dict(sample_count=len(sampled), centroid_m=centroid[0], z_rotation_degrees=angle,
                centering_point_max_abs_error_m=center_pc_error, centering_hand_max_abs_error_m=center_hand_error,
                augmented_point_max_abs_error_m=float(np.max(np.abs(augmented - centered @ rz.T))),
                augmented_rotation_max_abs_error=float(np.max(np.abs(targets['right_hand_rot'] - rz @ before['right_hand_rot']))),
                palm_frame_point_invariance_max_abs_error_m_by_stage=errors,
                fixed_left_unchanged=all(np.array_equal(targets[k], before[k]) for k in
                                         ('left_hand_trans', 'left_hand_rot', 'left_hand_joint')),
                right_joints_unchanged=np.array_equal(targets['right_hand_joint'], before['right_hand_joint']))


def build(args):
    sys.path.insert(0, str(args.bench_repo / 'src'))
    from util.hand_util import MjHO
    grasp_root = args.dataset_root / 'BimanBODex_single_hand_demo_v2/shadow/right_full'
    grasp_path = grasp_root / f'{args.scene_id}.npy'
    grouped = load(grasp_path)
    index = args.grasp_index
    if not 0 <= index < len(grouped['succ_flag']) or not bool(grouped['succ_flag'][index]):
        raise ValueError('Select an in-range successful grasp record')
    if str(grouped['grasp_type'][index]) != 'right_full':
        raise ValueError('This inspection script requires right_full')
    scene_path = args.dataset_root / str(grouped['scene_path'])
    scene = load(scene_path)
    if scene['scene_id'] != args.scene_id or scene_path.resolve() != (args.dataset_root / 'object/DGN_2k/scene_cfg' / f'{args.scene_id}.npy').resolve():
        raise ValueError('Scene path/ID mismatch')
    obj_name = args.scene_id.split('/')[0]
    if scene['task']['obj_name'] != obj_name:
        raise ValueError('Object ID mismatch')
    obj_cfg = scene['scene'][obj_name]
    object_root = args.dataset_root / 'object/DGN_2k/processed_data' / obj_name
    mesh_path = (scene_path.parent / obj_cfg['file_path']).resolve()
    pc_path = object_root / 'complete_point_cloud.npy'
    local_pc = np.load(pc_path)
    dataset = RobotMultiDexDataset.__new__(RobotMultiDexDataset)
    dataset.mode = 'train'
    dataset.config = OmegaConf.create({'num_points': 1024})
    _, scale, rotation, translation = dataset._extract_object_meta(scene, str(scene_path))
    world_pc = dataset._transform_complete_pc(local_pc, scale, rotation, translation)
    independent_rot = quat2mat(np.asarray(obj_cfg['pose'][3:7], dtype=np.float64))
    independent_pc = (local_pc.astype(np.float64) * obj_cfg['scale']) @ independent_rot.T + obj_cfg['pose'][:3]
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    world_mesh = transform_mesh(mesh, obj_cfg['scale'], independent_rot, obj_cfg['pose'][:3])
    if not np.allclose(obj_cfg['scale'], obj_cfg['scale'][0], rtol=0, atol=0):
        raise ValueError('Benchmark uses scalar scale; anisotropic scene cannot be represented exactly')
    hand_xml = args.bench_repo / 'assets/hand/shadow/right_hand_v2.xml'
    ho = MjHO(obj_path=str(object_root), obj_scale=float(obj_cfg['scale'][0]), obj_density=1000,
              mj_arena_memory_bytes=None, hand_xml_path=str(hand_xml), hand_mocap=True,
              exclude_table_contact=None, friction_coef=[1.0, 0.01], has_floor_z0=True)
    names = list(grouped['joint_names'][index])
    mj_names = ho.get_joint_names()[1:]
    if len(names) != 22 or set(names) != set(mj_names):
        raise ValueError('Shadow 22-joint name contract mismatch')
    order = [names.index(name) for name in mj_names]
    hands, wrist_checks = {}, {}
    for stage in STAGES:
        qpos = np.asarray(grouped[f'{stage}_qpos'][index], dtype=np.float64).copy()
        qpos[7:] = qpos[7:][order]
        ho.reset_pose_qpos(qpos, obj_cfg['pose'], set_ctrl=False)
        body = ho.model.body('child-rh_palm').id
        recorded = grouped[f'{stage}_global_pose'][index]
        wrist_checks[stage] = dict(
            translation_max_abs_error_m=float(np.max(np.abs(ho.data.xpos[body] - recorded[:3]))),
            rotation_matrix_max_abs_error=float(np.max(np.abs(ho.data.xmat[body].reshape(3, 3) - quat2mat(recorded[3:7])))),
            joint_targets_max_abs_error=float(np.max(np.abs(grouped[f'{stage}_qpos'][index][7:] - grouped[f'{stage}_joint_pos'][index]))),
            quaternion_norm=float(np.linalg.norm(recorded[3:7])))
        hands[stage] = geometry_from_mujoco(ho, hand=True)
    collision = geometry_from_mujoco(ho, hand=False)
    pc_bounds = np.stack([world_pc.min(0), world_pc.max(0)])
    report = dict(scene_id=args.scene_id, object_id=obj_name, grasp_index=index, succ_flag=True,
                  units='meters; joint angles radians; quaternions wxyz',
                  distance_query='Joint translation/scaling of mesh and queries for numerical conditioning; distances converted back to meters.',
                  files=dict(grasp=grasp_path, scene=scene_path, mesh=mesh_path, point_cloud=pc_path, hand_xml=hand_xml),
                  sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (grasp_path, scene_path, mesh_path, pc_path, hand_xml)},
                  object_scale=obj_cfg['scale'], object_pose_wxyz=obj_cfg['pose'],
                  point_count=len(local_pc), point_cloud_storage_dtype=str(local_pc.dtype),
                  float16_rounding_envelope_max_norm_m=float(np.max(np.linalg.norm(np.abs(np.spacing(local_pc).astype(np.float64)) * .5 * obj_cfg['scale'], axis=1))),
                  object_quaternion_norm=float(np.linalg.norm(obj_cfg['pose'][3:7])),
                  dexlearn_vs_float64_world_transform_max_abs_error_m=float(np.max(np.abs(world_pc - independent_pc))),
                  point_to_bodex_surface_m=distance_stats(world_mesh, world_pc),
                  point_to_benchmark_convex_piece_surfaces_m=distance_stats(collision, world_pc),
                  local_point_to_bodex_surface_normalized_units=distance_stats(mesh, local_pc),
                  world_bounds_m=dict(point_cloud=pc_bounds, bodex_mesh=world_mesh.bounds, benchmark_collision=collision.bounds),
                  point_cloud_minus_mesh_bounds_m=pc_bounds - world_mesh.bounds,
                  benchmark_minus_bodex_bounds_m=collision.bounds - world_mesh.bounds,
                  wrist_fk=wrist_checks, skipped_benchmark_meshes=ho.skipped_tiny_object_meshes,
                  training=training_validation(dataset, grouped, index, local_pc, scale, rotation, translation, args.rotation_degrees),
                  provenance=provenance(args, grouped, index, scene, obj_cfg))
    # A seeded mesh-surface sample provides context for finite point-cloud bounding boxes.
    np.random.seed(0)
    sample, _ = trimesh.sample.sample_surface(world_mesh, len(world_pc))
    report['same_count_seed0_mesh_sample_minus_mesh_bounds_m'] = np.stack([sample.min(0), sample.max(0)]) - world_mesh.bounds
    return report, world_pc, world_mesh, collision, hands


def save_preview(path, report, pc, mesh, collision, hands):
    """Export a world-frame overview with identical axes for all three stages."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(15, 5))
    all_vertices = np.concatenate([mesh.vertices] + [hand.vertices for hand in hands.values()])
    center = (all_vertices.min(0) + all_vertices.max(0)) / 2
    radius = np.max(np.ptp(all_vertices, axis=0)) * .55
    for i, stage in enumerate(STAGES):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        for geometry, color, alpha in [(mesh, '#9ca3af', .3), (hands[stage], '#b95f6c', .65)]:
            ax.add_collection3d(Poly3DCollection(geometry.triangles, facecolor=color,
                                               edgecolor='none', alpha=alpha, rasterized=True))
        ax.scatter(*pc.T, c='#08b8df', s=.4, alpha=.6, depthshade=False)
        for direction, color in zip(np.eye(3) * .05, ['r', 'g', 'b']):
            ax.quiver(0, 0, 0, *direction, color=color, linewidth=1)
        ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
               zlim=(center[2]-radius, center[2]+radius), xlabel='world X (m)', ylabel='world Y (m)', zlabel='world Z (m)', title=stage)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=24, azim=-55)
    fig.suptitle('Recorded Shadow right hand / BODex surface / DexLearn points (cyan)')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def visualize(args, report, pc, mesh, collision, hands):
    import viser
    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction('+z')
    server.scene.add_frame('/world', axes_length=.1, axes_radius=.001, origin_radius=.002)
    server.scene.add_mesh_simple('/object/surface', vertices=mesh.vertices, faces=mesh.faces,
                                 color=(160, 165, 180), opacity=.45)
    collision_handle = server.scene.add_mesh_simple('/object/benchmark_collision', vertices=collision.vertices,
                          faces=collision.faces, color=(240, 160, 60), opacity=.25, visible=False)
    server.scene.add_point_cloud('/object/dexlearn_points', points=pc, colors=(40, 210, 250), point_size=.0015)
    hand_handles = {stage: server.scene.add_mesh_simple(f'/hand/{stage}', vertices=hand.vertices, faces=hand.faces,
                    color=(190, 80, 95), visible=stage == args.stage) for stage, hand in hands.items()}
    server.gui.add_markdown(f"**{args.scene_id}**  \nSuccessful record {args.grasp_index}; world frame, meters.  \nCyan: DexLearn points; gray: BODex surface; pink: recorded hand.")
    stage = server.gui.add_dropdown('Stage', options=STAGES, initial_value=args.stage)
    collision_toggle = server.gui.add_checkbox('Benchmark convex pieces', initial_value=False)
    @stage.on_update
    def update_stage(_):
        for name, handle in hand_handles.items():
            handle.visible = name == stage.value
    @collision_toggle.on_update
    def update_collision(_):
        collision_handle.visible = collision_toggle.value
    @server.on_client_connect
    def camera(client):
        center = mesh.bounds.mean(0)
        client.camera.look_at = tuple(center)
        client.camera.position = tuple(center + np.array([.35, -.45, .3]))
    print(f'VIEWER_READY http://{args.host}:{server.get_port()}', flush=True)
    if args.viewer_seconds:
        time.sleep(args.viewer_seconds)
        server.stop()
    else:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=Path(os.environ.get('HUGS_DATASET_ROOT', Path.home() / 'hugs-dataset')))
    parser.add_argument('--scene-id', default=DEFAULT_SCENE)
    parser.add_argument('--grasp-index', type=int, default=0)
    parser.add_argument('--stage', choices=STAGES, default='grasp')
    parser.add_argument('--bench-repo', type=Path, default=WORKSPACE / 'HUGS-DexGraspBench')
    parser.add_argument('--bench-run', type=Path, default=WORKSPACE / 'HUGS-DexGraspBench/output/single_hand_demo_v2_right_full_shadow')
    parser.add_argument('--bodex-grasp-root', type=Path, default=WORKSPACE / 'HUGS-BODex/src/curobo/content/assets/output/sim_shadow/tabletop_full/single_hand_demo_v2/graspdata')
    parser.add_argument('--rotation-degrees', type=float, default=37.)
    parser.add_argument('--report', type=Path, default=REPO / 'output/grasp_alignment/metrics.json')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--snapshot', type=Path, help='Optional three-stage PNG overview.')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8087)
    parser.add_argument('--viewer-seconds', type=float, default=0, help='Exit viewer after this duration (smoke testing).')
    args = parser.parse_args()
    result = build(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result[0], indent=2, default=json_default) + '\n')
    print(json.dumps(result[0], indent=2, default=json_default), flush=True)
    if args.snapshot:
        save_preview(args.snapshot, *result)
    if not args.headless:
        visualize(args, *result)


if __name__ == '__main__':
    main()
