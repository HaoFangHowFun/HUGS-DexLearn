"""Compare existing URDF/IK viewer FK with independent MuJoCo FK on 9 poses."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]/'.venv/lib/python3.10/site-packages'))
from coordinate_audit import *

def main():
    torch.manual_seed(123)
    cases=json.loads((OUT/'cases.json').read_text())
    selections=[(cases[0],0),(cases[6],0),(cases[6],1)]
    cfg=OmegaConf.create({'device':'cpu','test_data':{'robot_urdf_path':str(REPO/'assets/robot/shadow_hand/dual_dummy_arm_shadow.urdf'),'robot_mesh_dir_path':str(REPO/'assets/robot/shadow_hand'),'metadata_group':'right_full','grasp_path':str(TRAIN.parent)}})
    assets=vis.build_robot_components(cfg); result=[]
    for c,idx in selections:
        d,names,q=poses(c['variants'][idx]); s,obj,raw,pc,mesh,ho=scene_data(c)
        two=np.concatenate([q,np.tile([0,0,-.5,1,0,0,0]+[0]*22,(3,1))],axis=1)
        robot,ok,_=vis.solve_stage_pose_batch(two,assets,'cpu')
        vv=assets['robot_visualizer']; vv.set_robot_parameters(robot,joint_names=assets['robot_helper'].robot_joint_names)
        row={'case':c['label'],'variant':c['variants'][idx]['label'],'ik_success':ok,'stages':{}}
        for j,stage in enumerate(STAGES):
            set_pose(ho,names,q[j],obj); pos=[]; rot=[]; rotations={}
            for link,tf in vv.current_status.items():
                if not link.startswith('rh_'):continue
                try: body=ho.model.body('child-'+link).id
                except KeyError:continue
                matrix=tf.get_matrix()[j].detach().numpy()
                pos.append(float(np.linalg.norm(matrix[:3,3]-ho.data.xpos[body])))
                rot.append(float(abs(matrix[:3,:3]-ho.data.xmat[body].reshape(3,3)).max()))
                rotations[link]=rot[-1]
            urdf_mesh=vv.get_robot_trimesh_data(i=j,include_link_prefixes=('rh_',))
            mj_mesh=geometry_from_mujoco(ho,True)
            distances=distance_stats(mj_mesh,urdf_mesh.vertices[np.linspace(0,len(urdf_mesh.vertices)-1,256,dtype=int)])
            row['stages'][stage]={'world_urdf_vertices_to_mujoco_surface_m':distances,'frame_rotation_by_link':rotations,'links':len(pos),'max_link_position_error_m':max(pos),'max_link_rotation_matrix_error':max(rot)}
        result.append(row)
    dump(OUT/'viewer_fk_metrics.json',result); print(json.dumps(result,indent=2,default=lambda x:x.tolist()))

if __name__=='__main__':main()
