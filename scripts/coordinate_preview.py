"""Static previews using the same read-only MuJoCo FK as the dataset viewer."""
from coordinate_audit import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def main():
    cases=json.loads((OUT/'cases.json').read_text()); fig=plt.figure(figsize=(15,10)); groups=[]
    for row,case in enumerate([cases[0],cases[6]]):
        s,obj,raw,pc,mesh,ho=scene_data(case); entries=[]
        for j in range(3):
            variant=case['variants'][0] if row==0 else case['variants'][j]
            st=j if row==0 else 0
            _,names,q=poses(variant); set_pose(ho,names,q[st],obj)
            hand=geometry_from_mujoco(ho,True); entries.append((hand,variant['label']+' / '+STAGES[st]))
        points=np.concatenate([pc]+[hand.vertices for hand,_ in entries]); mid=(points.max(0)+points.min(0))/2; rad=np.ptp(points,axis=0).max()*.55
        for j,(hand,title) in enumerate(entries):
            ax=fig.add_subplot(2,3,row*3+j+1,projection='3d')
            ax.add_collection3d(Poly3DCollection(hand.triangles,facecolor='#d4779e',alpha=.35,edgecolor='none'))
            ax.scatter(*pc[::max(1,len(pc)//1500)].T,s=1,c='#f89800',depthshade=False)
            ax.set(xlim=(mid[0]-rad,mid[0]+rad),ylim=(mid[1]-rad,mid[1]+rad),zlim=(mid[2]-rad,mid[2]+rad),xlabel='world X (m)',ylabel='world Y (m)',zlabel='world Z (m)',title=title)
            ax.set_box_aspect((1,1,1)); ax.view_init(elev=20,azim=-65)
    fig.suptitle('Full hand: independent MuJoCo FK; orange: complete point cloud\nTop: actual training data. Bottom: matched test scene, pregrasp.',fontsize=13)
    fig.tight_layout(); fig.savefig(OUT/'training_and_matched_preview.png',dpi=150)
    print(OUT/'training_and_matched_preview.png')

if __name__=='__main__':main()
