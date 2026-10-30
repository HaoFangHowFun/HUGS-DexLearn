"""Integration checks for actual DataLoader captures. Run --audit first.
Uses real dataset/scene labels as independent reference, not mocked tensors.
"""
import json
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import debug_dataloader_grasp_alignment as audit


class DataLoaderAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=json.loads((audit.DEBUG_OUT/'items.json').read_text())
        assert len(cls.rows)==18, 'Run the full diagnostic capture first'

    def test_scene_transform_from_original_files(self):
        for row in self.rows:
            with self.subTest(case=row['case_index'],mode=row['mode']), np.load(audit.DEBUG_OUT/row['artifact']) as a:
                scene=np.load(row['case']['scene'],allow_pickle=True).item()
                obj=scene['scene'][scene['task']['obj_name']]
                raw=np.load(audit.DATA/'object/DGN_2k/processed_data'/scene['task']['obj_name']/'complete_point_cloud.npy')
                ref=(raw[a['indices']].astype(float)*np.asarray(obj['scale']))@audit.quat2mat(obj['pose'][3:]).T+obj['pose'][:3]
                np.testing.assert_allclose(a['precenter'],ref,rtol=0,atol=1e-5)

    def test_center_rotation_noise_inverse_and_forward(self):
        for row in self.rows:
            with self.subTest(case=row['case_index'],mode=row['mode']), np.load(audit.DEBUG_OUT/row['artifact']) as a:
                expected=(a['precenter']-a['centroid'])@a['augmentation'].T+a['noise']
                np.testing.assert_allclose(a['pc'],expected,rtol=0,atol=1e-5)
                pc,_,_=audit.inverse(a['pc'],a['trans'],a['rot'],a['centroid'],a['augmentation'],a['noise'])
                np.testing.assert_allclose(pc,a['reference_pc'],rtol=0,atol=1e-5)
                # Negative control: forgetting centroid must be detected.
                self.assertGreater(float(abs(pc-a['centroid']-a['reference_pc']).max()),1e-5)

    def test_all_stages_against_original_labels(self):
        for row in self.rows:
            source=row['case']['variants'][0]
            data=np.load(source['path'],allow_pickle=True).item()
            with np.load(audit.DEBUG_OUT/row['artifact']) as a:
                _,t,R=audit.inverse(a['pc'],a['trans'],a['rot'],a['centroid'],a['augmentation'],a['noise'])
                for j,stage in enumerate(audit.STAGES):
                    original=data[stage+'_global_pose'][source['index']]
                    np.testing.assert_allclose(t[j],original[:3],rtol=0,atol=1e-5)
                    self.assertLess(audit.geodesic(R[j],audit.quat2mat(original[3:])),1e-4)
                    np.testing.assert_allclose(a['joints'][j],data[stage+'_joint_pos'][source['index']],rtol=0,atol=1e-5)
                    self.assertLess(row['metrics']['stages'][stage]['palm_fingertip_fk_max_m'],1e-5)

    def test_sparse_collation_and_float32_model_boundary(self):
        b=json.loads((audit.DEBUG_OUT/'batch.json').read_text())
        self.assertTrue(b['collation_values_unchanged']);self.assertTrue(b['sparse_mapping_pass'])
        self.assertEqual(b['keys']['right_hand_joint']['shape'],[18,1,3,22])
        for e in b['post_InfLoader_inverse']:
            self.assertLess(e['pc_max_m'],1e-5);self.assertLess(e['wrist_max_m'],1e-5);self.assertLess(e['rotation_max_rad'],1e-4)

    def test_joint_order_negative_control(self):
        row=self.rows[0];case=row['case'];src=case['variants'][0]
        d,names,q=audit.poses(src);_,obj,_,_,_,ho=audit.scene_data(case)
        audit.set_pose(ho,names,q[1],obj)
        ids=[ho.model.body('child-rh_'+f+'distal').id for f in ['ff','mf','rf','lf','th']]
        good=ho.data.xpos[ids].copy()
        audit.set_pose(ho,list(reversed(names)),q[1],obj)
        self.assertGreater(float(abs(good-ho.data.xpos[ids]).max()),1e-3)

    def test_heldout_control_and_input_integrity(self):
        e=json.loads((audit.DEBUG_OUT/'heldout.json').read_text())
        self.assertLess(e['oracle_export_max_m'],1e-5);self.assertEqual(e['logger_max_error'],0)
        for p,h in json.loads((audit.DEBUG_OUT/'read_sha256.json').read_text()).items():
            self.assertEqual(audit.hashlib.sha256(Path(p).read_bytes()).hexdigest(),h,p)


if __name__=='__main__':unittest.main()
