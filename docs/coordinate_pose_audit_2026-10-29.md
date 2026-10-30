# Coordinate / pose audit — 2026-10-29

## 結論與範圍

**目前證據支持「預測姿態品質問題，加上 pregrasp 外推敏感性」；未重現系統性的座標、centering、58→29 export 或 wrist FK 錯誤。整體成功率落差的單一根因仍未確定。** 這是目前程式的受控重建檢查，不是原始 inference 執行歷史的完整重播。

本次僅新增四個 `scripts/coordinate_*.py` 診斷工具、本報告及獨立 `output/coordinate_audit_2026-10-29/`。沒有修改 production loader、模型、sample/export、viewer 或 evaluator；沒有訓練、模型推論、BODex synthesis 或動態 rollout。先前已存在的工作目錄變更保留。

依據既有 [matched comparison](../../HUGS-DexGraspBench/docs/bodex_dexlearn_matched_test_comparison.md) 與 [single-hand formatter report](../../HUGS-DexGraspBench/docs/single_hand_learning_formatter_report.md)：同 672 scenes、各 6,720 grasps，DexLearn 331 成功、3,230 initial penetration failures；BODex 2,759 成功、16 initial penetration failures。兩者 pregrasp ratio=2、squeeze ratio=1、penetration threshold=0.01 m。此 audit 不重算成功率。

抽樣 seed=123：6 個真實 train-split 物件的 collected BODex successful training grasps，以及 10 個 held-out 物件各一個同場景 BODex success / DexLearn penetration failure。後者刻意選失敗案例，不能據此估計全部樣本的失敗原因比例。每個姿態檢查 pregrasp、grasp、squeeze。另對四組既有檔案各 6,720 筆做純陣列統計。

## Frame / 函式追蹤

下列 DexLearn 路徑相對本 repo；Bench 路徑相對 `../HUGS-DexGraspBench`。位置為此次檢查時行號。

| 路徑／函式 | 輸入 → 輸出與慣例 |
| --- | --- |
| `dexlearn/dataset/robot_multidex.py:460`, `_transform_complete_pc` | object-local complete PC `(N,3)` → scene/world，逐軸 scale，再旋轉、平移；world 單位 m |
| 同檔 `:508`, `_load_train_data` | collected 三階段 `*_global_pose` 的 world wrist xyz + wxyz quaternion，配 `*_joint_pos`；右手 22 joints、rad；順序 pregrasp/grasp/squeeze |
| 同檔 `:584`, `_load_test_data` | scene metadata + subsampled complete PC → world PC；保存 scene/PC 路徑 |
| 同檔 `:619`, `_apply_pc_centering` | world PC 減當次採樣 centroid；train 的有效右手 wrist xyz 同減 centroid；test 把 centroid 放進回傳 dict |
| 同檔 `:651`, `_apply_rotation_aug` | train centered PC、wrist xyz 及 wrist rotation 同乘旋轉；固定無效左手 slot 不旋轉 |
| 同檔 `__getitem__`, `minkowski_collate_fn` | 旋轉後可加 PC noise；test runtime PC scale 若啟用另生效；sparse coors=PC/voxel，collate 保留 `pc_centroid` |
| `dexlearn/network/backbones/mink_unet.py`, `robot_hierarchical.py` | sparse point/global features 與 grasp-type embedding conditioning；沒有在這裡再加 scene translation |
| `dexlearn/utils/RMS.py`, Normalization | normalization `(x-mean)/std`，inverse `x*std+mean`；不是 scene transform |
| `dexlearn/network/final_layers/diffusion.py:132`, `_sample_single_hand` | 解碼前兩階段 RT、三階段 joints；最後一階段 RT 來自 conditioning 的 `grasp_rt`。此名稱雖叫 grasp，training 使用最後階段 squeeze；oracle 證實 stage ordering 一致 |
| `dexlearn/task/sample.py:456`, `task_sample` | model sample、候選 selection 後，`:530` 使用 batch 的 `pc_centroid` 呼叫 decenter；原始推論 centroid 未儲存在現有 samples |
| 同檔 `:62`, `_decenter_robot_pose` | centered wrist → world wrist；右手加 centroid 一次；right_full 不移動無效左手 slot |
| `dexlearn/utils/logger.py:82`, `save_samples` | 三階段各 `(58,)` 原值轉 CPU NumPy；不再做座標算術 |
| `dexlearn/task/visualize.py:3708`, `solve_stage_pose_batch` | world wrist 透過 dummy-arm IK 得到 URDF root chain；22 named joints；完整右手與 MuJoCo 交叉驗證 |
| 同檔 `:734`, `robot_mesh_filter_for_grasp_type` | fingertip mesh scope 會只顯示指尖；目前 visualize config 使用 fingertips，可造成斷開手指的觀感 |
| Bench `src/task/convert_format.py:464`, `_convert_learning_stage_single` | right_full 58→29，只取原始右手 `[xyz, wxyz, 22 joints]`；不再乘 object pose、不經 dummy-arm IK |
| Bench `src/task/eval_func/base.py:321`, `_apply_eval_pose_adjustments` | world-space stage commands 依 ratio 外推；pregrasp ratio=2 不代表 formatter 改了 pose |
| 同檔 `:471`, `_extrapolate_mocap_qpos_from_grasp` | translation 相對 grasp 外推，orientation 保持該 stage quaternion；finger commands 另按 evaluator 公式外推 |
| 同檔 `:595`, `_eval_simulate_under_extforce` | 初始 hand-object / hand-hand penetration 超 0.01 m 時直接失敗；本 audit 只 static FK/contact，沒有重跑 rollout |
| 同檔 `:724`, `_extract_robot_poses` | `child-rh_palm` FK 輸出 world global pose；關節依 metadata name order 保存 |

### 明確公式

令 row-vector 點雲為 P、scene quaternion 為 **wxyz**、旋轉矩陣為 R、scene translation 為 t、scale 為 s：

```text
P_world = (P_raw * s) @ R.T + t
c = mean(P_world[本次採樣 indices])
P_centered = P_world - c
wrist_centered = wrist_world - c
wrist_world_export = wrist_centered_prediction + c
```

Train 旋轉增強 A：`P_aug = P_centered @ A.T`、`w_aug = A @ w_centered`、`R_wrist_aug = A @ R_wrist`。所以 palm-relative 點座標保持不變。wrist 是 palm/root 的世界 pose，不是額外的 object-relative offset。不能在 formatter 或 viewer 再加 object translation 或 centroid。

58-D 是兩個 29-D slots；right_full/type=3 只使用第一個。MuJoCo free-joint quaternion 同為 wxyz；診斷以 joint names 重排到 MuJoCo order，避免把 metadata array order 當成模型內部順序。URDF dummy joints 只用於 viewer IK，不出現在單手 Bench 29-D。

## 數值檢查

結果見 [case_metrics.json](../output/coordinate_audit_2026-10-29/case_metrics.json)、[population_metrics.json](../output/coordinate_audit_2026-10-29/population_metrics.json)、[viewer_fk_metrics.json](../output/coordinate_audit_2026-10-29/viewer_fk_metrics.json)。完整 scene poses、scale、PC/mesh bounding boxes、joint bounds、wrist distance、三階段接觸深度皆在 JSON。

| 檢查 | 實測最大誤差 | 判定／解讀 |
| --- | ---: | --- |
| loader vs 獨立 float64 scene transform | 4.1761e-8 m；最大 mean error 7.0682e-9 m | PASS，<1e-6 m |
| visualization PC vs loader，同一批點 | 0 m | PASS |
| 實際 train/test loader vs 固定 1,024 indices 的參考 | 0 m | PASS |
| PC centering/inverse | 1.4902e-8 m | PASS，<1e-6 m |
| wrist centering/inverse | 0 m | PASS |
| 旋轉增強後 palm-relative PC invariance | 6.4845e-9 m | PASS |
| known BODex world pose → center → actual decenter → actual formatter | 7.4506e-9 m | PASS |
| sampled complete PC → transformed object mesh surface | 5.0182e-5 m | PASS，<1e-4 m；每場景 256 點，並非全點表面證明 |
| MuJoCo palm FK vs input wrist translation | 0 m | PASS；完整 rotation/global-pose 比較見 JSON |
| actual stage decoder oracle，translation / joints | 0 m / 0 rad | PASS，繞過學習的 network outputs，驗證 packing/unpacking |
| oracle rotation matrix | 2.3842e-7 | PASS，<1e-6 |
| actual Logger save/load 三階段 | 0 | PASS；只寫入獨立 export_control 目錄 |
| existing URDF viewer vs MuJoCo，全右手 sampled surface | 最大 0.3424 mm，最大 p95 0.1133 mm | PASS，<0.5 mm；9 poses、每個 256 vertices 的單向距離 |

URDF IK 9/9 成功；23 個右手 links 的位置最大差約 0.155 mm。直接比較 link rotation matrix 時 `rh_lfmetacarpal` 差異可達約 0.56，不能因此判定手旋轉錯誤：URDF `assets/robot/shadow_hand/dual_dummy_arm_shadow.urdf:396` 後的 LFJ5 frame 有 0.6108649 rad 固定 y 旋轉、LFJ4 再抵消；Bench `assets/hand/shadow/right_hand.xml:182` 使用不同 frame/axis 表達。world mesh 距離確認幾何仍接近。這是抽樣幾何檢查，不是所有可能姿態的等價證明。

### Positive controls 與個案

6 筆 training controls 真正來自 train split 的 collected success；10 筆 matched controls 是 held-out 同場景 BODex baseline success，**不是這些場景的 training ground truth**。成功標籤取自既有 evaluation，沒有再次驗證動態成功。控制實驗繞過 diffusion，驗證 load/center/restore/format/FK 能解讀既有成功姿態。

完整精確來源路徑、training record index 及 matched prediction/evaluation paths：[cases.json](../output/coordinate_audit_2026-10-29/cases.json)。以下 scene 路徑相對 `/home/hao-fang/hugs-dataset/object/DGN_2k/scene_cfg/`。

| # | 種類 | 精確 scene |
| --- | --- | --- |
| 0 | Training success | `core_bottle_3f41aaa42a237fa3c785f06f424b9d06/tabletop_ur10e/scale023_pose000_0.npy` |
| 1 | Training success | `core_bottle_ad33ed7da4ef1b1cca18d703b8006093/tabletop_ur10e/scale006_pose005_0.npy` |
| 2 | Training success | `core_jar_451d7b1633aa8ee741da9a75408092f6/tabletop_ur10e/scale016_pose001_0.npy` |
| 3 | Training success | `mujoco_LEGO_Star_Wars_Advent_Calendar/tabletop_ur10e/scale008_pose002_0.npy` |
| 4 | Training success | `mujoco_Xyli_Pure_Xylitol/tabletop_ur10e/scale011_pose002_0.npy` |
| 5 | Training success | `sem_TableClock_c98d0ef0e3ef0789ad26df5ad8113c0b/tabletop_ur10e/scale006_pose004_0.npy` |
| 6 | Matched control/failure | `core_camera_98fc1afc8dec9773b10c2418bc64b141/tabletop_ur10e/scale003_pose002_0.npy` |
| 7 | Matched control/failure | `core_cellphone_d7c350e5d29074c9b4c216fa1de7d5a1/tabletop_ur10e/scale003_pose000_0.npy` |
| 8 | Matched control/failure | `core_jar_e1835d106d89c69ef1f3019850abf317/tabletop_ur10e/scale003_pose002_0.npy` |
| 9 | Matched control/failure | `ddg_bigbird_chewy_dipps_chocolate_chip/tabletop_ur10e/scale004_pose000_0.npy` |
| 10 | Matched control/failure | `ddg_gd_mushroom_poisson_004/tabletop_ur10e/scale002_pose005_0.npy` |
| 11 | Matched control/failure | `ddg_gd_tetra_pak_poisson_004/tabletop_ur10e/scale002_pose004_0.npy` |
| 12 | Matched control/failure | `ddg_ycb_072_a_toy_airplane/tabletop_ur10e/scale002_pose003_0.npy` |
| 13 | Matched control/failure | `mujoco_Canon_Ink_Cartridge_Green_6/tabletop_ur10e/scale004_pose001_0.npy` |
| 14 | Matched control/failure | `sem_SoapBar_b3718bdc7025004813f5ec8466b8a488/tabletop_ur10e/scale002_pose000_0.npy` |
| 15 | Matched control/failure | `sem_ToiletPaper_dbe344f40aae59e6a9944a6bf3d6e5a2/tabletop_ur10e/scale002_pose004_0.npy` |

以下是初始 **object penetration**（mm），BODex 與 DexLearn evaluated 欄皆含既有 pregrasp=2 調整；raw 欄是已保存 DexLearn 預測的第一個 29-D slot。靜態 contact queries 不推進 simulation。

| 物件 | BODex success | DexLearn raw | DexLearn evaluated |
| --- | ---: | ---: | ---: |
| `core_camera_98fc1afc8dec9773b10c2418bc64b141` | 0.000 | 10.432 | 10.929 |
| `core_cellphone_d7c350e5d29074c9b4c216fa1de7d5a1` | 0.000 | 13.871 | 12.786 |
| `core_jar_e1835d106d89c69ef1f3019850abf317` | 0.000 | 11.151 | 11.123 |
| `ddg_bigbird_chewy_dipps_chocolate_chip` | 0.000 | 18.257 | 12.082 |
| `ddg_gd_mushroom_poisson_004` | 0.000 | 8.434 | 10.389 |
| `ddg_gd_tetra_pak_poisson_004` | 0.000 | 11.677 | 12.602 |
| `ddg_ycb_072_a_toy_airplane` | 0.000 | 8.129 | 10.384 |
| `mujoco_Canon_Ink_Cartridge_Green_6` | 0.000 | 10.853 | 11.699 |
| `sem_SoapBar_b3718bdc7025004813f5ec8466b8a488` | 0.000 | 2.582 | 10.054 |
| `sem_ToiletPaper_dbe344f40aae59e6a9944a6bf3d6e5a2` | 0.000 | 11.205 | 10.478 |

7/10 在 raw prediction 已超過 10 mm；另外 3/10 在 evaluator pregrasp 外推後跨過門檻。BODex controls 的初始 object penetration 均為 0；self-contact 指標另外保存在 JSON，不能把 object penetration 與 self penetration 混為一談。

### 全部既有樣本的陣列統計

每組 6,720 筆；joint-range violation 容忍值 1e-5 rad。這是 command-range 檢查，不等同 rollout 時實際越界。

| 組別 | pregrasp 超界筆數 | grasp | squeeze |
| --- | ---: | ---: | ---: |
| DexLearn raw | 66 | 39 | 164 |
| DexLearn evaluated | 204 | 39 | 164 |
| BODex formatted | 0 | 0 | 978 |
| BODex evaluated | 1,733 | 0 | 978 |

BODex 的 extrapolated command 超界更多，成功率卻較高，因此「關節超界數量」不能單獨解釋成功率落差。各 stage 的 violation size、quaternion norm error、wrist-to-grasp distance 見 population JSON。

每 scene 的 DexLearn 與 BODex 平均 wrist 差為 (-1.86, -1.45, +25.43) mm；每座標 RMS 44.28 mm。將完整 PC centroid 全面加到 DexLearn 會使 RMS 變成 82.41 mm，減掉則為 54.69 mm。**候選 grasps 沒有一對一對應，完整 PC centroid 也不是原始隨機 1,024 點 centroid**；這只能表示目前沒有支持「整批加／減 centroid 就修好」的證據，不能拿來證明歷史推論完全正確。

## 根因排序、限制及下一步

1. **直接觀察：部分預測本身已產生穿透。** 7 個選定失敗 raw poses 已超標；目前 transform/control 檢查通過。模型生成的姿態／軌跡品質是主要待查方向，但未證明全部 3,230 failures 都同因，也未區分模型能力、conditioning 或資料分佈問題。
2. **直接觀察：pregrasp=2 外推可能放大問題。** 3 個選定案例外推後才跨門檻。早期 training-data collection 的 eval config 使用 pregrasp=1，目前 matched eval 兩方法均為 2；這是 training/evaluation 分佈差異。兩方法仍使用相同 matched 設定，不能宣稱 baseline comparison 不公平，也不能只靠它解釋所有失敗。
3. **顯示因素：fingertips scope 會讓完整手看似分離。** 新 viewer 顯示完整右手，且採用 MuJoCo FK。現有 URDF viewer 抽查幾何接近；沒有找到足以解釋厘米級穿透的 dummy-arm offset。
4. **未重現：systematic scene transform、wxyz/xyzw、double/missing decenter、stage ordering、58→29、joint-name mapping。** 這些在目前控制路徑通過，暫無依據修改 production pipeline。

限制：原 experiment `.hydra/config.yaml` 已被後續 visualize 執行覆寫，無法確證當時 proposal count、完整 runtime options；只能確認每 scene 保存 10 筆。原始 sampled PC indices/centroids 未隨 samples 保存。此次 oracle 沒有執行 learned network、checkpoint normalization statistics 或重播當時候選 selection；不能將目前 source-level/controlled roundtrip 結果當成歷史 checkpoint 推論的完整保證。PC/mesh 與 viewer surface distance 是有限抽樣；沒有全面 rollout。

**建議最小下一個實驗（尚未執行）**：在固定小批 matched scenes，以完全相同既有 poses 做 pregrasp ratio=1 vs 2 的 paired static-contact 統計，再視需要做有限 paired rollout。先量化調整敏感性，無需重訓或重新生成 6,720 grasps。目前沒有要求重跑全 benchmark 的依據。

若之後要查歷史 inference，先補新實驗的完整 config、checkpoint hash、PC indices/centroid 及 centering/runtime-scale 記錄；用一個 scene 的固定 seed roundtrip 再檢查 learned normalization/conditioning。這需要另外獲准執行新的 sampling。沒有證據前不加「補 centroid」修正，不改資料或既有結果。未來 production patch 的 regression 應覆蓋非平凡 scene transform、batch/stage decenter、right_full 無效左 slot 不動、oracle stage packing 與 named-joint FK；本次僅提供診斷控制。

## Browser + SSH viewer

入口：[`scripts/coordinate_audit.py`](../scripts/coordinate_audit.py)。已啟動在遠端 `127.0.0.1:8087`，HTTP 回應 200、初始化成功；沒有佔用原 8080 prediction viewer。未做瀏覽器自動點擊測試。

本機開 tunnel（保留執行，替換實際 hostname/IP）：

```bash
ssh -N -L 8087:127.0.0.1:8087 hao-fang@REMOTE_HOST
```

瀏覽器開 `http://localhost:8087`。本機 8087 被占用時改用 `-L 18087:127.0.0.1:8087`，開 `http://localhost:18087`。

遠端 viewer 若已停止，重新啟動：

```bash
cd ~/hugs-workspace/HUGS-DexLearn
export HUGS_DATASET_ROOT="$HOME/hugs-dataset"
../HUGS-DexGraspBench/.venv/bin/python scripts/coordinate_audit.py --serve --port 8087
```

`Dataset / case` 前六筆為真實 training success，其餘為 matched test controls；`Stage` 切三階段，`Pose source` 切 BODex evaluated / DexLearn raw / DexLearn evaluated。粉色完整右手、橘色 complete PC、灰色 object mesh，全部 world frame / meters。Complete PC 顯示時均勻減量至約 4,096 點以維持流暢；並非 partial/camera PC。畫面列出精確來源與 index。可以關閉 object mesh 或 PC 以檢查接觸；只有 static FK，沒有模擬動畫。

靜態對照：[training_and_matched_preview.png](../output/coordinate_audit_2026-10-29/training_and_matched_preview.png)。上列是真實 training sample 三階段；下列是同一 test scene 的三個來源。不同列尺度不同，同列尺度一致。

## 重現與 artifact integrity

在上述 repo 與環境下逐一執行；僅重新寫入專用 audit output。依賴目前兩個 repo 的既有環境與 `tests/visualize_grasp_alignment.py` 幾何工具，不安裝套件。

```bash
../HUGS-DexGraspBench/.venv/bin/python scripts/coordinate_audit.py --audit
../HUGS-DexGraspBench/.venv/bin/python scripts/coordinate_population.py
../HUGS-DexGraspBench/.venv/bin/python scripts/coordinate_viewer_check.py
../HUGS-DexGraspBench/.venv/bin/python scripts/coordinate_preview.py
```

viewer_check 與 population script 使用既有 DexLearn Python 3.10 site-packages 補充依賴；MuJoCo/NumPy/Torch 優先採 Bench 環境。全部採 CPU static reconstruction，不載入模型進行推論。

`read_artifact_sha256.json` 與 `population_read_sha256.json` 分別保存 13,504 與 27,557 個讀取檔案的 SHA-256（後者也包括新建 oracle export）；腳本執行結尾重新檢查這些檔案內容一致。這涵蓋來源 sample/format/evaluation 與選定 training records，不是對整個資料集所有檔案的全域 snapshot。checkpoint 未載入或改寫。完整 logs 在同一 output 目錄；source training data、checkpoint、既有 predictions 與 baseline results 都不是任何寫入目的地。

最終驗證：16 scene transform / palm FK、9 URDF poses、decoder 與 Logger 數值門檻全部通過；四支新腳本 `py_compile` 及 `git diff --check` 通過；viewer 最終 HTTP 檢查為 200。
