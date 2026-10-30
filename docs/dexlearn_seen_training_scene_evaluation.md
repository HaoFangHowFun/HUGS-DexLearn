# DexLearn seen-training-scene inference & MuJoCo evaluation

Date: 2026-10-29. Completed smoke and full selected-manifest runs. No retraining or production code changes.

## 結果與解讀

**在這批真正具有訓練記錄的場景上，DexLearn 成功 21/260 = 8.08%；仍然偏低。** Unseen 既有結果為 331/6,720 = 4.93%。Seen 的非穿透 rollout 成功率較高，但初始穿透比例也更高；結果不支持把問題單純歸因於 unseen 泛化。它支持繼續檢查 generated-pose/contact quality、模型對訓練資料的 fitting 與 sampling/trajectory 策略，**不單獨證明 underfitting**。

兩組不是 matched scenes/objects：seen 使用 10 個物件、26 個有訓練記錄的 scenes，unseen 是不同的 10 個物件、672 scenes。scene difficulty、物件形狀、scene 數量與未確證的歷史 proposal budget 都可能影響差異，不作因果或顯著泛化改善的宣稱。

| 指標 | Seen training scenes（新） | Unseen test（既有） |
| --- | ---: | ---: |
| Objects / scenes | 10 / 26 | 10 / 672 |
| Saved / formatted / evaluated | 260 / 260 / 260 | 6,720 / 6,720 / 6,720 |
| Success | **21 / 260 (8.08%)** | **331 / 6,720 (4.93%)** |
| Initial penetration rejects | **152 / 260 (58.46%)** | **3,230 / 6,720 (48.07%)** |
| Physical rollouts | 108 | 3,490 |
| Conditional rollout success | **21 / 108 (19.44%)** | **331 / 3,490 (9.48%)** |
| Failed completed rollouts | 87 | 3,159 |
| Scenes with ≥1 success | 13 / 26 (50.00%) | 211 / 672 (31.40%) |
| Missing evaluations | 0 | 0 |

Unseen 數據沿用 [matched comparison report](../../HUGS-DexGraspBench/docs/bodex_dexlearn_matched_test_comparison.md)，本次沒有重跑 unseen 或 BODex baseline。Initial rejection 使用 evaluator 的 `(delta_pos,delta_angle)=(100,100)` sentinel，保留在總 denominator；未把失敗/穿透樣本排除後才報整體成功率。

### 不確定性

Seen candidate-level Wilson 95%：success **5.34%–12.03%**、penetration **52.39%–64.29%**、conditional rollout success **13.08%–27.90%**。Scene-has-success Wilson 為 32.06%–67.94%。這些是描述性的 binomial intervals；同物件/scene 的 candidates 相關，不能將它們當獨立 generalization trials。

另作 object-stratified scene bootstrap（5,000 replicates、seed=0，同一 scene 的全部 10 candidates 一起抽樣、每 object 保留原 scene 數）：success 95% 為 **6.54%–9.62%**。這個區間只條件於固定 10 個物件及已保存 samples；多個物件只有一個 scene，因此不包含其 scene-selection 不確定性，也不包含模型重跑變異或對所有 train objects 的推廣誤差。沒有進行 seen/unseen paired test。

## 場景選擇與 provenance

從 `~/hugs-dataset/BimanBODex_shadow_100k_v1/shadow/right_full/` 取得與 `valid_split/train.json` 相交的 object directories，字典排序後以 `numpy.random.default_rng(0).choice(...,10,replace=False)` 選物件。**在觀察任何成功率之前凍結名單**，每個物件納入全部具 successful records 的 scene files，沒有按成功率重選物件。

選出的物件各只有 1–7 個實際 training scenes，共 26 個，故無法達到建議的每物件 20–100 scenes；沒有用任意 train-object scene 補足數量。每個 source `.npy` 可能包含多個 successful grasp records，manifest 保存所有成功 record indices 與 record_count。

每個 canonical scene path 皆存在、來自對應訓練 record、object 在 train split 且不在 test split。與 6,720 原 unseen predictions 反推出的 672 scene paths 交集為 **0**。這裡的 seen 定義是「存在於實際 training loader 可載入的資料」，沒有逐-step sampler log 證明指定 record 曾在某個 training step 被抽中。

Artifacts（相對 `output/seen_train_right_full_50k_v1_control/`）：

- [full.json](../output/seen_train_right_full_50k_v1_control/full.json)、full.txt：26 個精確 scenes、object IDs、split membership、source file/record indices、scene/source SHA-256。
- [smoke.json](../output/seen_train_right_full_50k_v1_control/smoke.json)、smoke.txt：每物件第一個排序 scene，共 10 scenes。
- [results.json](../output/seen_train_right_full_50k_v1_control/results.json)：總計、逐物件、逐 scene rates/intervals、checkpoint hash。

### 逐物件

| Object ID | Scenes | Success / grasps | Success rate（Wilson 95%） | Penetration | Rollout success | Scenes ≥1 success |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `core_bottle_4185c4eb651bd7e03c752b66cc923fdb` | 7 | 1/70 | 1.43% (0.25%–7.66%) | 46/70 | 1/24 (4.17%) | 1/7 |
| `core_bottle_8adc0ce79962ac2021072d05c97a5e0a` | 3 | 2/30 | 6.67% (1.85%–21.32%) | 16/30 | 2/14 (14.29%) | 2/3 |
| `core_bottle_defc45107217afb846564a8a219239b` | 2 | 0/20 | 0.00% (0.00%–16.11%) | 10/20 | 0/10 (0.00%) | 0/2 |
| `core_can_af444e72a44e6db153c22afefaf6f2a4` | 3 | 2/30 | 6.67% (1.85%–21.32%) | 16/30 | 2/14 (14.29%) | 2/3 |
| `core_jar_b99191dfcaa751ad50e02ef26f47fe02` | 1 | 0/10 | 0.00% (0.00%–27.75%) | 9/10 | 0/1 (0.00%) | 0/1 |
| `core_mug_34869e23f9fdee027528ae0782b54aae` | 1 | 4/10 | 40.00% (16.82%–68.73%) | 4/10 | 4/6 (66.67%) | 1/1 |
| `ddg_kit_SmacksCereals` | 1 | 0/10 | 0.00% (0.00%–27.75%) | 3/10 | 0/7 (0.00%) | 0/1 |
| `mujoco_Perricone_MD_The_Power_Treatments` | 2 | 3/20 | 15.00% (5.24%–36.04%) | 16/20 | 3/4 (75.00%) | 2/2 |
| `sem_FoodItem_7118953c0ef40d01e6ef6db272d5f06` | 5 | 5/50 | 10.00% (4.35%–21.36%) | 29/50 | 5/21 (23.81%) | 4/5 |
| `sem_MilkCarton_fa98055f6f2d0dd916b3af7b23ac8a74` | 1 | 4/10 | 40.00% (16.82%–68.73%) | 3/10 | 4/7 (57.14%) | 1/1 |

### 逐 scene

路徑相對 `~/hugs-dataset/object/DGN_2k/scene_cfg/`；每 scene denominator=10。

| Scene | Success | Penetration | Failed rollout |
| --- | ---: | ---: | ---: |
| `core_bottle_4185c4eb651bd7e03c752b66cc923fdb/tabletop_ur10e/scale008_pose008_0.npy` | 0/10 | 4/10 | 6 |
| `core_bottle_4185c4eb651bd7e03c752b66cc923fdb/tabletop_ur10e/scale011_pose005_0.npy` | 0/10 | 5/10 | 5 |
| `core_bottle_4185c4eb651bd7e03c752b66cc923fdb/tabletop_ur10e/scale011_pose009_0.npy` | 1/10 | 6/10 | 3 |
| `core_bottle_4185c4eb651bd7e03c752b66cc923fdb/tabletop_ur10e/scale013_pose005_0.npy` | 0/10 | 7/10 | 3 |
| `core_bottle_4185c4eb651bd7e03c752b66cc923fdb/tabletop_ur10e/scale016_pose000_0.npy` | 0/10 | 10/10 | 0 |
| `core_bottle_4185c4eb651bd7e03c752b66cc923fdb/tabletop_ur10e/scale019_pose001_0.npy` | 0/10 | 5/10 | 5 |
| `core_bottle_4185c4eb651bd7e03c752b66cc923fdb/tabletop_ur10e/scale019_pose009_0.npy` | 0/10 | 9/10 | 1 |
| `core_bottle_8adc0ce79962ac2021072d05c97a5e0a/tabletop_ur10e/scale006_pose006_0.npy` | 1/10 | 2/10 | 7 |
| `core_bottle_8adc0ce79962ac2021072d05c97a5e0a/tabletop_ur10e/scale013_pose005_0.npy` | 1/10 | 6/10 | 3 |
| `core_bottle_8adc0ce79962ac2021072d05c97a5e0a/tabletop_ur10e/scale019_pose000_0.npy` | 0/10 | 8/10 | 2 |
| `core_bottle_defc45107217afb846564a8a219239b/tabletop_ur10e/scale006_pose007_0.npy` | 0/10 | 3/10 | 7 |
| `core_bottle_defc45107217afb846564a8a219239b/tabletop_ur10e/scale011_pose003_0.npy` | 0/10 | 7/10 | 3 |
| `core_can_af444e72a44e6db153c22afefaf6f2a4/tabletop_ur10e/scale011_pose002_0.npy` | 1/10 | 4/10 | 5 |
| `core_can_af444e72a44e6db153c22afefaf6f2a4/tabletop_ur10e/scale013_pose003_0.npy` | 0/10 | 6/10 | 4 |
| `core_can_af444e72a44e6db153c22afefaf6f2a4/tabletop_ur10e/scale013_pose008_0.npy` | 1/10 | 6/10 | 3 |
| `core_jar_b99191dfcaa751ad50e02ef26f47fe02/tabletop_ur10e/scale023_pose000_0.npy` | 0/10 | 9/10 | 1 |
| `core_mug_34869e23f9fdee027528ae0782b54aae/tabletop_ur10e/scale008_pose000_0.npy` | 4/10 | 4/10 | 2 |
| `ddg_kit_SmacksCereals/tabletop_ur10e/scale008_pose003_0.npy` | 0/10 | 3/10 | 7 |
| `mujoco_Perricone_MD_The_Power_Treatments/tabletop_ur10e/scale016_pose003_0.npy` | 1/10 | 8/10 | 1 |
| `mujoco_Perricone_MD_The_Power_Treatments/tabletop_ur10e/scale019_pose003_0.npy` | 2/10 | 8/10 | 0 |
| `sem_FoodItem_7118953c0ef40d01e6ef6db272d5f06/tabletop_ur10e/scale006_pose001_0.npy` | 2/10 | 1/10 | 7 |
| `sem_FoodItem_7118953c0ef40d01e6ef6db272d5f06/tabletop_ur10e/scale016_pose003_0.npy` | 0/10 | 8/10 | 2 |
| `sem_FoodItem_7118953c0ef40d01e6ef6db272d5f06/tabletop_ur10e/scale019_pose000_0.npy` | 1/10 | 8/10 | 1 |
| `sem_FoodItem_7118953c0ef40d01e6ef6db272d5f06/tabletop_ur10e/scale019_pose001_0.npy` | 1/10 | 4/10 | 5 |
| `sem_FoodItem_7118953c0ef40d01e6ef6db272d5f06/tabletop_ur10e/scale019_pose003_0.npy` | 1/10 | 8/10 | 1 |
| `sem_MilkCarton_fa98055f6f2d0dd916b3af7b23ac8a74/tabletop_ur10e/scale008_pose004_0.npy` | 4/10 | 3/10 | 3 |

## Sampling 與 checkpoint 正確性

Checkpoint 原路徑：`output/shadowMulti_robotMultiHierar_right_full_100k_v1/ckpts/step_050000.pth`；production `task_sample` strict `model.load_state_dict` 成功、checkpoint iteration=50,000。沒有複製/改寫 weights 或 optimizer。

SHA-256：`b4d6a241ace57e4ca4f265fd6733cd72d0094645970451b91189b15d73f60ff6`。

設定：`robotMultiHierar` / `shadowMulti`、complete PC、1,024 points、centering=true、test runtime PC scale 預設 1、right_full metadata、seed=0、batch_size=256、test_split=train，加上精確 `test_scene_list_path` manifest。沒有使用 eval mode（該 mode 可對應 test.json），也沒有僅靠 train object split 再 glob 任意 scenes。

模型 architecture/config 與保留下來的原 experiment visualize config **逐欄相同**（`model_config_comparison.json`）；checkpoint strict loading 再驗證參數匹配。DDIM scheduler 與原 sampling console 的設定一致：128 train timesteps、squaredcos_cap_v2、v_prediction、clip_sample=false；新 run 明確設定 10 inference timesteps。

**歷史限制**：原 experiment `.hydra` 曾被 visualize 覆寫，原 sampling console 沒記錄 inference timestep override、候選數與完整 selection overrides，因此不能保證歷史 runtime options 完全相同。新 run 使用現存一致的 model config 與明確保存的 10-step / topk 設定。原 unseen 只能獨立確認每 scene 保存 10 筆；不宣稱 generation compute 完全 matched。新 run 改 num_workers=0、preload=false 以便 bounded deterministic instrumentation；這會影響 PC/RNG realization 與效能，但不更改 scene transform 或模型 weights。

### 實際 proposal 數量，不只看 CLI 名稱

`test_grasp_num=20` 是 **每 grasp type 20 candidates**。沿用 `0_any` availability sampling 時，實作會為全部 5 個 real types 都計算候選，因此每 scene 的內部 tensor 實際是 **100 × 3 × 58**。Availability masking 後，本 run 每個 scene 都只保存 type=3 的 top 10，選自該 type 的 20 個候選。

Full：內部 2,600 candidates，right_full proposal budget 520，保存/評估 260。Smoke：內部 1,000，right_full budget 200，保存 100，取 20 做 smoke evaluation。這避免把 `test_grasp_num=20` 錯稱為總共只計算 20 個候選。

Selection=`enabled=true, scope=per_type, mode=topk, test_topk=10`。Logger 使用 `candidate_valid_mask` 把選中的 sparse candidate indices 壓成連續檔名編號；例如 `_0.npy` 不必對應內部 candidate 0。檢查時以完整 pose 精確匹配回原 tensor，而不是假設檔名序號等於候選索引。

所有 260 保存結果 `pred_grasp_type_id=3`，三階段皆 `(58,)` 且 finite；每個 manifest scene 正好 10 個，無額外、遺漏或重複 scenes。

### 沒有 ground-truth leakage；真實推論輸出還原檢查

新入口 [seen_training_sample.py](../scripts/seen_training_sample.py) 僅在獨立 process 掛 observational hooks，隨後執行原 production Hydra sample task，不修改 production source。model.sample 輸入 keys 實際為 `save_path, pc_path, pc_centroid, point_clouds, grasp_type_id, scene_path, coors, feats, original2quantize, quantize2original`；assert 沒有 right/left hand labels、record index 或 training type targets。`grasp_type_id=0` 是固定 inference request，不是來源 grasp label；network conditioning 是 PC features 加模型的 type sampling。

每個 run 的 `reconstruction_trace/batch_000.{json,npz}` 保存實際 centroid、centered pose、world pose、input keys 與 candidate shape。觀察 actual `_decenter_robot_pose`：右手 xyz 等於 centered xyz + 該 batch centroid，最大誤差 **0 m**；右手 quaternion/joints 完全不變。Logger 接收 tensors 與 world trace 相同；每個已保存 candidate 的三階段 58-D 值能精確匹配 trace，所選 raw indices 全在 type=3 的 40–59 範圍。

因此這次不只是 oracle roundtrip：**真正 learned prediction 的 export 還原已被記錄與檢查**。未記錄原始完整 PC indices；若需跨環境逐 bit 重播仍需保存 input PC/RNG 狀態，這不影響本次直接觀察到的加回 centroid 檢查。

## MuJoCo protocol 與 smoke gate

沿用已驗證 Learning runner：`--hand shadow --grasp-type right_full --format-arg +task.learning_metadata_group=right_full`。完整 `task`、`hand`、`setting` 已與 `output/bodex_matched_test_v1/resolved_configs/dexlearn_eval_hydra.yaml` 比對：full 相同，smoke 只將 task.max_num 改成 20。同 hand asset 與目前 evaluator source：

- `tabletop` / `assets/hand/shadow/right_hand_v2.xml`、mocap=true。
- pregrasp ratio=2.0、squeeze ratio=1.0、global z offset=0。
- penetration threshold=0.01 m、translation threshold=0.05 m、angle threshold=15°、approach=false。
- density policy=scale_power，ref 700 kg/m³ @ scale 0.06、exponent=-2、mass cap=0.5 kg；friction=[0.8,0.02]。
- timestep=0.004 s、各 grasp/squeeze/lift phase 25×10 steps、lift=0.1 m、downward force=10*object_mass；同 evaluator NumPy/Python seed=12。

Smoke generation 10 scenes/100 saved；format/eval 20 candidates 完整對應，0 success、8 initial rejections、12 failed rollouts。**Smoke gate 是檔案、格式、finite poses、設定與完整執行正確，不是要求成功率超過門檻**；沒有因 0 success 重選資料。Smoke 20 筆由原 runner 的 max_num 選取規則決定，涵蓋 10 scenes，精確檔案留在 smoke run。

通過後獨立重新 sample 全 26 scenes；full 與 smoke 的 RNG consumption/batch 不同，不能當成同 scene 的同組 candidates。最終報告只用 full 的 260 筆，不把 smoke 加入 denominator。

每個 converted 29-D stage 與 raw prediction 前 29 個值 **exact equality**，22 joint names 與 metadata 相同，object pose/scale 與 source scene 相同；每個 formatted record 都有 evaluation。Logs：0 read failures、0 skipped pre-existing evaluations，未見 evaluation exceptions、nonfinite failures 或 tiny-mesh skips。

### 新樣本的獨立 static FK / penetration 檢查

[static_pose_check.json](../output/seen_train_right_full_50k_v1_control/static_pose_check.json) 對全部 260 candidates、三個 stages 用獨立 MuJoCo reconstruction：palm position vs saved wrist 最大差 **0 m**，rotation matrix 最大差 **6.6614e-16**，quaternion norm error 最大 **8.6908e-8**。這是 static FK/contact，不是重跑 rollout。

使用與 evaluator 相同的 hand-object 或 hand-hand penetration >0.01 m 判準：

| 狀態 | Rejections |
| --- | ---: |
| Raw saved pregrasp | 182 / 260 (70.00%) |
| Evaluator-adjusted pregrasp | 152 / 260 (58.46%) |
| 原本 raw 通過、調整後拒絕 | 1 |
| 原本 raw 拒絕、調整後通過 | 31 |

獨立 adjusted static contact flags 與 260 個既有 evaluation flags **零不一致**。本批 ratio=2 實際淨減少穿透拒絕 30 個，不能沿用先前刻意挑選 failure cases 的現象，宣稱外推一定惡化穿透。高穿透已存在於 learned raw output；本次沒有 evidence 支持 missing/double centroid、wxyz/xyzw 或 wrist FK offset 作為其來源。Raw penetration flags 不是 ratio=1 的動態成功率，不能用來替代另一次 paired rollout。

## 執行時間與吞吐

下列為記錄的 subprocess wall time，包含啟動、模型載入、格式處理及 I/O；不是純 GPU kernel benchmark。

| Run | Sampling wall / saved per second | Format wall | Evaluation wall / evaluated per second |
| --- | --- | --- | --- |
| smoke | 2.517 s / 39.72 | 1.336 s | 1.620 s / 12.35 |
| full | 2.614 s / 99.46 | 1.430 s | 3.167 s / 82.09 |

GPU 為既有 RTX 5090；n_worker=8 evaluation、format n_worker=4、OMP_NUM_THREADS=2。確切 start timestamps、argv、cwd、env、exit codes 及 logs 保存在 commands.jsonl 和各 *_timing.json。

## 輸出、重現與驗證

新 run IDs：

- DexLearn smoke：`output/shadowMulti_robotMultiHierar_seen_train_right_full_50k_v1_smoke/`
- DexLearn full：`output/shadowMulti_robotMultiHierar_seen_train_right_full_50k_v1_full/`
- Bench smoke：`output/seen_train_right_full_50k_v1_smoke_right_full_shadow/`
- Bench full：`output/seen_train_right_full_50k_v1_full_right_full_shadow/`
- Experiment manifest/config/results/logs：`HUGS-DexLearn/output/seen_train_right_full_50k_v1_control/`

各 DexLearn run 的 `.hydra/{config,overrides}.yaml` 保存該次 sampling settings，沒有在這些 run 中啟動 visualize。Bench 的 `log/format`、`log/eval` 保存 resolved Hydra configs。Git HEAD：DexLearn `a27a912`、DexGraspBench `c97f84c`；完整 revisions/起始 dirty state 見 git_state.json；本次新增 scripts/tests/docs，沒有把先前既有 production modifications 當成本次變更。

實際執行順序（從 DexLearn repo）：

```bash
.venv/bin/python scripts/seen_training_scene_experiment.py prepare
.venv/bin/python scripts/seen_training_scene_experiment.py run --size smoke --stage sample
.venv/bin/python scripts/seen_training_scene_experiment.py audit --size smoke
.venv/bin/python scripts/seen_training_scene_experiment.py run --size smoke --stage format
.venv/bin/python scripts/seen_training_scene_experiment.py run --size smoke --stage eval
.venv/bin/python scripts/seen_training_scene_experiment.py audit --size smoke
.venv/bin/python scripts/seen_training_scene_experiment.py run --size full --stage sample
.venv/bin/python scripts/seen_training_scene_experiment.py audit --size full
.venv/bin/python scripts/seen_training_scene_experiment.py run --size full --stage format
.venv/bin/python scripts/seen_training_scene_experiment.py run --size full --stage eval
.venv/bin/python scripts/seen_training_scene_experiment.py audit --size full
OMP_NUM_THREADS=2 .venv/bin/python scripts/seen_training_pose_check.py
.venv/bin/python scripts/summarize_seen_training_scene_experiment.py
```

這些已完成。prepare/run 在 manifest、log 或 output 已存在時拒絕覆寫；要重新生成請先在實驗腳本更換 CONTROL 與所有 run IDs，保持原 checkpoint 不變。不要對這些已完成目錄使用 cleanup/--yes。audit、summarize、tests 可直接重跑（僅更新專用診斷 JSON）。每個 child stage 的完整原始命令見 commands.jsonl，不依賴猜測的 CLI。

Tests：

```bash
.venv/bin/python -m unittest discover -s tests -p test_robot_multidex_scene_list.py -v
.venv/bin/python -m unittest discover -s tests -p test_seen_training_scene_experiment.py -v
```

共 **6 tests passed**：現有 manifest selection 測試 3 個（精確 paths、重複/數量錯誤、禁用額外隨機 scene slicing），新增 3 個（seed/provenance、split 零重疊、真實 inference restore/Logger 保存一致）。新 scripts py_compile 與 git diff --check 通過。格式/full-count/protocol checks 另由 experiment audit 執行。

SHA-256 再檢查 **36,744 protected paths 全部未變**，包含原 checkpoint/metadata、6,720 unseen predictions、既有 learned/BODex benchmark npy 與本次來源 scenes/training records。這不是全 workspace snapshot；未修改訓練資料、original scenes、checkpoint 或任何既有推論/評估結果。

## 建議下一步

目前可排除「只在 unseen 才壞」作為唯一解釋；seen 也有大量 raw-pose penetration，且這次實際 sampling restoration 已通過。下一個小實驗可分別量化 **模型對固定 training targets 的 fitting** 與 **候選品質/選擇**（例如保存所有 right_full proposals 的 score/contact diagnostics，檢查 top-k 是否偏好穿透姿態）。此診斷尚未執行，不必立即重訓或擴大到全 benchmark。若要改 training loss、selection policy 或 evaluator protocol，先提出獨立 patch/experiment，保留本次結果。
