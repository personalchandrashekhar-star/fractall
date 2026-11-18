# Team Fractal — SO-101 Sim-to-Real ACT Manipulation (`Prototype`)

[![Stage 1 Rank](https://img.shields.io/badge/Stage_1_Simulation-%231_Seed_(10%2F10_%40_5.8s)-1d4ed8?style=flat-square)](#5-evaluation-results--honest-prototype-limits)
[![Stage 2 Real Arm](https://img.shields.io/badge/Stage_2_Physical_SO--101-2%2F3_Makes_(7.67%2F10_Overall)-15803d?style=flat-square)](#5-evaluation-results--honest-prototype-limits)
[![Stack](https://img.shields.io/badge/LeRobot-v0.4.4_(ACT)-f59e0b?style=flat-square)](https://github.com/huggingface/lerobot)
[![Simulator](https://img.shields.io/badge/NVIDIA-Isaac_Lab-76b900?style=flat-square)](https://developer.nvidia.com/isaac/lab)
[![Status](https://img.shields.io/badge/Designation-Research_Prototype-111827?style=flat-square)](./Team_Fractal_Jury_Summary.pdf)

**Team:** Chandrashekhar Paliwal & Pratham Mehta  
**Event:** Bharat1.ai × NWaves SO-101 Sim-to-Real Buildathon (Stage 2 Finalist — BIEC Bengaluru, 18 Nov 2026)  
**Designation:** Research & Engineering **Prototype**  
**1-Page Jury Hand-In Dossier (PDF):** [`Team_Fractal_Jury_Summary.pdf`](./Team_Fractal_Jury_Summary.pdf)

> **Explicit Platform & Open-Source Attribution:**  
> This **Prototype** is built upon and gratefully credits **Hugging Face LeRobot 0.4.4** (Action Chunking with Transformers policy architecture & normalizer pipeline), **NVIDIA Isaac Lab** (600-episode USD domain-randomized simulation data engine & multi-seed evaluator), and the **NWaves Platform** (`nwaves-train`, `nwaves-eval`, `nwaves-realdata`, and **My Robot** real-arm execution).

![Physical SO-101 Follower Arm at BIEC Bengaluru](./biec_so101_real.jpg)

---

## 1. Problem Statement
Executing autonomous **30 Hz (`< 33 ms/step`)** RGB-only 6-DoF tabletop pick-and-place (`"Put the cube on the tray."`) on physical **Feetech STS3215** servos from dual UVC cameras (`0.52 m` overhead `scene` + close-range `wrist` camera, `640×480 @ 30 fps`) across:
1. **Multi-Colour & Pose Generalization:** Picking arbitrary bright PLA cube colours across randomized table placements and moved tray locations without depth sensors or AprilTags.
2. **Physical UVC Camera & Mount Tolerances:** Bridging synthetic rendering to auto-exposure UVC webcams, hand-mounted `0.52 m` overhead extrinsics, and millimetre-level table/tray offsets under venue hall lighting.
3. **Preventing Catastrophic Forgetting:** Adapting to 30 real-arm teleop demonstrations without erasing Stage-1 multi-colour visual priors or corrupting joint normalizer statistics.

---

## 2. Technical Approach & Sim-to-Real Engineering

### A. NVIDIA Isaac Lab Data Engine (`600` Curated Episodes → `#1` Stage-1 Seed)
- Engineered a trajectory-shaped **NVIDIA Isaac Lab** data generator (`scripts/generate.py`, `configs/datagen.yaml`) producing **600 curated episodes** (`ppprock11/so101_pickplace_training_v2`).
- Enforced strict **24–32 mm vertical tray wall clearance**, randomized bright cube colours, lighting intensity/temperature, camera pose jitter, and zero hover hesitation at release — achieving the **#1 Stage-1 Simulation Score (`10 / 10` successes, `5.8 s` mean completion time)**.

### B. Four Key Sim-to-Real Fixes That Enabled Physical Execution at BIEC
1. **Strict LeRobot v3.0 Feature & Task Schema Parity (`cspaliwal/so101_real_finetune_v5_scene`):**
   - Standard `lerobot-record` teleop setups default to `observation.images.top` and `so_follower`, which silently mismatches Stage-1's `observation.images.scene` visual encoder. We built an automated Parquet/JSON/video schema normalizer (`scripts/pack_stage2_real.py`) that remaps `observation.images.top` → `observation.images.scene`, enforces `"robot_type": "so101_follower"`, and aligns `meta/tasks.parquet` to `"Put the cube on the tray."`.
2. **Joint & Action Normalizer Recalibration (`policy_preprocessor` / `policy_postprocessor`):**
   - Warm-starting `lerobot-train` (`--policy.path`) from our 35,000-step Stage-1 ACT checkpoint (`batch_size=8`, `3,000` steps) preserved the multi-colour ResNet-18 + Transformer representations while recomputing `policy_preprocessor_step_3_normalizer_processor.safetensors` and `policy_postprocessor_step_0_unnormalizer_processor.safetensors` for the physical follower arm's joint calibration.
3. **Eliminating Temporal-Ensembling Lag & Gripper Smearing (`policy/policy.py`):**
   - Naive exponential temporal ensembling across 30-step chunks smears the gripper dimension (`index 5`) between open (`~38.0`) and closed (`~6.5`), causing hover hesitation above the cube. We replaced smeared gripper averaging with direct chunk execution and clean newest-observation weighting.
4. **Stall-Safe Physical STS3215 Servo Bounds (`policy/inference.json`):**
   - Configured non-snapping compliant gripper limits (`gripper_min=6.0`, `gripper_max=58.0`, `snap_gripper=false`) under **NWaves My Robot (Route B)** so physical jaws close with compliant contact force and release without triggering STS3215 current/thermal overload trips.

---

## 3. Repository Structure

```text
fractall/
├── Team_Fractal_Jury_Summary.pdf   # 1-Page Printable A4 Jury Hand-In Dossier (Prototype)
├── Team_Fractal_Jury_Summary.html  # Helvetica Magazine A4 HTML Source
├── biec_so101_real.jpg             # Physical SO-101 Follower Arm & Target Tray at BIEC
├── configs/
│   └── datagen.yaml                # NVIDIA Isaac Lab domain randomization parameters
├── docs/
│   ├── Team_Fractal_Jury_Summary.md
│   └── Team_Fractal_Pitch_Script.md
├── nwaves_hackathon/               # Isaac Lab task env, IK, tray geometry, randomizer & server
├── policy/
│   ├── policy.py                   # Stage-2 ACT runtime wrapper (cuda -> mps -> cpu)
│   ├── config.json                 # LeRobot 0.4.4 ACT model configuration
│   ├── inference.json              # Action chunk horizon & safe physical gripper bounds
│   ├── policy_preprocessor.json    # State/visual normalizer pipeline config
│   └── policy_postprocessor.json   # Action unnormalizer pipeline config
└── scripts/
    ├── generate.py                 # 600-episode trajectory-shaped Isaac Lab data engine
    ├── evaluate.py                 # Multi-seed headless/GUI simulation evaluator
    ├── teleop.py                   # Leader-follower & keyboard teleoperation recorder
    ├── to_lerobot.py               # Episode converter to LeRobot v3.0 dataset format
    └── pack_stage2_real.py         # Validator-clean checkpoint packager & smoke tester
```

---

## 4. Reproducing Data Generation, Training & Deployment

```bash
# 1. Generate 600 domain-randomized simulation episodes in NVIDIA Isaac Lab
isaaclab -p scripts/generate.py --episodes 600 --config configs/datagen.yaml --name sim_pickplace_v2
nwaves-dataset sim_pickplace_v2 ppprock11/so101_pickplace_training_v2

# 2. Train Stage-1 Base ACT Policy (35,000 steps, batch_size=8)
nwaves-train ppprock11/so101_pickplace_training_v2 --policy act --steps 35000 --batch 8 --name best_success_35k_original

# 3. Warm-Start Fine-Tune on Normalized BIEC Real-Arm Demonstrations (3,000 steps, batch_size=8)
hf download cspaliwal/so101_real_finetune_v5_scene --repo-type dataset --local-dir ~/workspace/lerobot_datasets/v5_scene

lerobot-train \
  --policy.path=~/workspace/outputs/best_success_35k_original/checkpoints/last/pretrained_model \
  --dataset.repo_id=cspaliwal/so101_real_finetune_v5_scene \
  --dataset.root=~/workspace/lerobot_datasets/v5_scene \
  --policy.device=cuda --policy.push_to_hub=false \
  --steps=3000 --batch_size=8 --save_freq=3000 \
  --output_dir=~/workspace/outputs/act_35k_v5_fixed

cp policy/policy.py ~/workspace/outputs/act_35k_v5_fixed/checkpoints/last/pretrained_model/policy.py

# 4. Evaluate & Submit to NWaves My Policies
nwaves-eval --policy ~/workspace/outputs/act_35k_v5_fixed
nwaves-submit --name act_35k_v5_fixed.zip --note "35k warmstart + v5_scene fixed" ~/workspace/outputs/act_35k_v5_fixed
```

---

## 5. Evaluation Results & Honest Prototype Limits

| Evaluation Stage | Score / Makes | Mean Time / Notes |
|---|---|---|
| **Stage-1 Simulation (`nwaves-eval` / Official Leaderboard)** | **`10 / 10`** (`100%`) | **`5.8 s`** (#1 Stage-1 Seed Rank) |
| **Stage-2 Physical SO-101 Arm at BIEC (`act_35k_v5_fixed`)** | **`2 / 3 Makes`** (`66.7%`) | **Only Finalist Team to Score on Physical Hardware** |
| **Combined Overall Score (`30%` Sim + `70%` Real)** | **`7.67 / 10`** | `3.00 / 3.00` Sim + `4.67 / 7.00` Real |
| **Policy Inference Latency (`25`-step chunk on GPU)** | **`< 23 ms`** | Strict `< 33 ms` (`30 Hz`) compliance |

### Honest Engineering Limits of This Prototype
1. **`0.52 m` Scene Camera Extrinsics Sensitivity (`> 1 cm` Shift):** Monocular RGB depth inference from the overhead camera degrades if the camera mount shifts by `> 1 cm` relative to the calibrated tabletop frame before close-range wrist-camera acquisition.
2. **Physical STS3215 Gear Backlash & Thermal Cutoff (`> 60°C`):** `~1.2°` mechanical gear play and continuous-load servo heating (`> 60°C` thermal protection) require non-snapping compliant gripper bounds and brief cooling intervals between runs.
3. **UVC Webcam Auto-Exposure Swings:** Direct BIEC hall spotlights reflecting off glossy PLA cube facets and white tabletop surfaces induce hardware auto-exposure shifts across consecutive trials.

---

## 6. Checkpoints & Datasets on Hugging Face
- **Stage-1 Base Checkpoint (`35k`):** [`ppprock11/so101_best_success`](https://huggingface.co/ppprock11/so101_best_success)
- **Stage-2 Deployment Packages & Scripts:** [`cspaliwal/so101-stage2-deploy`](https://huggingface.co/cspaliwal/so101-stage2-deploy)
- **Stage-2 Normalized Real-Arm Dataset (`30` BIEC Episodes, `scene` + `wrist`):** [`cspaliwal/so101_real_finetune_v5_scene`](https://huggingface.co/datasets/cspaliwal/so101_real_finetune_v5_scene)
