# Team Fractal (Chandrashekhar Paliwal & Pratham) — SO-101 Sim-to-Real Prototype Summary
**Date & Venue:** 18 Nov 2026, BIEC Bengaluru | **System Designation:** Engineering **Prototype**  
**Core Attribution:** Built on **Hugging Face LeRobot 0.4.4**, **NVIDIA Isaac Lab**, and the **NWaves Platform**.

---

## 1. Problem
Bridging zero-shot and few-shot RGB-only Action Chunking with Transformers (`ACT`, `lerobot==0.4.4`) control at **30 Hz (`< 33 ms/step`)** from **NVIDIA Isaac Lab** to physical **Feetech STS3215** servos on the 6-DoF SO-101 follower arm across:
- Arbitrary bright cube colours and randomized spawn/tray poses.
- Physical `0.52 m` overhead (`scene`) + `wrist` UVC camera extrinsics and auto-exposure tolerances.
- Millimetre-level tabletop and tray boundary offsets under real venue lighting.

---

## 2. Approach (Data Engine · NVIDIA Isaac Lab · Sim-to-Real)
1. **NVIDIA Isaac Lab Data Engine (`600` Curated Episodes):**
   - Generated **600 clean, trajectory-shaped simulation episodes** in **NVIDIA Isaac Lab** featuring strict **24–32 mm tray wall clearance**, domain-randomized bright cube colours and tray poses, and zero hover noise — achieving the **#1 Stage-1 simulation score (`10 / 10`, `5.8 s` mean completion time)**.
2. **Forgetting-Safe Sim + Real Co-Training (`Hugging Face LeRobot 0.4.4` + `NWaves Platform`):**
   - Warm-started from our Stage-1 ACT checkpoint while co-training simulated episodes alongside `ArunMurugesan/real_pickplace_v4` and live BIEC **My Robot** demonstrations pulled via `nwaves-realdata`.
   - Verified via `nwaves-eval` prior to `nwaves-submit` so rare cube colours and edge poses are never catastrophically forgotten during real-arm adaptation.
3. **Hardware-Safe Execution (`NWaves My Robot` — Route B):**
   - Deployed through the **NWaves Platform (`My Robot` Route B)** with servo stall/overload protection, dry-run validation, 40% initial speed verification, and calibrated joint travel verification (`189°–204°` arm joints, `360°` wrist roll, `~139°` gripper).

---

## 3. Results (Stage 1 Verified & Stage 2 Live BIEC Scorecard — 15:45 IST)

| Evaluation Metric | Score / Attempts | Mean Completion Time |
|---|---|---|
| **Stage-1 Simulation Score (`nwaves-eval` / Leaderboard)** | **`10 / 10`** (#1 Stage-1 Rank) | **`5.8 s`** |
| **Real-Arm 3-Point (Sim-Only Policy) Attempts** | `[  _____  /  _____  ]` | `[  _____._ s  ]` |
| **Real-Arm 2-Point (Sim + Real Co-Trained) Attempts** | `[  _____  /  _____  ]` | `[  _____._ s  ]` |
| **Total Real-Arm Score & Overall Mean Completion Time** | **`[  _________ pts  ]`** | **`[  _____._ s  ]`** |

---

## 4. Honest Prototype Limits
1. **RGB-Only Depth Sensitivity to Extrinsics:** Monocular RGB depth inference from the `0.52 m` overhead scene camera is sensitive to **`> 1 cm` camera mount shifts** if the mount is bumped between calibration and evaluation.
2. **Physical STS3215 Gear Backlash & Thermal Cutoff:** Mechanical gear play (`~1.2°`) and continuous high-load thermal cutoff (**`> 60°C`**) on `shoulder_lift` and `elbow_flex` require compliant grasp bounds and rest intervals between runs.
3. **UVC Webcam Auto-Exposure Shifts:** Direct venue hall spotlights over glossy cube facets and light plywood induce hardware auto-exposure swings across episodes.
