# Team Fractal — 90-Second Jury Pitch Script (BIEC Stage 2 Finalist)
**Speakers:** Chandrashekhar Paliwal & Pratham  
**Target Awards:** Overall Winner + Tie-Breaker #1 (Presentation) + ₹5,00,000 Special Awards (*Best Data Engine*, *Best Sim-to-Real*, *Best Use of NVIDIA Isaac Lab*)

---

## 1. Opening & Explicit Attribution (0:00 – 0:20)
> "Hi Judges — we’re **Team Fractal** (Chandrashekhar and Pratham). What we’re demonstrating today is an engineering **Prototype** for 30 Hz RGB-only pick-and-place on the 6-DoF SO-101 arm, built on **Hugging Face LeRobot 0.4.4**, **NVIDIA Isaac Lab**, and the **NWaves Platform**."

## 2. The Data Engine & #1 Stage-1 Seed (0:20 – 0:45) — *Targets: Best Data Engine & Best Use of Isaac Lab*
> "Instead of brute-forcing noisy teleop in simulation, we built a trajectory-shaped **NVIDIA Isaac Lab Data Engine** of **600 curated episodes**. We enforced a strict **24 to 32 mm vertical clearance** over the tray walls, randomized bright cube colours and tray poses, and eliminated hover hesitation at release. That clean data distribution gave our ACT policy the **#1 Stage-1 simulation score: 10 out of 10 with a 5.8-second mean completion time**."

## 3. Forgetting-Safe Sim + Real Co-Training & Per-Attempt Strategy (0:45 – 1:10) — *Targets: Best Sim-to-Real*
> "On the physical Feetech STS3215 arm here at BIEC, pure real-data fine-tuning on 25 demos causes catastrophic forgetting of unseen cube colours. To solve that inside the **NWaves Platform**, we warm-started from our Stage-1 ACT checkpoint and **co-trained our 600 Isaac Lab episodes alongside `ArunMurugesan/real_pickplace_v4` and our live BIEC `My Robot` recordings** (`nwaves-realdata`), validating every checkpoint with `nwaves-eval` before `nwaves-submit`. Under **Section 11 per-attempt rules**, we pair our 3-point zero-shot sim policy on center spawns with our 2-point co-trained policy on edge and moved-tray placements."

## 4. Results & Honest Prototype Limits (1:10 – 1:30) — *Targets: Section 12 & Tie-Breaker #1*
> "On our scorecard, we achieved **10/10 (5.8 s)** in Stage 1 and **[X] points** on the physical arm. As an honest **Prototype** assessment, our three main physical limits are: (1) monocular RGB depth sensitivity if the 0.52 m overhead camera mount shifts by more than 1 cm, (2) ~1.2° STS3215 servo gear play and the 60°C thermal cutoff under continuous retries, and (3) UVC webcam auto-exposure swings under direct hall spotlights."
