"""Check recorded episodes: play their actions back in the official task and see whether the cube ends on the tray.

    isaaclab -p scripts/replay.py first                 # the first 5 episodes of ~/workspace/datasets/first
    isaaclab -p scripts/replay.py first --episodes 20

Each episode's scene is rebuilt from what was stored with it (where the cube and the tray were), the arm starts at
home, and the recorded joint targets are sent one per tick, without looking at anything. If most episodes succeed
again, the dataset holds what a policy needs: the actions really do the task. Small differences between the run
that was recorded and the replay (friction, mass, motor gains are at their nominal values here) make a few fail.
"""
import argparse

import _start  # noqa: F401  (first: this clone's nwaves_hackathon goes to the front of Python's path)
import warp as wp

wp.config.enable_backward = False
from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

p = argparse.ArgumentParser(description="Replay recorded episodes in the official task.")
p.add_argument("dataset", help="dataset name under ~/workspace/datasets, or a folder with ep_*.npz")
p.add_argument("--episodes", type=int, default=5, help="how many episodes to replay")
_start.window_args(p)
add_launcher_args(p)
a = p.parse_args()
FOLDER = _start.checked("replay", _start.need_episodes, a.dataset)
_start.window(a, "replay")

import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import nwaves_hackathon  # noqa: E402,F401
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402
from nwaves_hackathon import layout as L  # noqa: E402
from nwaves_hackathon import mdp as H  # noqa: E402
from nwaves_hackathon import randomization as R  # noqa: E402

TASK = "NWaves-SO101-PickPlace-v0"


def main():
    folder = FOLDER                                  # a folder, or a dataset name: ~/workspace/datasets/NAME
    files = sorted(glob.glob(os.path.join(folder, "ep_*.npz")))[: a.episodes]
    cfg, _ = resolve_task_config(TASK, "", overrides=[])
    R.apply(cfg, L.FIXED_RANDOMIZATION)             # nominal light and physics; the layout comes from each episode
    cfg.scene.num_envs = 1
    cfg.scene.view_cam = None
    with launch_simulation(cfg, a):
        env = gym.make(TASK, cfg=cfg)
        u = env.unwrapped
        st = H.state(u)
        good = 0
        for k, path in enumerate(files):
            ep = np.load(path, allow_pickle=True)
            meta = json.loads(str(ep["meta"]))
            actions = ep["action"]
            if "layout" not in meta:
                print(f"[replay] {os.path.basename(path)}: no scene stored with this episode, skipped", flush=True)
                continue
            if "tray_xy" not in meta["layout"]:
                print(f"[replay] {os.path.basename(path)}: recorded before version 0.3.0, for another target, skipped. "
                      f"Episodes from before 0.3.0 do not fit the tray: generate new ones.", flush=True)
                continue
            st["force_layout"] = meta["layout"]
            env.reset()
            done = False
            for row in actions:
                act = torch.tensor([L.units_to_rad([float(v) for v in row])], dtype=torch.float32, device=u.device)
                _, _, term, trunc, _ = env.step(act)
                if bool(term.any()) or bool(trunc.any()):
                    done = True
                    break
            hold = torch.tensor([L.units_to_rad([float(v) for v in actions[-1]])], dtype=torch.float32, device=u.device)
            for _ in range(45):                     # the recording may stop before the scorer's 0.5 s are over
                if done:
                    break
                _, _, term, trunc, _ = env.step(hold)
                done = bool(term.any()) or bool(trunc.any())
            ok = bool(st["finished"][0][-1]["success"]) if done else bool(st["success"][0])
            good += int(ok)
            print(f"[replay] {k + 1}/{len(files)} {os.path.basename(path)}: "
                  f"{'cube on the tray again' if ok else 'NOT reproduced'}", flush=True)
        print(f"[replay] {good} of {len(files)} episodes reproduced by their recorded actions", flush=True)
        env.close()


if __name__ == "__main__":
    main()
