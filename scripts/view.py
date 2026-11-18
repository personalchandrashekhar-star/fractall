"""Look at the scene: the official task with a new random layout every few seconds. The arm stands still at home.

Run it from your clone (nwaves-view is the short form of the same command):

    isaaclab -p scripts/view.py                              the official randomisation (what the scorer draws)
    isaaclab -p scripts/view.py --config                     the ranges of configs/datagen.yaml (your generator's)
    isaaclab -p scripts/view.py --config configs/my.yaml     the ranges of your own settings file
    isaaclab -p scripts/view.py --seconds 3                  a new layout every 3 s

Use it to see what your randomisation settings really do before you generate a dataset with them.
--save DIR writes the scene, wrist and third-person pictures of every layout into DIR (works without a window).
"""
import argparse

import _start  # noqa: F401  (first: this clone's nwaves_hackathon goes to the front of Python's path)
import warp as wp

wp.config.enable_backward = False
from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

p = argparse.ArgumentParser(description="Show the scene with a new random layout every few seconds.")
p.add_argument("--seconds", type=float, default=5.0, help="time per layout")
p.add_argument("--config", nargs="?", const="", default=None,
               help="draw from a generator settings file instead of the official ranges (no value: configs/datagen.yaml)")
p.add_argument("--seed", type=int, default=None)
p.add_argument("--layouts", type=int, default=0, help="stop after this many layouts (default: run until Ctrl-C)")
p.add_argument("--save", default=None, help="folder for pictures of every layout")
_start.window_args(p)
add_launcher_args(p)
a = p.parse_args()
_start.window(a, "view")

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

import nwaves_hackathon  # noqa: E402,F401
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402
from nwaves_hackathon import mdp as H  # noqa: E402
from nwaves_hackathon import randomization as R  # noqa: E402

TASK = "NWaves-SO101-PickPlace-v0"          # joint-position control, the official scene


def main():
    cfg, _ = resolve_task_config(TASK, "", overrides=[])
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 1e6               # we reset ourselves
    if a.config is not None:
        conf = _start.load_settings("view", a.config or None)
        R.apply(cfg, conf["randomization"])
        print(f"[view] ranges of {conf['_path']}", flush=True)
    else:
        print("[view] the official ranges (nwaves_hackathon/layout.py)", flush=True)
    if a.seed is not None:
        cfg.seed = a.seed
    if a.save is None:
        cfg.scene.view_cam = None
    with launch_simulation(cfg, a):
        env = gym.make(TASK, cfg=cfg)
        u = env.unwrapped
        st = H.state(u)
        obs, _ = env.reset()
        hold = H._t(u.scene["robot"].data.joint_pos).clone()
        steps = max(int(a.seconds / u.step_dt), 1)
        k = 0
        print("[view] scene ready (Ctrl-C to stop)", flush=True)
        try:
            while True:
                lay = st["layout"][0]
                built = lambda rgb: "as built" if rgb is None else str(rgb)       # noqa: E731
                print(f"[view] layout {k + 1}: cube at {lay['cube_xy']} turned {lay['cube_yaw_deg']} deg, colour "
                      f"{lay['cube_rgb']}, tray at {lay['tray_xy']} turned {lay['tray_yaw_deg']} deg, surface "
                      f"{lay['surface']}, walls {built(lay['wall_rgb'])}, arm {built(lay['arm_rgb'])}, lights "
                      f"{lay['lights']}", flush=True)
                for _ in range(steps):
                    obs, *_ = env.step(hold)
                if a.save:
                    from PIL import Image
                    os.makedirs(a.save, exist_ok=True)
                    for cam in ("scene", "wrist"):
                        Image.fromarray(obs["visual"][cam][0].cpu().numpy().astype(np.uint8)).save(
                            os.path.join(a.save, f"layout_{k + 1:03d}_{cam}.png"))
                    view = H.view_image(u)
                    if view is not None:
                        Image.fromarray(view).save(os.path.join(a.save, f"layout_{k + 1:03d}_view.png"))
                k += 1
                if a.layouts and k >= a.layouts:
                    break
                obs, _ = env.reset()
                hold = H._t(u.scene["robot"].data.joint_pos).clone()
        except KeyboardInterrupt:
            pass
        env.close()


if __name__ == "__main__":
    main()
