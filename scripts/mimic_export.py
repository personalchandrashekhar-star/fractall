"""Replay Isaac Lab HDF5 demonstrations (recorded or Mimic-generated) with the cameras on and write episode files
(.npz with both camera images, joint states, joint targets, task sentence), the same files the generator and the
teleop write. Only episodes that end with the cube on the tray are kept.

Run it from your clone (nwaves-mimic export is the short form of the same command):

    isaaclab -p scripts/mimic_export.py ~/workspace/mimic/generated.hdf5 mimic_v1
    isaaclab -p scripts/mimic_export.py generated.hdf5 mimic_v1 --config configs/my.yaml   # lights, camera jitter, colours and
                                                       # physics from your settings file. The cube and tray positions
                                                       # always come from the HDF5 file. Visual changes keep the replay
                                                       # faithful; physical ones may make some episodes fail: dropped.
"""
import argparse

import _start  # noqa: F401  (first: this clone's nwaves_hackathon goes to the front of Python's path)
import warp as wp

wp.config.enable_backward = False
from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("input", help="HDF5 file from nwaves-teleop --hdf5 or nwaves-mimic generate")
p.add_argument("name", help="dataset name under ~/workspace/datasets (or an absolute directory)")
p.add_argument("--config", default=None, help="settings file for lights, camera jitter, colours and physics "
                                             "(default: configs/datagen.yaml of this repository)")
p.add_argument("--limit", type=int, default=0, help="export at most this many episodes")
p.add_argument("--settle", type=int, default=20, help="extra hold steps after the last action before the success check")
_start.window_args(p)
add_launcher_args(p)
a = p.parse_args()
a.input = _start.checked("mimic-export", _start.need_demos, a.input, "export")
_start.window(a, "mimic-export")

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import nwaves_hackathon  # noqa: E402,F401
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402
from nwaves_hackathon import layout as L  # noqa: E402
from nwaves_hackathon import mdp as H  # noqa: E402
from nwaves_hackathon import randomization as R  # noqa: E402
from nwaves_hackathon.recorder import EpisodeRecorder  # noqa: E402

T = H._t
TASK = "NWaves-SO101-PickPlace-Kit-v0"


def log(msg):
    print(f"[mimic-export] {msg}", flush=True)


def main():
    conf = _start.load_settings("mimic-export", a.config)
    cfg, _ = resolve_task_config(TASK, "", overrides=[])
    R.apply(cfg, conf["randomization"])
    cfg.scene.num_envs = 1
    cfg.scene.view_cam = None          # the stream's third-person camera: not needed here
    out = a.name if os.path.isabs(a.name) else os.path.join(os.environ.get("WORKSPACE", os.path.expanduser("~/workspace")),
                                                             "datasets", a.name)
    with launch_simulation(cfg, a):
        from isaaclab.utils.datasets import HDF5DatasetFileHandler

        env = gym.make(TASK, cfg=cfg)
        u = env.unwrapped
        robot = u.scene["robot"]
        handler = HDF5DatasetFileHandler()
        handler.open(a.input)
        names = list(handler.get_episode_names())
        if a.limit:
            names = names[: a.limit]
        log(f"{len(names)} episodes in {a.input} -> {out}")
        rec = EpisodeRecorder(out)
        env.reset()
        kept = 0
        env_ids = torch.tensor([0], device=u.device)
        for i, name in enumerate(names):
            ep = handler.load_episode(name, u.device)
            actions = ep.data["actions"]
            u.reset_to(ep.data["initial_state"], env_ids, is_relative=True)
            obs = u.observation_manager.compute()
            rec.clear()
            t = 0.0
            seq = [actions[k:k + 1] for k in range(actions.shape[0])] + [actions[-1:]] * a.settle
            for act in seq:
                prev = obs
                obs, _, term, trunc, _ = env.step(act)
                rec.add(prev["policy"]["joint_pos"][0].tolist(), T(robot.data.joint_pos_target)[0].tolist(),
                        prev["visual"]["scene"][0].cpu().numpy().astype(np.uint8),
                        prev["visual"]["wrist"][0].cpu().numpy().astype(np.uint8), t)
                t += u.step_dt
            complete = bool(H.success_now(u)[0])
            if complete:
                origin = u.scene.env_origins[0]
                where = {k: [round(v, 4) for v in (T(u.scene[k].data.root_pos_w)[0] - origin)[:2].tolist()] for k in ("cube", "tray")}
                rec.save({"task": L.TASK, "instruction": L.INSTRUCTION, "success": True, "time_s": round(t, 3),
                          "source": f"hdf5:{os.path.basename(a.input)}:{name}", "end": where,
                          "config": os.path.basename(conf["_path"])})
                kept += 1
            else:
                rec.clear()
            log(f"{i + 1}/{len(names)} {name}: {'kept' if complete else 'dropped (the cube is not on the tray on replay)'}")
        handler.close()
        log(f"done: {kept} of {len(names)} episodes written to {out}")
        env.close()


if __name__ == "__main__":
    main()
