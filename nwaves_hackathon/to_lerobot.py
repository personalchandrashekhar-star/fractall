"""Convert recorded episodes (.npz from the generator, the teleop or the Mimic export) into a LeRobot v3 dataset.

    python -m nwaves_hackathon.to_lerobot EPISODE_DIR OUT_DIR --repo-id user/so101_pickplace

Runs in the LeRobot venv. State/action names follow lerobot-record's SO-101 follower convention
(``shoulder_pan.pos`` ... ``gripper.pos``) and the cameras are ``observation.images.scene`` and
``observation.images.wrist``, so a policy trained here drops straight onto the real arm.
"""
import argparse
import glob
import io
import json
import os
import shutil
import time

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def _decode(blob):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(bytes(blob))).convert("RGB"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("episodes", help="directory with ep_*.npz")
    ap.add_argument("out", help="dataset root to create (removed first if --force)")
    ap.add_argument("--repo-id", required=True, help="Hugging Face id of the dataset, e.g. user/so101_pickplace")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    files = sorted(glob.glob(os.path.join(a.episodes, "ep_*.npz")))
    if a.limit:
        files = files[: a.limit]
    if not files:
        name = os.path.basename(os.path.abspath(a.episodes))
        raise SystemExit(f"[to_lerobot] There are no episodes (ep_*.npz) in {os.path.abspath(a.episodes)}. Make some "
                         f"first:  isaaclab -p scripts/generate.py --episodes 50 --name {name}")
    if os.path.exists(a.out):
        if not a.force:
            raise SystemExit(f"[to_lerobot] {a.out} exists already. Add --force to replace it, or give the dataset "
                             f"another name.")
        shutil.rmtree(a.out)
    first = np.load(files[0], allow_pickle=True)
    meta = json.loads(str(first["meta"]))
    h, w = meta["image_hw"]
    names = [f"{j}.pos" for j in JOINTS]
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": names},
        "action": {"dtype": "float32", "shape": (6,), "names": names},
        "observation.images.scene": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]},
        "observation.images.wrist": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]},
    }
    kwargs = dict(repo_id=a.repo_id, fps=int(round(meta["fps"])), root=a.out, robot_type="so101_follower",
                  features=features, use_videos=True)
    try:        # frames go straight into the video encoder (several times faster than writing PNG files first)
        ds = LeRobotDataset.create(streaming_encoding=True, encoder_queue_maxsize=120, **kwargs)
    except TypeError:                                   # older LeRobot: PNG files, then one video per episode
        ds = LeRobotDataset.create(image_writer_threads=4, **kwargs)
    frames_total, t0 = 0, time.time()
    for i, path in enumerate(files):
        ep = np.load(path, allow_pickle=True)
        m = json.loads(str(ep["meta"]))
        task = m.get("instruction") or m.get("task", "")
        # read every array once: indexing the .npz again and again would unpack the whole file each time
        state, action, scene, wrist = ep["state"], ep["action"], ep["scene"], ep["wrist"]
        for t in range(len(state)):
            ds.add_frame({
                "observation.state": state[t].astype(np.float32),
                "action": action[t].astype(np.float32),
                "observation.images.scene": _decode(scene[t]),
                "observation.images.wrist": _decode(wrist[t]),
                "task": task,
            })
        ds.save_episode()
        frames_total += len(state)
        print(f"[to_lerobot] episode {i + 1}/{len(files)}: {len(state)} frames, task '{task}'", flush=True)
    if hasattr(ds, "finalize"):
        ds.finalize()
    print(f"[to_lerobot] done: {len(files)} episodes, {frames_total} frames -> {a.out} "
          f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
