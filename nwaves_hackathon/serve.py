"""nwaves-serve: run a team's policy.py behind the hackathon policy protocol (nwaves_hackathon/protocol.py).

The same server is used by the simulation scorer and by the real-arm runner, so a submission
runs the identical code path in both. Every client connection gets its own Policy instance.

    nwaves-serve --policy ~/workspace/my_submission            # a folder with policy.py
    nwaves-serve --policy ~/workspace/outputs/<run>/checkpoints/last/pretrained_model
                                                               # a LeRobot checkpoint without policy.py: the
                                                               # template of its model (policy/<model>/policy.py)
    nwaves-serve --policy ... --port 7600 --host 0.0.0.0       # reachable from another container

policy.py must define:

    class Policy:
        def __init__(self, model_dir: str): ...                 # model_dir = the folder policy.py is in
        def reset(self, task: str, instruction: str): ...       # optional; called at the start of each task
        def act(self, obs: dict) -> np.ndarray: ...             # (6,) or (k, 6) joint targets, LeRobot units

obs = {"scene": uint8 (480, 640, 3) RGB, "wrist": uint8 (480, 640, 3) RGB,
       "joint_state": float32 (6,) [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
                      in degrees (LeRobot use_degrees=True; gripper 0..100),
       "task": "pickplace", "instruction": "Put the cube on the tray.", "time_left": float seconds}
"""
import argparse
import importlib.util
import json
import os
import socket
import sys
import threading
import time
import traceback

import numpy as np

from nwaves_hackathon.protocol import decode_jpeg, recv_exact, recv_msg, send_msg  # noqa: E402


def log(msg):
    print(f"[nwaves-serve] {msg}", flush=True)


def resolve(policy):
    """A local folder, or user/repo[@revision] downloaded from the Hugging Face Hub."""
    path = os.path.expanduser(policy)
    if os.path.isdir(path):
        return os.path.abspath(path)
    repo, _, rev = policy.partition("@")
    if repo.startswith("hf:"):
        repo = repo[3:]
    from huggingface_hub import snapshot_download
    local = snapshot_download(repo_id=repo, revision=rev or None, repo_type="model")
    log(f"downloaded {repo}@{rev or 'main'} to {local}")
    return local


TEMPLATES = os.environ.get("NWAVES_POLICY_TEMPLATES") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "policy")


def model_type(model_dir):
    """The LeRobot model type of a checkpoint folder ("act", "diffusion", "smolvla", ...), or None."""
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            return json.load(f).get("type")
    except (OSError, ValueError):
        return None


def load_policy_class(model_dir):
    path = os.path.join(model_dir, "policy.py")
    if not os.path.isfile(path):
        kind = model_type(model_dir)
        template = os.path.join(TEMPLATES, str(kind), "policy.py")
        if kind and os.path.isfile(template):
            log(f"no policy.py in {model_dir}: using the {kind} template {template}")
            path = template
        else:
            raise FileNotFoundError(f"no policy.py in {model_dir}")
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    spec = importlib.util.spec_from_file_location("team_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Policy"):
        raise AttributeError("policy.py has no class Policy")
    return module.Policy


def policy_name(model_dir):
    """<run> for <run>/checkpoints/<step>/pretrained_model, else the folder's own name."""
    parts = os.path.abspath(model_dir).rstrip("/").split(os.sep)
    if len(parts) >= 4 and parts[-1] == "pretrained_model" and parts[-3] == "checkpoints":
        return parts[-4]
    return parts[-1]


def serve_client(conn, addr, policy_cls, model_dir, lock):
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    policy = None
    try:
        while True:
            msg = recv_msg(conn)
            kind = msg.get("type")
            if kind == "hello":
                try:
                    with lock:                       # model loading is not always thread safe
                        policy = policy_cls(model_dir)
                    send_msg(conn, {"ok": True, "name": policy_name(model_dir), "model_type": model_type(model_dir)})
                except Exception as exc:
                    traceback.print_exc()
                    send_msg(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                    return
            elif kind == "reset":
                try:
                    if hasattr(policy, "reset"):
                        policy.reset(msg.get("task", ""), msg.get("instruction", ""))
                    send_msg(conn, {"ok": True})
                except Exception as exc:
                    traceback.print_exc()
                    send_msg(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            elif kind == "observation":
                obs = {name: decode_jpeg(recv_exact(conn, int(msg["sizes"][name]))) for name in msg["images"]}
                obs["joint_state"] = np.asarray(msg["joint_state"], dtype=np.float32)
                for key in ("task", "instruction", "time_left"):
                    obs[key] = msg.get(key)
                t0 = time.perf_counter()
                try:
                    actions = np.asarray(policy.act(obs), dtype=np.float64)
                    send_msg(conn, {"actions": actions.tolist(), "ms": (time.perf_counter() - t0) * 1000.0})
                except Exception as exc:
                    traceback.print_exc()
                    send_msg(conn, {"error": f"{type(exc).__name__}: {exc}"})
            elif kind == "bye":
                return
            else:
                send_msg(conn, {"error": f"unknown message {kind!r}"})
    except (ConnectionError, OSError):
        pass
    finally:
        conn.close()
        log(f"client {addr} closed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", required=True, help="folder with policy.py (or a LeRobot checkpoint folder), "
                                                    "or user/repo[@revision] on the Hugging Face Hub")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7600)
    args = ap.parse_args()
    # the files ACT and SmolVLA fetch when they load come from the lab's own copies; without the internet the
    # Hugging Face libraries are told not to wait for it (see models.py). Before policy.py imports anything.
    from nwaves_hackathon import models
    linked = models.link_lab_models()
    if linked:
        log(f"model files of this lab linked into the caches: {', '.join(linked)}")
    if models.go_offline_if_needed():
        log("no connection to the Hugging Face Hub: loading from the files on disk")
    model_dir = resolve(args.policy)
    policy_cls = load_policy_class(model_dir)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(8)
    log(f"serving {model_dir} on {args.host}:{args.port}")
    lock = threading.Lock()
    while True:
        conn, addr = srv.accept()
        log(f"client {addr}")
        threading.Thread(target=serve_client, args=(conn, addr, policy_cls, model_dir, lock), daemon=True).start()


if __name__ == "__main__":
    main()
