"""The scorer: runs a policy through N episodes of the pick-and-place task and counts the successes.

    nwaves-eval --policy ~/workspace/outputs/<run>          the OFFICIAL score: the copy of this script built into
                                                            the lab (practice: 10 episodes, public seed)
    isaaclab -p scripts/evaluate.py --policy ~/workspace/outputs/<run>
                                                            the same from your clone: runs YOUR copy of the task,
                                                            with your changes. Useful for experiments. Its number is
                                                            not the official score, and it says so.

It starts the policy server (your policy.py behind the socket protocol of nwaves_hackathon/protocol.py) and then
plays the episodes. The jury runs the built-in copy with 10 episodes and a seed you do not know.

Every episode: a new scene from the official randomisation (nwaves_hackathon/layout.py), the arm at home, 30 s.
The policy sees what it will see on the real arm: the scene and wrist images, the six joint positions (degrees,
gripper 0..100), the task name, the instruction and the time left. It answers with one or more joint targets in
the same units; each is applied for one 30 Hz tick. The episode is a success when the cube lies flat on the floor
of the tray's pocket, inside the walls, and the gripper has let go, for 0.5 s.

Result = number of successes. Ties are decided by the mean time of the successful episodes.
The scenes depend only on the seed and the episode number, so two policies scored with one seed see the same scenes.
"""
import argparse

import _start  # noqa: F401  (first: this clone's nwaves_hackathon goes to the front of Python's path)
import warp as wp

wp.config.enable_backward = False
from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

p = argparse.ArgumentParser(description="Official scorer of the SO-101 pick-and-place task.")
p.add_argument("--policy", default=None, help="policy folder (policy.py + model files, or a training run): starts the "
                                             "policy server for it. Without it a server must already listen on --port")
p.add_argument("--host", default="127.0.0.1", help="policy server address")
p.add_argument("--port", type=int, default=None, help="policy server port (default: a free one with --policy, else 7600)")
p.add_argument("--episodes", type=int, default=None, help="number of episodes (default: 10, the practice run)")
p.add_argument("--seed", type=int, default=None, help="seed of the scenes (default: the public practice seed)")
p.add_argument("--json", default=None, help="write the result here (default with --policy: "
                                           "~/workspace/results/eval-<date>/result.json)")
p.add_argument("--mp4", default=None, help="write a video of the run here, scene and wrist camera side by side "
                                          "(default with --policy: eval.mp4 next to the result)")
p.add_argument("--no-video", action="store_true", help="do not write the video")
p.add_argument("--max-chunk", type=int, default=30, help="use at most this many actions of a returned chunk")
p.add_argument("--first-episode", type=int, default=0,
               help="start at this episode number (0 = the first). Scenes depend only on the seed and the episode "
                    "number, so a run that was cut short can be continued where it stopped: with the same --json "
                    "file the episodes already played are kept and the new ones are added")
_start.window_args(p)
add_launcher_args(p)
a = p.parse_args()
if a.policy:
    a.policy = _start.checked("eval", _start.need_policy, a.policy)
_start.window(a, "eval")

import atexit  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import nwaves_hackathon  # noqa: E402,F401
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402
from nwaves_hackathon import layout as L  # noqa: E402
from nwaves_hackathon import mdp as H  # noqa: E402
from nwaves_hackathon.protocol import PolicyClient  # noqa: E402

TASK = "NWaves-SO101-PickPlace-Eval-v0"
ACCEPTED_MODELS = ("act", "diffusion", "smolvla")
MODEL_RULE = "Only ACT, Diffusion Policy and SmolVLA are accepted. A policy built on any other model is not scored."


OFFICIAL = _start.is_official_copy()
NOT_OFFICIAL = ("NOT THE OFFICIAL SCORE: this is the scorer of the code in " + _start.REPO +
                ". The official score comes from nwaves-eval.")


def log(msg):
    print(f"[eval] {msg}", flush=True)


def start_policy_server(folder, port, log_path):
    """Run nwaves_hackathon/serve.py of this same code with the LeRobot Python; stopped when this script ends."""
    python = os.environ.get("LEROBOT_PY", "/opt/lerobot-venv/bin/python")
    env = dict(os.environ, PYTHONPATH=_start.REPO)
    env.pop("NWAVES_POLICY_TEMPLATES", None)
    out = open(log_path, "a")

    def die_with_parent():                # the server must not outlive this script, even if the simulator crashes
        import ctypes
        import signal
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)          # PR_SET_PDEATHSIG

    proc = subprocess.Popen([python, "-m", "nwaves_hackathon.serve", "--policy", folder, "--port", str(port)],
                            cwd=_start.REPO, env=env, stdout=out, stderr=subprocess.STDOUT, preexec_fn=die_with_parent)
    atexit.register(lambda: proc.poll() is None and proc.terminate())
    return proc


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Video:
    """H.264 MP4 through ffmpeg (plays in a browser); falls back to OpenCV's mp4v if ffmpeg is missing."""

    def __init__(self, path, fps, width, height):
        import shutil
        import subprocess
        self.cv2 = __import__("cv2")
        self.proc, self.w = None, None
        if shutil.which("ffmpeg"):
            self.proc = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                 "-r", str(int(fps)), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", path], stdin=subprocess.PIPE)
        else:
            self.w = self.cv2.VideoWriter(path, self.cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    def add(self, scene, wrist, text):
        frame = np.ascontiguousarray(np.concatenate([scene, wrist], axis=1)).copy()
        self.cv2.putText(frame, text, (8, 22), self.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, self.cv2.LINE_AA)
        self.cv2.putText(frame, text, (8, 22), self.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (118, 185, 0), 1, self.cv2.LINE_AA)
        if self.proc is not None:
            try:
                self.proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                pass
        else:
            self.w.write(self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR))

    def close(self):
        if self.proc is not None:
            try:
                self.proc.stdin.close()
            finally:
                self.proc.wait(timeout=120)
        else:
            self.w.release()


def main():
    episodes = a.episodes if a.episodes is not None else L.PRACTICE_EPISODES
    seed = a.seed if a.seed is not None else L.PRACTICE_SEED
    port = a.port if a.port is not None else (free_port() if a.policy else 7600)
    server = None
    if not OFFICIAL:
        log(NOT_OFFICIAL)
    if a.policy:
        folder = a.policy                 # checked at the start: a folder with policy.py or a model (a run's last checkpoint)
        if a.json is None:
            workspace = os.environ.get("WORKSPACE") or os.path.expanduser("~/workspace")
            a.json = os.path.join(workspace, "results", "eval-" + time.strftime("%Y%m%d-%H%M%S"), "result.json")
        out_dir = os.path.dirname(os.path.abspath(a.json))
        os.makedirs(out_dir, exist_ok=True)
        if a.mp4 is None and not a.no_video:
            a.mp4 = os.path.join(out_dir, "eval.mp4")
        log(f"policy {folder}")
        log(f"results in {out_dir}")
        server = start_policy_server(folder, port, os.path.join(out_dir, "policy_server.log"))
    if a.no_video:
        a.mp4 = None
    cfg, _ = resolve_task_config(TASK, "", overrides=[])
    cfg.scene.num_envs = 1
    cfg.scene.view_cam = None
    cfg.seed = seed
    with launch_simulation(cfg, a):
        env = gym.make(TASK, cfg=cfg)
        u = env.unwrapped
        st = H.state(u)
        st["seed"] = int(seed)
        st["episode"][:] = int(a.first_episode)
        try:
            client = PolicyClient(a.host, port)
        except Exception as exc:
            if server is not None:
                log(f"the policy did not start ({exc}); see policy_server.log next to the result")
            raise
        model = client.info.get("model_type")
        log(f"policy: {client.info.get('name')}" + (f" (model type: {model})" if model else ""))
        if model and model not in ACCEPTED_MODELS:
            log(f"NOT SCORED: this policy is built on '{model}'. {MODEL_RULE}")
            client.close()
            env.close()
            raise SystemExit(3)
        log(f"{episodes} episodes, seed {seed}" + (" (the public practice seed)" if seed == L.PRACTICE_SEED else ""))
        obs, _ = env.reset()
        video = None
        if a.mp4:
            os.makedirs(os.path.dirname(os.path.abspath(a.mp4)), exist_ok=True)
            if a.first_episode > 0 and os.path.isfile(a.mp4):       # a continued run: do not overwrite the first part
                root, ext = os.path.splitext(a.mp4)
                a.mp4 = f"{root}_from_episode_{a.first_episode + 1}{ext}"
            video = Video(a.mp4, L.CONTROL_HZ, 2 * L.IMAGE_W, L.IMAGE_H)
        results, late, successes = [], 0, 0
        t_wall = time.time()
        first = int(a.first_episode)
        if first > 0 and a.json and os.path.isfile(a.json):
            # continue a run that was cut short: keep what it had played, if it is the same run
            try:
                with open(a.json) as f:
                    old = json.load(f)
                same = (old.get("seed") == seed and old.get("episodes") == episodes and not old.get("complete")
                        and old.get("first_episode", 0) + old.get("episodes_played", 0) == first
                        and old.get("policy") == client.info.get("name"))
                if same:
                    results = list(old.get("results", []))
                    successes = sum(int(r["success"]) for r in results)
                    first = int(old.get("first_episode", 0))
                    log(f"continuing the run in {a.json}: {len(results)} episodes already played, {successes} succeeded")
            except (OSError, ValueError, KeyError):
                pass

        def summary(complete):
            times = [r["time_s"] for r in results if r["success"]]
            lat = client.latencies_ms or [0.0]
            return {
                "task": L.TASK, "seed": seed, "episodes": episodes, "first_episode": first,
                "episodes_played": len(results), "complete": complete, "successes": successes,
                "success_rate": round(successes / max(len(results), 1), 4),
                "mean_time_s": round(float(np.mean(times)), 2) if times else None,
                "policy": client.info.get("name"), "model_type": model,
                "policy_ms": {"median": round(float(np.median(lat)), 1),
                              "p90": round(float(np.percentile(lat, 90)), 1), "late_calls": late},
                "wall_s": round(time.time() - t_wall, 1), "version": nwaves_hackathon.__version__,
                "official": OFFICIAL, "results": results,
            }

        def write(complete):
            if a.json:
                os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
                with open(a.json + ".tmp", "w") as f:
                    json.dump(summary(complete), f, indent=2)
                os.replace(a.json + ".tmp", a.json)

        for k in range(a.first_episode, episodes):
            client.reset(L.TASK, L.INSTRUCTION)
            queue, t, done = [], 0.0, False
            while not done:
                scene = obs["visual"]["scene"][0].cpu().numpy().astype(np.uint8)
                wrist = obs["visual"]["wrist"][0].cpu().numpy().astype(np.uint8)
                if not queue:
                    jp = obs["policy"]["joint_pos"][0].tolist()
                    actions = client.act(L.rad_to_units(jp), {"scene": scene, "wrist": wrist}, L.TASK, L.INSTRUCTION,
                                         L.EPISODE_SECONDS - t)
                    if client.latencies_ms[-1] > 1000.0 / L.CONTROL_HZ:
                        late += 1
                    queue = [row for row in actions[: a.max_chunk]]
                units = queue.pop(0)
                act = torch.tensor([L.units_to_rad(units)], dtype=torch.float32, device=u.device)
                if video is not None:
                    on_tray = bool(H.cube_on_tray(u)[0])
                    video.add(scene, wrist, f"episode {k + 1}/{episodes}   {t:4.1f} s   cube on tray: "
                                            f"{'yes' if on_tray else 'no'}   successes so far: {successes}")
                obs, _, term, trunc, _ = env.step(act)
                t += u.step_dt
                done = bool(term.any()) or bool(trunc.any())
            res = st["finished"][0][-1]
            results.append(res)
            successes += int(res["success"])
            log(f"episode {k + 1}/{episodes}: " + (f"SUCCESS at {res['time_s']:.1f} s" if res["success"] else "no success")
                + f"   (cube at {res['layout']['cube_xy']}, tray at {res['layout']['tray_xy']})")
            write(complete=False)                    # the result so far survives a simulator crash
        if video is not None:
            video.close()
        final = summary(complete=True)
        mean = f"{final['mean_time_s']:.1f} s" if final["mean_time_s"] is not None else "-"
        played = len(results)
        log(f"RESULT: {successes} of {played} episodes succeeded, mean time of the successful ones {mean}"
            + (f"   (episodes {first + 1} to {episodes} of {episodes})" if first else ""))
        if not OFFICIAL:
            log(NOT_OFFICIAL)
        write(complete=True)
        print("[eval] JSON " + json.dumps({k: v for k, v in final.items() if k != "results"}), flush=True)
        if a.json:
            log(f"result: {a.json}")
        if a.mp4:
            log(f"video:  {a.mp4}")
        client.close()
        if server is not None:
            server.terminate()
        env.close()


if __name__ == "__main__":
    main()
