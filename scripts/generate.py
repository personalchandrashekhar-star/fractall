"""Automatic dataset generator: a scripted pick and place that knows where the cube and the tray are.

It builds scenes from the ranges in configs/datagen.yaml (section ``randomization``), moves the arm the way the
section ``behavior`` says, and keeps only the episodes that end with the cube on the tray. The episodes are the
same files the keyboard teleop writes; nwaves-dataset turns a folder of them into a LeRobot dataset.

Run it from your clone (nwaves-generate is the short form of the same command):

    isaaclab -p scripts/generate.py --config configs/datagen.yaml --episodes 50 --name first
                                                     50 episodes with the shipped settings -> ~/workspace/datasets/first
    isaaclab -p scripts/generate.py --config configs/my.yaml --episodes 200 --name wide_v1
                                                     your own copy of the settings, your name for the dataset
    isaaclab -p scripts/generate.py --episodes 10 --dry-run          watch it work, save nothing
    isaaclab -p scripts/generate.py --episodes 200 --name big --headless     no Isaac Sim window: faster
    isaaclab -p scripts/generate.py --episodes 10 --hdf5 ~/workspace/mimic/src.hdf5
                                                     also keep them as Isaac Lab demonstrations (for Mimic)

The shipped settings are narrow on purpose. Widening them (and proving that the generator still succeeds) is
your job: the official scorer draws from wider ranges, see nwaves_hackathon/layout.py.

How the arm is driven: the script commands a fingertip pose (straight down, jaw turned to a cube face) and Isaac
Lab's IK turns it into joint targets. What is recorded is what a policy needs: joint positions, joint targets,
both camera images. The steps of one episode: open, hover over the cube, go down, close, lift to carry height,
carry over the tray's wall, lower the cube into the pocket, let go, back up, home. The jaws hold the cube by its
upper half, so they stay above the tray's walls when the cube is set down.
"""
import argparse

import _start  # noqa: F401  (first: this clone's nwaves_hackathon goes to the front of Python's path)
import warp as wp

wp.config.enable_backward = False
from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

p = argparse.ArgumentParser(description="Automatic dataset generator (scripted pick and place).")
p.add_argument("--episodes", type=int, default=None, help="successful episodes to keep (default: the config's 'episodes')")
p.add_argument("--name", default="generated", help="dataset folder under ~/workspace/datasets (or an absolute path)")
p.add_argument("--config", default=None, help="settings file (default: configs/datagen.yaml of this repository)")
p.add_argument("--seed", type=int, default=None, help="seed of scenes and behaviour (default: the config's 'seed')")
p.add_argument("--envs", type=int, default=None, help="scenes simulated side by side (default: the config's 'envs')")
p.add_argument("--dry-run", action="store_true", help="run and report the success rate, save nothing")
p.add_argument("--hdf5", default=None, help="also keep the successful episodes as Isaac Lab demonstrations in this "
                                           "file (the input of nwaves-mimic); runs one scene at a time")
p.add_argument("--debug", action="store_true", help="print what every episode did")
_start.window_args(p)
add_launcher_args(p)
a = p.parse_args()
_start.window(a, "generate")

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as mu  # noqa: E402
import nwaves_hackathon  # noqa: E402,F401
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402
from nwaves_hackathon import layout as L  # noqa: E402
from nwaves_hackathon import mdp as H  # noqa: E402
from nwaves_hackathon import randomization as R  # noqa: E402
from nwaves_hackathon.recorder import EpisodeRecorder, enable_hdf5_recorder, export_success  # noqa: E402

TASK = "NWaves-SO101-PickPlace-Kit-v0"
HALF = L.CUBE_SIZE / 2
T = H._t

# Every behaviour setting and its value when the config file does not give one (see configs/datagen.yaml).
# A length or a time may also be given as [low, high]: it is then drawn anew for every episode.
BEHAVIOR = {
    "speed": 1.0,                  # 1 = normal pace; 0.5 = every move takes twice as long
    "speed_jitter": 0.0,           # +- fraction of the speed, drawn once per episode
    "approach_height": 0.07,       # m above the cube where the gripper stops before going down
    "grasp_depth": 0.008,          # m: fingertip height above the cube centre at the grasp
    "grasp_along": -0.012,         # m: fingertip shift along the jaw's closing direction (calibrated, centres the cube)
    "grasp_offset_noise": 0.0,     # m: random error of the grasp point, +- in both horizontal directions
    "carry_clearance": 0.02,       # m between the cube's underside and the top of the tray's walls while carrying
    "release_height": 0.003,       # m between the cube's underside and the pocket's floor when the gripper lets go
    "place_offset_noise": 0.0,     # m: random offset of the place point from the tray's centre
    "pause": [0.0, 0.0],           # s: a random wait before each step of the episode
    "action_noise": 0.0,           # m: random wobble of the commanded fingertip position
    "retries": 1,                  # tries again this many times when the cube was not lifted
    "return_home": True,           # glide back to the home pose after letting go
    "end_hold": 0.3,               # s of standing still recorded at the end
}


def log(msg):
    print(f"[generate] {msg}", flush=True)


def behavior_settings(conf):
    given = dict(conf.get("behavior") or {})
    unknown = sorted(set(given) - set(BEHAVIOR))
    if unknown:
        raise SystemExit(f"[generate] {conf['_path']}: unknown behaviour setting(s) {unknown}; known: {sorted(BEHAVIOR)}")
    b = dict(BEHAVIOR)
    b.update(given)
    for key in RANGED:                        # a number, or [low, high] drawn per episode
        value = b[key]
        pair = [value, value] if isinstance(value, (int, float)) and not isinstance(value, bool) else value
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2 and all(isinstance(v, (int, float)) for v in pair)
                and pair[0] <= pair[1]):
            raise SystemExit(f"[generate] behavior.{key} must be a number or [low, high], got {value!r}")
        b[key] = [float(pair[0]), float(pair[1])]
    if b["speed"][0] <= 0:
        raise SystemExit("[generate] behavior.speed must be above 0")
    if b["carry_clearance"][0] < MIN_CARRY:
        raise SystemExit(f"[generate] behavior.carry_clearance must be at least {MIN_CARRY} m (the cube is carried over "
                         f"the tray's walls)")
    if b["release_height"][0] < 0:
        raise SystemExit("[generate] behavior.release_height must be 0 or more (below 0 the cube is pressed into the tray)")
    if b["release_height"][1] > L.TRAY_WALL_HEIGHT:
        raise SystemExit(f"[generate] behavior.release_height must be below {L.TRAY_WALL_HEIGHT:.3f} m, the height of "
                         f"the tray's walls: the cube is let go inside the pocket")
    return b


MIN_CARRY = 0.005         # m: smallest carry_clearance
LIFTED = 0.012            # m: the cube counts as lifted when it is this far above where it stood

# settings that take a number or a [low, high] pair
RANGED = ("speed", "approach_height", "grasp_depth", "grasp_along", "grasp_offset_noise", "carry_clearance",
          "release_height", "place_offset_noise", "pause", "action_noise", "end_hold")


def draw_behavior(behavior, rng):
    """The behaviour of one episode: every [low, high] setting drawn once (the pause stays a range: it is drawn
    again before each step of the episode)."""
    b = dict(behavior)
    for key in RANGED:
        if key != "pause":
            lo, hi = behavior[key]
            b[key] = lo if hi <= lo else float(rng.uniform(lo, hi))
    return b


class Scene:
    """Handles shared by all arms: the env, the robot, and the command buffers that become one action."""

    def __init__(self, env):
        self.env, self.u = env, env.unwrapped
        self.n = self.u.num_envs
        self.robot = self.u.scene["robot"]
        self.dev = self.u.device
        self.dt = self.u.step_dt
        self.body = self.robot.body_names.index("gripper")
        self.tcp = torch.tensor([L.TCP_OFFSET], device=self.dev)
        self.origins = self.u.scene.env_origins
        self.arm_term = self.u.action_manager.get_term("arm_action")
        self.arm_ids = [self.robot.joint_names.index(j) for j in L.HOME_RAD]
        self.home_q = torch.tensor([L.HOME_RAD[j] for j in L.HOME_RAD], device=self.dev)
        self.pos = torch.zeros(self.n, 3, device=self.dev)          # commanded fingertip position, env frame
        self.quat = torch.zeros(self.n, 4, device=self.dev)         # commanded gripper orientation (xyzw)
        self.quat[:, 3] = 1.0
        self.grip = torch.full((self.n, 1), L.JAW_OPEN_RAD, device=self.dev)
        self.override = torch.zeros(self.n, 5, device=self.dev)     # joint targets while an arm glides home
        self.override_on = torch.zeros(self.n, dtype=torch.bool, device=self.dev)

    def tcp_pose(self, i):
        pos = T(self.robot.data.body_pos_w)[i, self.body]
        quat = T(self.robot.data.body_quat_w)[i, self.body]
        return pos + mu.quat_apply(quat.unsqueeze(0), self.tcp)[0] - self.origins[i], quat

    def obj(self, name, i):
        d = self.u.scene[name].data
        return T(d.root_pos_w)[i] - self.origins[i], T(d.root_quat_w)[i]

    def action(self):
        root_p, root_q = T(self.robot.data.root_pos_w), T(self.robot.data.root_quat_w)
        pos_b, quat_b = mu.subtract_frame_transforms(root_p, root_q, self.pos + self.origins, self.quat)
        if bool(self.override_on.any()):
            self.arm_term.joint_override, self.arm_term.joint_override_mask = self.override, self.override_on
        else:
            self.arm_term.joint_override, self.arm_term.joint_override_mask = None, None
        return torch.cat([pos_b, quat_b, self.grip], dim=-1)


def slerp(q0, q1, s):
    d = float((q0 * q1).sum())
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + (q1 - q0) * s
        return q / q.norm()
    th = math.acos(min(d, 1.0))
    return (math.sin((1.0 - s) * th) * q0 + math.sin(s * th) * q1) / math.sin(th)


class Arm:
    """The scripted pick and place of one environment. ``run()`` is a coroutine: every ``yield`` is one control
    tick (1/30 s) in which the arm follows the command left in the scene's buffers."""

    def __init__(self, scene, i, behavior, rng):
        self.s, self.i, self.rng = scene, i, rng
        self.b = behavior = draw_behavior(behavior, rng)
        self.dev, self.dt = scene.dev, scene.dt
        self.speed = behavior["speed"] * (1.0 + behavior["speed_jitter"] * float(rng.uniform(-1.0, 1.0)))
        self.noise = torch.zeros(3, device=self.dev)
        self.notes = []
        self.released = False
        self.hang = HALF + self.b["grasp_depth"]      # fingertip height above the cube's underside while it is held

    # -- helpers
    def v(self, x, y, z):
        return torch.tensor([float(x), float(y), float(z)], device=self.dev)

    def down_quat(self, yaw):
        z = torch.zeros(1, device=self.dev)
        return mu.quat_from_euler_xyz(z, z, torch.full((1,), float(yaw), device=self.dev))[0]

    def command(self, pos, quat):
        """Leave a fingertip pose command for this tick."""
        sigma = float(self.b["action_noise"])
        if sigma > 0.0:                         # smooth wobble: a random walk pulled back to zero
            step = torch.tensor(self.rng.normal(0.0, sigma, 3), device=self.dev, dtype=torch.float32)
            self.noise = 0.8 * self.noise + 0.6 * step
            pos = pos + self.noise
        self.s.pos[self.i], self.s.quat[self.i] = pos, quat
        self.target = (pos, quat)

    def move(self, pos, quat, seconds, settle=0.2, arrive=0.006, patience=2.0):
        """Glide the fingertip to a pose (smooth start and stop), wait until it is within ``arrive`` metres (the
        joint targets are speed limited like on the real arm, so a long move can take longer than planned; gives
        up after ``patience`` seconds), then hold it there for ``settle`` seconds."""
        start, q_start = self.s.tcp_pose(self.i)
        start, q_start = start.clone(), q_start.clone()
        steps = max(int(round(seconds / self.speed / self.dt)), 1)
        for k in range(1, steps + 1):
            f = k / steps
            f = f * f * (3 - 2 * f)
            self.command(start + f * (pos - start), slerp(q_start, quat, f))
            yield
        for _ in range(int(round(patience / self.dt))):
            if float((self.s.tcp_pose(self.i)[0] - pos).norm()) < arrive:
                break
            self.command(pos, quat)
            yield
        for _ in range(int(round(settle / self.dt))):
            self.command(pos, quat)
            yield
        return float((self.s.tcp_pose(self.i)[0] - pos).norm())

    def jaw(self, value, seconds):
        start = float(self.s.grip[self.i, 0])
        steps = max(int(round(seconds / self.dt)), 1)
        for k in range(1, steps + 1):
            self.s.grip[self.i, 0] = start + (value - start) * k / steps
            yield

    def hold(self, seconds):
        for _ in range(int(round(seconds / self.dt))):
            yield

    def pause(self):
        lo, hi = self.b["pause"]
        if hi > 0.0:
            yield from self.hold(float(self.rng.uniform(lo, hi)))

    def floor_z(self):
        """Height of the floor of the tray's pocket, where the cube is put."""
        return float(self.s.obj("tray", self.i)[0][2]) + L.TRAY_FLOOR

    def wall_z(self):
        """Height of the top of the tray's walls, which the cube is carried over."""
        return float(self.s.obj("tray", self.i)[0][2]) + L.TRAY_HEIGHT

    def face_yaw(self, cube_quat, near):
        """The jaw closes on two opposite cube faces: of the four face directions, the one nearest to ``near``."""
        cy = float(mu.euler_xyz_from_quat(cube_quat.unsqueeze(0))[2][0])
        cands = [cy + k * math.pi / 2 for k in range(-4, 5)]
        return min(cands, key=lambda y: abs(math.atan2(math.sin(y - near), math.cos(y - near))))

    # -- the steps of an episode
    def pick(self):
        b = self.b
        pos, quat = self.s.obj("cube", self.i)
        # From the SO-101 home the gripper points forward; tilting it down puts the jaw's closing direction along
        # the arm. So the natural grasp direction is base -> cube, and the cube face nearest to it is taken.
        near = math.atan2(float(pos[1]), float(pos[0]))
        yaw = self.face_yaw(quat, near)
        down = self.down_quat(yaw)
        n = float(b["grasp_offset_noise"])
        along = b["grasp_along"] + (float(self.rng.uniform(-n, n)) if n > 0 else 0.0)
        across = float(self.rng.uniform(-n, n)) if n > 0 else 0.0
        grasp = pos + self.v(along * math.cos(yaw) - across * math.sin(yaw), along * math.sin(yaw) + across * math.cos(yaw), 0.0)
        yield from self.jaw(L.JAW_OPEN_RAD, 0.3)
        yield from self.pause()
        e1 = yield from self.move(grasp + self.v(0, 0, b["approach_height"]), down, 1.2, settle=0.3, arrive=0.012)
        yield from self.pause()
        e2 = yield from self.move(grasp + self.v(0, 0, b["grasp_depth"]), down, 0.9, settle=0.5, arrive=0.004)
        yield from self.pause()
        # where the fingertip really is against the cube's underside: the place motion puts the cube down with it
        self.hang = float(self.s.tcp_pose(self.i)[0][2]) - (float(self.s.obj("cube", self.i)[0][2]) - HALF)
        yield from self.jaw(L.JAW_CLOSED_RAD, 0.5)
        yield from self.hold(0.15)
        carry_z = self.wall_z() + b["carry_clearance"] + self.hang
        yield from self.move(self.v(grasp[0], grasp[1], carry_z), down, 0.9, settle=0.1, arrive=0.02)
        now = self.s.obj("cube", self.i)[0]
        lifted = float(now[2]) > float(pos[2]) + LIFTED
        self.notes.append(f"pick: hover error {e1 * 1000:.1f} mm, grasp error {e2 * 1000:.1f} mm, lifted {lifted}")
        return lifted, yaw, along, across

    def place(self, yaw, along, across):
        """Carry the cube over the tray's wall, lower it into the pocket and let go."""
        b = self.b
        tray = self.s.obj("tray", self.i)[0]
        quat = self.down_quat(yaw)
        n = float(b["place_offset_noise"])
        dx, dy = (float(self.rng.uniform(-n, n)), float(self.rng.uniform(-n, n))) if n > 0 else (0.0, 0.0)
        # the cube sits -along, -across from the fingertip: aim the fingertip so the cube ends over the tray's centre
        x = float(tray[0]) + dx + along * math.cos(yaw) - across * math.sin(yaw)
        y = float(tray[1]) + dy + along * math.sin(yaw) + across * math.cos(yaw)
        carry_z = self.wall_z() + b["carry_clearance"] + self.hang
        release_z = self.floor_z() + b["release_height"] + self.hang
        here = self.s.tcp_pose(self.i)[0]
        dist = math.hypot(x - float(here[0]), y - float(here[1]))
        yield from self.pause()
        yield from self.move(self.v(x, y, carry_z), quat, max(0.6, dist / 0.14), settle=0.2, arrive=0.008)   # ~14 cm/s
        # down into the pocket until the cube's underside is release_height above its floor: set down, not dropped
        yield from self.move(self.v(x, y, release_z), quat, max(0.5, (carry_z - release_z) / 0.08), settle=0.3, arrive=0.004)
        yield from self.pause()
        yield from self.jaw(L.JAW_OPEN_RAD, 0.3)
        self.released = True
        yield from self.hold(0.2)
        yield from self.move(self.v(x, y, carry_z), quat, 0.5, settle=0.0, arrive=0.03)

    def home(self):
        """Glide the joints back to the home pose (smooth start and stop), as the real arm parks."""
        s = self.s
        q0 = T(s.robot.data.joint_pos)[self.i, s.arm_ids].clone()
        far = float((s.home_q - q0).abs().max())
        steps = max(int(max(1.5 * far / 1.6, 0.8) / self.speed / self.dt), 1)     # peak joint speed about 1.6 rad/s
        s.override_on[self.i] = True
        for k in range(1, steps + 1):
            f = k / steps
            f = f * f * (3 - 2 * f)
            s.override[self.i] = q0 + (s.home_q - q0) * f
            yield

    def run(self):
        st = H.state(self.s.u)
        self.command(*self.s.tcp_pose(self.i))
        ok, tries = False, 0
        while True:
            ok, yaw, along, across = yield from self.pick()
            if ok or tries >= int(self.b["retries"]):
                break
            tries += 1
            self.notes.append("retry")
        if ok:
            yield from self.place(yaw, along, across)
        if self.b["return_home"]:
            yield from self.home()
        yield from self.hold(self.b["end_hold"])
        for _ in range(int(2.0 / self.dt)):                 # wait for the scorer's 0.5 s (gives up after 2 s)
            if bool(st["success"][self.i]) or not ok:
                break
            yield


def main():
    conf = _start.load_settings("generate", a.config)
    behavior = behavior_settings(conf)
    target = a.episodes if a.episodes is not None else int(conf.get("episodes", 50))
    seed = a.seed if a.seed is not None else int(conf.get("seed", 1))
    n_envs = a.envs if a.envs is not None else int(conf.get("envs", 1))
    root = os.environ.get("WORKSPACE") or os.path.expanduser("~/workspace")
    out = a.name if os.path.isabs(a.name) else os.path.join(root, "datasets", a.name)
    log(f"settings: {conf['_path']}")
    for note in R.reach_warnings(conf["randomization"]):
        log(f"note: {note}")
    cfg, _ = resolve_task_config(TASK, "", overrides=[])
    R.apply(cfg, conf["randomization"])
    if a.hdf5:
        n_envs = 1                            # Isaac Lab's demo recorder is used for one scene at a time here
        enable_hdf5_recorder(cfg, os.path.abspath(os.path.expanduser(a.hdf5)))
    cfg.scene.num_envs = n_envs
    cfg.scene.view_cam = None                 # the third-person camera is not recorded: do not render it
    cfg.seed = seed
    with launch_simulation(cfg, a):
        t0 = time.time()
        env = gym.make(TASK, cfg=cfg)
        scene = Scene(env)
        u = scene.u
        st = H.state(u)
        recs, already = None, 0
        if not a.dry_run:
            recs = [EpisodeRecorder(out) for _ in range(n_envs)]
            already = recs[0].index
            log(f"episodes go to {out} ({already} there already)")
            # adding to a dataset with the same seed continues its list of scenes instead of repeating it
            st["episode"][:] = already
        log(f"ready in {time.time() - t0:.0f} s: {target} episodes wanted, seed {seed}, {n_envs} scene(s) at a time")
        kept = attempts = frames = good = 0
        max_attempts = 4 * target + 4 * n_envs
        t1 = time.time()
        sim_steps = 0
        while kept < target and attempts < max_attempts:
            obs, _ = env.reset()
            scene.override_on[:] = False
            scene.grip[:] = T(scene.robot.data.joint_pos)[:, 5:6]
            arms = [Arm(scene, i, behavior, np.random.default_rng([seed, 7919, i, already + attempts + i]))
                    for i in range(n_envs)]
            runs = [arm.run() for arm in arms]
            active = [True] * n_envs
            if recs is not None:
                for r in recs:
                    r.clear()
            t = 0.0
            while True:
                for i in range(n_envs):
                    if active[i]:
                        try:
                            next(runs[i])
                        except StopIteration:
                            active[i] = False
                if not any(active):
                    break
                prev = obs
                obs, _, term, trunc, _ = env.step(scene.action())
                sim_steps += 1
                if recs is not None:
                    joints = prev["policy"]["joint_pos"].tolist()
                    targets = T(scene.robot.data.joint_pos_target).tolist()
                    cam_scene = prev["visual"]["scene"].cpu().numpy().astype(np.uint8)
                    cam_wrist = prev["visual"]["wrist"].cpu().numpy().astype(np.uint8)
                    for i in range(n_envs):
                        if active[i]:
                            recs[i].add(joints[i], targets[i], cam_scene[i], cam_wrist[i], t)
                t += scene.dt
            for i in range(n_envs):
                attempts += 1
                ok = bool(st["success"][i])
                good += int(ok)
                lay = st["layout"][i]
                if ok and kept < target:
                    kept += 1
                    if a.hdf5:
                        export_success(env)
                    if recs is not None and len(recs[i]) > 30:
                        frames += len(recs[i])
                        path = recs[i].save({"task": L.TASK, "instruction": L.INSTRUCTION, "success": True,
                                             "time_s": round(float(st["success_t"][i]), 3), "source": "generator",
                                             "seed": seed, "layout": lay, "config": os.path.basename(conf["_path"])})
                        if a.debug:
                            log(f"  saved {path}")
                if a.debug or not ok:
                    log(f"attempt {attempts}: {'success' if ok else 'FAILED'} cube {lay.get('cube_xy')} yaw "
                        f"{lay.get('cube_yaw_deg')} tray {lay.get('tray_xy')} yaw {lay.get('tray_yaw_deg')} | "
                        + "; ".join(arms[i].notes) + " | "
                        + json.dumps(H.debug_state(u, i)))
            wall = time.time() - t1
            log(f"{kept}/{target} episodes kept, {attempts} attempts, success rate {100.0 * good / attempts:.0f} %, "
                f"{60.0 * kept / wall:.1f} episodes/min")
        wall = time.time() - t1
        rate = 100.0 * good / max(attempts, 1)
        log(f"DONE: kept {kept} episodes; {good} of {attempts} attempts succeeded (success rate {rate:.0f} %) in {wall:.0f} s = "
            f"{60.0 * kept / wall:.1f} episodes per minute, {1000.0 * wall / max(sim_steps, 1):.0f} ms per step")
        if recs is not None:
            total = len([f for f in os.listdir(out) if f.endswith(".npz")])
            log(f"dataset: {out} ({total} episodes, {frames} new frames). Next: nwaves-dataset "
                f"{os.path.basename(out)} <hf_user>/<name>")
        if kept < target:
            log(f"stopped early: only {kept} of {target} after {attempts} attempts. The settings make the task too "
                f"hard for the script; narrow the ranges or look at one run with --dry-run --debug.")
        print("[generate] JSON " + json.dumps({"kept": kept, "attempts": attempts, "succeeded": good,
                                               "success_rate": round(rate / 100.0, 4),
                                               "seconds": round(wall, 1), "episodes_per_minute": round(60.0 * kept / wall, 2),
                                               "dataset": None if a.dry_run else out}), flush=True)
        env.close()
    sys.exit(0 if kept >= target else 1)


if __name__ == "__main__":
    main()
