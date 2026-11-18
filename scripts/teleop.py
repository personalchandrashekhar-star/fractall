"""Keyboard teleoperation of the SO-101 in the hackathon scene, with episode recording (Isaac Lab 3).

Run it from your clone (nwaves-teleop is the short form of the same command):

    isaaclab -p scripts/teleop.py                                 drive only
    isaaclab -p scripts/teleop.py --record my_teleop              + episodes -> ~/workspace/datasets/my_teleop/ep_*.npz
    isaaclab -p scripts/teleop.py --record my_teleop --config configs/my.yaml     scenes from your own settings file

The task: put the cube on the tray. The scenes are drawn from the ``randomization`` section of
configs/datagen.yaml, the same ranges the automatic generator uses.

The keys are the NWaves platform's keys for every arm (the same as on the real robots). They work in
the Terminal this runs in and in the Isaac Sim window after you click into it. Motion keys move while held down.
    Up / Down      fingertip forward / back           w / s  (or PgUp / PgDn)   up / down
    Left / Right   fingertip left / right             a / d                     wrist roll left / right
    q / e          gripper pitch up / down (tilt)
    space          open / close the gripper           , / .                     slower / faster (0.25x .. 2x)
    h              arm back to its home pose (not while recording)
    Enter          new scene with the arm at home, then RECORDING starts; Enter again saves the episode
                   (only if the cube lies on the tray and is released, unless --save-failed)
    Backspace      cancel the episode being recorded, nothing kept
    r              new scene (new layout, arm at home)   v   change the view: overview, scene camera, wrist camera
    Esc / Ctrl-C   quit

Episodes are the same files the automatic generator writes: joint states and joint targets in interface units,
both camera images, the task sentence. nwaves-dataset turns a folder of them into a LeRobot dataset. --hdf5 also
writes the Isaac Lab HDF5 that Isaac Lab Mimic reads.

How the arm moves: the keys move the fingertip in straight lines (forward / back, left / right, up / down as seen
from behind the arm, which is also how the window shows the scene), tilt the gripper about the fingertip, and roll
the wrist. The arm's kinematics turn that into joint targets (nwaves_hackathon/kinematics.py), so the arm goes where
the key says and stops when you let go. The fingertip position comes first: where the arm has to stretch, the
gripper comes up by itself and goes back down when there is room again. The speeds are per second on your clock,
whatever the speed of the simulation; the joint targets stay inside what the real arm allows.

Tips: tilt the gripper down (e) when the hand is low over the table; high up the wrist cannot bend that far. The
fingertip point the keys move lies on the fixed jaw: to grasp, put it about 12 mm beside the cube's centre, on the
side of the arm, and turn the wrist (a / d) until the jaws are square to the cube. Use , for fine moves. Hold the
cube by its upper half: the tray's walls are 29 mm high, and jaws that reach lower touch them when the cube goes in.
"""
import argparse

import _start  # noqa: F401  (first: this clone's nwaves_hackathon goes to the front of Python's path)
import warp as wp

wp.config.enable_backward = False
from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--record", default=None, help="dataset name under ~/workspace/datasets (or an absolute folder) for "
                                             "the episodes; nothing is saved without it")
p.add_argument("--hdf5", default=None, help="also store saved episodes in this HDF5 file (Isaac Lab Mimic input)")
p.add_argument("--config", default=None, help="settings file whose 'randomization' section draws the scenes "
                                             "(default: configs/datagen.yaml of this repository)")
p.add_argument("--seed", type=int, default=None)
p.add_argument("--speed", type=float, default=0.06, help="fingertip speed in m/s while a key is held (at 1x)")
p.add_argument("--turn", type=float, default=45.0, help="gripper turn speed in deg/s while a key is held (at 1x)")
p.add_argument("--sim-time", action="store_true", help="speeds per simulated second instead of per second on your "
                                                       "clock (the hand then moves slower when the simulation is slow)")
p.add_argument("--save-failed", action="store_true", help="Enter also saves episodes that are not a success")
p.add_argument("--script", default=None, help=argparse.SUPPRESS)   # test hook: "UP:1.5,wait:1,SPACE,..." then quit
_start.window_args(p)
add_launcher_args(p)
a = p.parse_args()
_start.window(a, "teleop")

import atexit  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import select  # noqa: E402
import sys  # noqa: E402
import termios  # noqa: E402
import time  # noqa: E402
import tty  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as mu  # noqa: E402
import nwaves_hackathon  # noqa: E402,F401
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402
from nwaves_hackathon import layout as L  # noqa: E402
from nwaves_hackathon import mdp as H  # noqa: E402
from nwaves_hackathon import randomization as R  # noqa: E402
from nwaves_hackathon.kinematics import ArmCommand, ArmKinematics  # noqa: E402
from nwaves_hackathon.recorder import EpisodeRecorder, enable_hdf5_recorder, export_success  # noqa: E402

T = H._t
TASK = "NWaves-SO101-PickPlace-Kit-v0"
JAW_OPEN, JAW_CLOSED = L.JAW_OPEN_RAD, L.JAW_CLOSED_RAD
WORK_MIN, WORK_MAX = (0.03, -0.38, 0.004), (0.47, 0.38, 0.36)        # fingertip box in the scene frame [m]
TILT_MIN, TILT_MAX = math.radians(-45.0), math.radians(120.0)       # gripper tilt: 0 = level, 90 deg = straight down
R_MIN = 0.10                   # m: fingertip distance from the arm's turning axis (closer, the base whips round)
HOLD_TORQUE = 1.2              # N m: when the keys are let go no joint keeps pushing harder than this (holding the arm
                               # up against gravity takes up to about 0.7 N m)
MAX_CATCH_UP = 3.0             # speeds are per second on your clock: one tick moves at most this many ticks' worth
HOME_SPEED = 1.0               # rad/s: joint speed of h (back to the SO-101 home), at least HOME_MIN_S long
HOME_MIN_S, HOME_SETTLE_S = 0.8, 0.4
SCALE_MIN, SCALE_MAX = 0.25, 2.0
# The window looks at the scene from behind the arm, a little to its right and from above: Up moves the hand away from
# you, Left moves it to the left of the picture.
VIEW_EYE, VIEW_TARGET = (-0.52, -0.17, 0.66), (0.21, -0.02, 0.05)
VIEWS = (("overview", "/OmniverseKit_Persp"), ("scene camera", "/World/envs/env_0/SceneCam"),
         ("wrist camera", "/World/envs/env_0/Robot/gripper/wowrobo_2MP_camera"))
MOTION = {"UP", "DOWN", "LEFT", "RIGHT", "W", "S", "PAGE_UP", "PAGE_DOWN", "A", "D", "Q", "E"}
EDGES = {"SPACE", "H", "COMMA", "PERIOD", "ENTER", "NUMPAD_ENTER", "BACKSPACE", "R", "V", "ESCAPE"}


def log(msg):
    print(f"[teleop] {msg}", flush=True)


CONTROLS = """
  keys (in this Terminal, or in the Isaac Sim window after clicking into it; motion keys move while held):
    Up / Down     forward / back          w / s (PgUp / PgDn)   up / down
    Left / Right  left / right            a / d                 wrist roll
    q / e         gripper tilt up / down  space                 open / close the gripper
    h             home                    , / .                 slower / faster
    Enter         new scene + RECORD; Enter again = save        Backspace   cancel the episode
    r             new scene               v                     change the view        Esc / Ctrl-C   quit
"""


# ----------------------------------------------------------------------------------------------------------- keys
class WindowKeys:
    """Keys pressed in the Isaac Sim window (the lab stream forwards the browser's keys into it)."""

    def __init__(self):
        import carb.input
        import omni.appwindow
        self.carb = carb.input
        self.held, self.edges = set(), []
        self._input = carb.input.acquire_input_interface()
        self._kb = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._kb, self._on_event)

    def _on_event(self, event, *args, **kwargs):
        name = getattr(event.input, "name", None)       # CHAR events carry a plain string here, not a key
        if name is None:
            return False
        t = event.type
        if t in (self.carb.KeyboardEventType.KEY_PRESS, self.carb.KeyboardEventType.KEY_REPEAT):
            if name in MOTION:
                self.held.add(name)
            elif name in EDGES and t == self.carb.KeyboardEventType.KEY_PRESS:
                self.edges.append("ENTER" if name == "NUMPAD_ENTER" else name)
        elif t == self.carb.KeyboardEventType.KEY_RELEASE:
            self.held.discard(name)
        return name in MOTION or name in EDGES          # ours: Kit does not also act on them

    def poll(self):
        edges, self.edges = self.edges, []
        return set(self.held), edges

    def close(self):
        try:
            self._input.unsubscribe_to_keyboard_events(self._kb, self._sub)
        except Exception:
            pass


class TerminalKeys:
    """Keys typed in the Terminal: read without blocking. A terminal only reports key presses (repeated while a key is
    held), so a motion key counts as held for a short while after each press."""

    SEQ = {"\x1b[A": "UP", "\x1b[B": "DOWN", "\x1b[C": "RIGHT", "\x1b[D": "LEFT", "\x1bOA": "UP", "\x1bOB": "DOWN",
           "\x1bOC": "RIGHT", "\x1bOD": "LEFT", "\x1b[5~": "PAGE_UP", "\x1b[6~": "PAGE_DOWN"}
    CHARS = {"w": "W", "s": "S", "a": "A", "d": "D", "q": "Q", "e": "E", " ": "SPACE", "h": "H", ",": "COMMA",
             ".": "PERIOD", "\r": "ENTER", "\n": "ENTER", "\x7f": "BACKSPACE", "\x08": "BACKSPACE", "r": "R", "v": "V"}
    TAP = 0.12           # s one press keeps a motion key down: a tap is a small nudge
    HOLD = 0.20          # s every further press of a held key (the keyboard's auto-repeat) keeps it down: long
                         # enough that a late key over the network does not stop the arm in the middle of a move
    REPEAT = 0.75        # s: a press this soon after the last one of the same key is auto-repeat, not a new tap

    def __init__(self):
        self.fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        self.saved = None
        self.until = {}
        self.last = {}
        if self.fd is not None:
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            atexit.register(self.close)

    def _press(self, name):
        now = time.monotonic()
        repeat = now - self.last.get(name, -10.0) < self.REPEAT
        self.last[name] = now
        self.until[name] = now + (self.HOLD if repeat else self.TAP)

    def poll(self):
        edges = []
        if self.fd is not None:
            data = ""
            while select.select([self.fd], [], [], 0)[0]:
                chunk = os.read(self.fd, 256).decode(errors="ignore")
                if not chunk:
                    break
                data += chunk
            i = 0
            while i < len(data):
                if data[i] == "\x1b":
                    for seq, name in self.SEQ.items():
                        if data.startswith(seq, i):
                            self._press(name)
                            i += len(seq)
                            break
                    else:
                        if i + 1 < len(data) and data[i + 1] in "[O":      # another escape sequence: skip it
                            j = i + 2
                            while j < len(data) and not data[j].isalpha() and data[j] != "~":
                                j += 1
                            i = j + 1
                        else:
                            edges.append("ESCAPE")
                            i += 1
                    continue
                name = self.CHARS.get(data[i].lower() if data[i].isalpha() else data[i])
                if name in MOTION:
                    self._press(name)
                elif name:
                    edges.append(name)
                i += 1
        now = time.monotonic()
        return {k for k, t in self.until.items() if t > now}, edges

    def close(self):
        if self.fd is not None and self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
            self.saved = None


class ScriptKeys:
    """Test hook: a fixed key sequence, e.g. "UP:1.5,wait:1,SPACE" (held keys with seconds, single presses)."""

    def __init__(self, text):
        self.steps = [s.strip() for s in text.split(",") if s.strip()]
        self.cur, self.left, self.done = None, 0.0, False

    def poll(self, dt):
        edges = []
        while self.left <= 0.0:
            if not self.steps:
                self.done = True
                return set(), edges
            name, _, sec = self.steps.pop(0).partition(":")
            if sec:
                self.cur, self.left = name.upper(), float(sec)
                log(f"script: {name} for {sec} s")
            else:
                edges.append(name.upper())
                log(f"script: press {name}")
                return set(), edges
        self.left -= dt
        return ({self.cur} if self.cur not in ("WAIT", None) else set()), edges


def free_kit_hotkeys():
    """Kit binds some of our keys to its own actions (e.g. Space = play / pause the timeline): drop those bindings."""
    ours = {k.replace("_", "") for k in MOTION | EDGES}
    try:
        from omni.kit.hotkeys.core import get_hotkey_registry
        reg = get_hotkey_registry()
        dropped = []
        for hk in list(reg.get_all_hotkeys()):
            text = str(getattr(hk, "key_text", "") or "")            # e.g. "SPACE", "CTRL + S"
            if "+" in text:
                continue                                              # combinations with modifiers are not ours
            if text.replace(" ", "").replace("_", "").upper() in ours:
                reg.deregister_hotkey(hk)
                dropped.append(f"{text} ({getattr(hk, 'action_text', '?')})")
        if dropped:
            log(f"Kit hotkeys released for the teleop: {', '.join(dropped)}")
    except Exception as exc:
        log(f"could not release Kit's hotkeys ({exc}); the teleop resumes the simulation if a key pauses it")


def keep_playing():
    """Isaac Lab blocks every step while the timeline is paused (toolbar button or a Kit hotkey), which would freeze
    the teleop. Resume right away; returns the subscription (keep a reference)."""
    import omni.timeline
    tl = omni.timeline.get_timeline_interface()

    def on_event(e):
        if getattr(e, "type", None) == int(omni.timeline.TimelineEventType.PAUSE):
            log("the simulation was paused; resuming (the teleop needs it running)")
            tl.play()
    try:
        return tl.get_timeline_event_stream().create_subscription_to_pop(on_event)
    except Exception as exc:
        log(f"(no timeline guard: {exc})")
        return None


def ensure_playing():
    """A key pressed in the Isaac Sim window while it was still loading (Space = Kit's play / pause) leaves the
    timeline paused before the guard above exists; every step would then block for ever. Start it."""
    try:
        import omni.timeline
        tl = omni.timeline.get_timeline_interface()
        if not tl.is_playing():
            log("the simulation was not running (a key pressed while Isaac Sim was loading pauses it); starting it")
            tl.play()
    except Exception as exc:
        log(f"(could not check the timeline: {exc})")


# ----------------------------------------------------------------------------------------------------------- env
def load_cfg():
    conf = _start.load_settings("teleop", a.config)
    log(f"scenes are drawn from {conf['_path']}")
    cfg, _ = resolve_task_config(TASK, "", overrides=[])
    R.apply(cfg, conf["randomization"])
    cfg.scene.num_envs = 1
    cfg.scene.view_cam = None          # the stream's third-person camera: not needed here
    cfg.viewer.eye, cfg.viewer.lookat = VIEW_EYE, VIEW_TARGET      # the window looks from behind the arm
    if a.seed is not None:
        cfg.seed = a.seed
    if a.hdf5:
        enable_hdf5_recorder(cfg, os.path.abspath(os.path.expanduser(a.hdf5)))
    if hasattr(cfg.events, "reset_robot"):          # new scenes start exactly at the SO-101 home
        cfg.events.reset_robot.params["position_range"] = (0.0, 0.0)
    return cfg, conf


def main():
    cfg, conf = load_cfg()
    with launch_simulation(cfg, a):
        env = gym.make(TASK, cfg=cfg)
        u = env.unwrapped
        robot = u.scene["robot"]
        dev = u.device
        dt = u.step_dt
        body = robot.body_names.index("gripper")
        tcp_off = torch.tensor([L.TCP_OFFSET], device=dev)
        origin = u.scene.env_origins[0:1]

        def tcp_env():
            """Fingertip pose in the scene frame (x forward from the base, y left, z up): the frame of the keys."""
            pw = T(robot.data.body_pos_w)[:, body]
            qw = T(robot.data.body_quat_w)[:, body]
            return pw + mu.quat_apply(qw, tcp_off) - origin, qw

        arm_term = u.action_manager.get_term("arm_action")
        arm_ids = [robot.joint_names.index(j) for j in L.HOME_RAD]
        home_q = torch.tensor([[L.HOME_RAD[j] for j in L.HOME_RAD]], device=dev)

        def joints():
            """Measured arm joint angles [5] (rad)."""
            return T(robot.data.joint_pos)[0, arm_ids].cpu().numpy().astype(float)

        class Homing:
            """h: the joints glide to the SO-101 home (smooth start and stop), like the real arm parks."""
            active = False

            def start(self):
                self.q0 = T(robot.data.joint_pos)[:, arm_ids].clone()
                far = float((home_q - self.q0).abs().max())
                self.n = max(int(max(far / HOME_SPEED, HOME_MIN_S) / dt), 1)
                self.i, self.active = 0, True

            def step(self):
                self.i += 1
                k = min(self.i / self.n, 1.0)
                s_ = k * k * (3.0 - 2.0 * k)
                arm_term.joint_override = self.q0 + (home_q - self.q0) * s_
                if self.i >= self.n + int(HOME_SETTLE_S / dt):
                    self.active = False
                    return True
                return False

            def stop(self):
                self.active = False

        def to_action(grip):
            """The arm follows the joint command (``arm_term.joint_override``). The action also carries the fingertip
            pose that goes with it, where the fingertip is expected to be (the command's pose, shifted by how far the
            arm hangs below its joint targets): that is what the Isaac Lab demo recorder stores for Mimic."""
            nonlocal sag
            off = tcp_env()[0][0].cpu().numpy() - kin.tip(cmd.sent)       # the arm against its last command
            if np.isfinite(off).all() and float(np.linalg.norm(off)) < 0.04:
                sag = 0.9 * sag + 0.1 * off
            if not homing.active:
                arm_term.joint_override = torch.tensor(cmd.target()[None], device=dev, dtype=torch.float32)
            pos, rot = kin.pose(cmd.q)
            pos_t = torch.tensor((pos + sag)[None], device=dev, dtype=torch.float32) + origin
            quat_t = mu.quat_from_matrix(torch.tensor(rot[None], device=dev, dtype=torch.float32))
            pos_b, quat_b = mu.subtract_frame_transforms(T(robot.data.root_pos_w), T(robot.data.root_quat_w), pos_t, quat_t)
            return torch.cat([pos_b, quat_b, torch.tensor([[grip]], device=dev)], dim=-1)

        def set_view(i):
            try:
                from omni.kit.viewport.utility import get_active_viewport
                get_active_viewport().camera_path = VIEWS[i][1]
                log(f"view: {VIEWS[i][0]}")
            except Exception as exc:
                log(f"could not switch the view ({exc})")

        script = ScriptKeys(a.script) if a.script else None
        window = None
        terminal = None
        guard = keep_playing()                         # noqa: F841 (kept alive for the whole session)
        ensure_playing()
        if script is None:
            free_kit_hotkeys()
            try:
                window = WindowKeys()
            except Exception as exc:
                log(f"no Isaac Sim window keyboard ({exc}); use the Terminal")
            terminal = TerminalKeys()

        rec = None
        if a.record:
            workspace = os.environ.get("WORKSPACE") or os.path.expanduser("~/workspace")
            folder = a.record if os.path.isabs(a.record) else os.path.join(workspace, "datasets", a.record)
            rec = EpisodeRecorder(folder)
            log(f"episodes go to {folder} ({rec.index} there already)")
        obs, _ = env.reset()
        ensure_playing()
        # the arm's kinematics, read from the robot in the scene, and a check against the simulated fingertip
        import omni.usd
        kin = ArmKinematics.from_stage(omni.usd.get_context().get_stage(), "/World/envs/env_0/Robot",
                                       (T(robot.data.root_pos_w) - origin)[0].tolist(), T(robot.data.root_quat_w)[0].tolist(),
                                       L.TCP_OFFSET, T(robot.data.soft_joint_pos_limits)[0, arm_ids].cpu().numpy())
        off = float(np.linalg.norm(kin.tip(joints()) - tcp_env()[0][0].cpu().numpy()))
        if off > 0.002:
            log(f"WARNING: the kinematics and the simulated arm disagree by {1000 * off:.1f} mm; the keys may not move "
                f"in straight lines")
        cmd = ArmCommand(kin, WORK_MIN, WORK_MAX, dt=dt, r_min=R_MIN, tilt_range=(TILT_MIN, TILT_MAX),
                         max_step=math.radians(L.MAX_STEP_DEG), max_lead=math.radians(L.MAX_LEAD_DEG),
                         hold_torque=HOLD_TORQUE)
        sag = np.zeros(3)                               # how far the fingertip hangs below its command (slow average)

        def anchor(q):
            cmd.anchor(q, T(robot.data.joint_stiffness)[0, arm_ids].cpu().numpy(),
                       T(robot.data.joint_damping)[0, arm_ids].cpu().numpy())

        anchor(joints())
        homing = Homing()
        moving = False
        grip = JAW_OPEN
        scale, view, recording, saved, episodes = 1.0, 0, False, 0, 0
        t, last_status, edge_note = 0.0, -10.0, 0.0
        tick_start = time.monotonic()
        print(CONTROLS, flush=True)
        log(f"task: {L.INSTRUCTION}" + ("  Enter starts recording." if rec is not None or a.hdf5 else ""))

        def new_scene(reason):
            nonlocal obs, recording, t, last_status
            if recording and rec is not None:
                rec.clear()
            recording = False
            homing.stop()
            obs, _ = env.reset()
            anchor(joints())
            t, last_status = 0.0, -10.0
            log(f"new scene ({reason}): arm at home")

        try:
            while True:
                # Speeds are per second on your clock. The simulation with its window and cameras runs slower than
                # real time, so one tick moves the hand by as much as the time it really took (at most MAX_CATCH_UP
                # ticks' worth); when it runs faster, the loop waits.
                now = time.monotonic()
                spare = dt - (now - tick_start)
                if spare > 0 and script is None:
                    time.sleep(spare)
                    now = time.monotonic()
                wall, tick_start = now - tick_start, now
                step_s = dt if (script is not None or a.sim_time) else min(max(wall, dt), MAX_CATCH_UP * dt)
                if script is not None:
                    held, edges = script.poll(dt)
                    if script.done:
                        log("script done")
                        break
                else:
                    held, edges = set(), []
                    if window is not None:
                        h, e = window.poll()
                        held |= h
                        edges += e
                    h, e = terminal.poll()
                    held |= h
                    edges += e

                for k in edges:
                    if k == "ESCAPE":
                        raise KeyboardInterrupt
                    if k == "COMMA":
                        scale = max(SCALE_MIN, scale / 1.5)
                        log(f"speed x{scale:.2f}")
                    elif k == "PERIOD":
                        scale = min(SCALE_MAX, scale * 1.5)
                        log(f"speed x{scale:.2f}")
                    elif k == "SPACE":
                        grip = JAW_CLOSED if grip == JAW_OPEN else JAW_OPEN
                        log("gripper -> " + ("close" if grip == JAW_CLOSED else "open"))
                    elif k == "H":
                        if recording:
                            log("h does nothing while recording (Backspace cancels the episode)")
                        else:
                            homing.start()
                            log("homing ...")
                    elif k == "V":
                        view = (view + 1) % len(VIEWS)
                        set_view(view)
                    elif k == "R":
                        new_scene("r")
                    elif k == "BACKSPACE":
                        if recording:
                            if rec is not None:
                                rec.clear()
                            recording = False
                            log("episode cancelled, nothing kept")
                    elif k == "ENTER":
                        if rec is None and not a.hdf5:
                            log("not recording: start with --record NAME (and/or --hdf5 FILE) to save episodes")
                        elif not recording:
                            new_scene("Enter")
                            grip = JAW_OPEN
                            recording = True
                            log(f"RECORDING episode {episodes + 1} ... (Enter = save, Backspace = cancel)")
                        else:
                            complete = bool(H.success_now(u)[0])
                            ok = complete or a.save_failed
                            n = len(rec) if rec is not None else 999
                            if not ok:
                                log("not saved: the cube is not lying flat on the tray with the gripper open yet "
                                    "(finish it, or Backspace to cancel)")
                            elif n <= 30:
                                log("not saved: too short")
                            else:
                                if a.hdf5 and complete:
                                    export_success(env)
                                    log(f"HDF5: episode stored in {a.hdf5}")
                                if rec is not None:
                                    st = H.state(u)
                                    path = rec.save({"task": L.TASK, "instruction": L.INSTRUCTION, "success": complete,
                                                     "time_s": round(t, 3), "source": "teleop-keyboard",
                                                     "layout": st["layout"][0],
                                                     "config": os.path.basename(conf["_path"])})
                                    log(f"saved {path} ({'success' if complete else 'not a success'})")
                                saved += 1
                                episodes += 1
                                recording = False
                                log("Enter: new scene and the next episode")

                # ---- motion
                if homing.active:
                    if homing.step():
                        anchor(home_q[0].cpu().numpy())
                        log("at home")
                elif held & MOTION:
                    moving = True
                    v = a.speed * scale * step_s
                    w = math.radians(a.turn) * scale * step_s
                    d = np.array([v * (("UP" in held) - ("DOWN" in held)),
                                  v * (("LEFT" in held) - ("RIGHT" in held)),
                                  v * ((("W" in held) or ("PAGE_UP" in held)) - (("S" in held) or ("PAGE_DOWN" in held)))])
                    why = cmd.move(d, w * (("E" in held) - ("Q" in held)), w * (("A" in held) - ("D" in held)), joints())
                    if why and time.monotonic() - edge_note > 2.0:
                        log("edge of the workspace: the arm does not reach further that way" if why == "edge" else
                            "the wrist cannot tilt further here: lower the hand or bring it closer to the base first")
                        edge_note = time.monotonic()
                elif moving:
                    moving = False
                    cmd.release(joints())                           # keys let go: the arm stops

                prev = obs
                obs, _, term, trunc, _ = env.step(to_action(grip))
                t += dt
                if recording and rec is not None:
                    rec.add(prev["policy"]["joint_pos"][0].tolist(), T(robot.data.joint_pos_target)[0].tolist(),
                            prev["visual"]["scene"][0].cpu().numpy().astype(np.uint8),
                            prev["visual"]["wrist"][0].cpu().numpy().astype(np.uint8), t - dt)
                if t - last_status >= (0.5 if script is not None else 2.0):
                    on_tray = bool(H.success_now(u)[0])
                    fp = tcp_env()[0][0].tolist()
                    q_now = joints()
                    state = f"REC {len(rec) if rec is not None else 0} frames" if recording else "not recording"
                    log(f"t {t:5.1f} s | {state} | fingertip x {fp[0]:.3f} y {fp[1]:.3f} z {fp[2]:.3f} m | tilt "
                        f"{math.degrees(kin.tilt(q_now)):.0f} roll {math.degrees(q_now[4]):.0f} deg | speed x{scale:.2f} | "
                        f"cube on tray: {'yes' if on_tray else 'no'} | saved {saved}")
                    if script is not None:
                        q = T(robot.data.joint_pos)[0].tolist()
                        log("joints " + " ".join(f"{n}={math.degrees(x):.1f}" for n, x in zip(robot.joint_names, q)))
                    last_status = t
                if bool(term.any()) or bool(trunc.any()):
                    new_scene("episode time is up" if bool(trunc.any()) else "task done")
        except KeyboardInterrupt:
            log("quit")
        finally:
            if terminal is not None:
                terminal.close()
            if window is not None:
                window.close()
        env.close()


if __name__ == "__main__":
    main()
