"""Where can the arm work? Commands straight-down fingertip poses over a grid and reports how close the arm gets.

This is how the limits in nwaves_hackathon/layout.py were found (CUBE_REACH, TRAY_REACH). Run it again if you
change the tray or want to push a region further out.

    isaaclab -p scripts/reach.py                       # the heights of the grasp, of setting the cube down in the
                                                       # tray, and of carrying it over the tray's walls
    isaaclab -p scripts/reach.py --z 0.016,0.10        # your own heights (fingertip, metres)

For every grid point it holds the pose for 1.5 s and measures the fingertip position error and how far the
gripper is from pointing straight down. In the tables a dot means "served" (error below 3 mm and 6 deg); a number
is the position error in mm, with a t when the gripper also tilts by more than 6 deg. The cube is picked up low
over the table, set down 2 cm higher on the floor of the tray's pocket, and carried 7 cm up, over the tray's walls.
A coarser grid is faster:  --x 0.12:0.30:0.03 --y -0.20:0.12:0.04 --yaw-offsets 0,45
"""
import argparse

import _start  # noqa: F401  (first: this clone's nwaves_hackathon goes to the front of Python's path)
import warp as wp

wp.config.enable_backward = False
from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--z", default=None, help="comma list of fingertip heights [m] (default: grasp, set-down and carry height)")
p.add_argument("--x", default="0.10:0.34:0.02", help="lo:hi:step [m]")
p.add_argument("--y", default="-0.22:0.20:0.03", help="lo:hi:step [m]")
p.add_argument("--yaw-offsets", default="0,45,-45", help="jaw yaw relative to the base->point direction [deg]")
p.add_argument("--json", default=None, help="write all measurements here")
_start.window_args(p)
add_launcher_args(p)
import sys  # noqa: E402

# a value that starts with a minus sign (--y -0.2:0.2:0.03) would be taken for an option: glue it to its name
_argv, _k = [], 1
while _k < len(sys.argv):
    if sys.argv[_k] in ("--x", "--y", "--z", "--yaw-offsets") and _k + 1 < len(sys.argv):
        _argv.append(f"{sys.argv[_k]}={sys.argv[_k + 1]}")
        _k += 2
    else:
        _argv.append(sys.argv[_k])
        _k += 1
a = p.parse_args(_argv)
_start.window(a, "reach")

import json  # noqa: E402
import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as mu  # noqa: E402
import nwaves_hackathon  # noqa: E402,F401
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402
from nwaves_hackathon import layout as L  # noqa: E402
from nwaves_hackathon import mdp as H  # noqa: E402

TASK = "NWaves-SO101-PickPlace-Kit-v0"
T = H._t
OK_MM, OK_DEG = 3.0, 6.0


def grid(spec):
    lo, hi, step = (float(v) for v in spec.split(":"))
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 4) for i in range(n + 1)]


def main():
    cfg, _ = resolve_task_config(TASK, "", overrides=[])
    cfg.scene.num_envs = 1
    cfg.scene.scene_cam = cfg.scene.wrist_cam = cfg.scene.view_cam = None      # no pictures needed: much faster
    cfg.observations.visual = None
    hang = L.CUBE_SIZE / 2 + 0.009                    # fingertip above the cube's underside while it is held
    grasp_z = L.TABLE_Z + hang                        # the cube on the table
    place_z = L.TRAY_FLOOR_Z + 0.003 + hang           # the cube 3 mm above the pocket's floor: where the jaw opens
    carry_z = L.TABLE_Z + L.TRAY_HEIGHT + 0.02 + hang  # the generator's carry_clearance: 2 cm over the walls
    heights = [float(v) for v in a.z.split(",")] if a.z else [grasp_z, place_z, carry_z]
    xs, ys = grid(a.x), grid(a.y)
    offsets = [math.radians(float(v)) for v in a.yaw_offsets.split(",")]
    with launch_simulation(cfg, a):
        env = gym.make(TASK, cfg=cfg)
        u = env.unwrapped
        robot = u.scene["robot"]
        body = robot.body_names.index("gripper")
        tcp = torch.tensor([L.TCP_OFFSET], device=u.device)
        origin = u.scene.env_origins[0:1]
        env.reset()
        # park the cube and the tray out of the way
        ids = torch.tensor([0], device=u.device)
        for name, xy in (("cube", (-0.08, 0.30)), ("tray", (-0.08, -0.30))):
            pose = torch.tensor([[xy[0], xy[1], L.CUBE_Z if name == "cube" else L.TABLE_Z, 0.0, 0.0, 0.0, 1.0]], device=u.device)
            pose[:, :3] += origin
            u.scene[name].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        zero = torch.zeros(1, device=u.device)
        rows = []
        for z in heights:
            table = {}
            for xi, x in enumerate(xs):
                for y in (ys if xi % 2 == 0 else ys[::-1]):                  # snake through the grid
                    worst_mm, worst_deg = 0.0, 0.0
                    for off in offsets:
                        yaw = math.atan2(y, x) + off
                        pos_e = torch.tensor([[x, y, z]], device=u.device)
                        quat_e = mu.quat_from_euler_xyz(zero, zero, torch.full((1,), yaw, device=u.device))
                        pb, qb = mu.subtract_frame_transforms(T(robot.data.root_pos_w), T(robot.data.root_quat_w),
                                                              pos_e + origin, quat_e)
                        act = torch.cat([pb, qb, torch.tensor([[L.JAW_OPEN_RAD]], device=u.device)], dim=-1)
                        for _ in range(45):
                            env.step(act)
                        pw = T(robot.data.body_pos_w)[:, body]
                        qw = T(robot.data.body_quat_w)[:, body]
                        tip = pw + mu.quat_apply(qw, tcp) - origin
                        err = float((tip - pos_e).norm()) * 1000
                        zax = mu.quat_apply(qw, torch.tensor([[0.0, 0.0, 1.0]], device=u.device))[0]
                        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(zax[2])))))
                        rows.append({"x": x, "y": y, "z": round(z, 4), "yaw_offset_deg": round(math.degrees(off)),
                                     "error_mm": round(err, 2), "tilt_deg": round(tilt, 2)})
                        worst_mm, worst_deg = max(worst_mm, err), max(worst_deg, tilt)
                    table[(x, y)] = (worst_mm, worst_deg)
            print(f"\n[reach] fingertip height {z * 1000:.0f} mm: worst position error [mm] over the jaw yaws "
                  f"{a.yaw_offsets} deg ('.' = served: below {OK_MM:.0f} mm and {OK_DEG:.0f} deg; t = tilted more)", flush=True)
            print("[reach]    y -> " + " ".join(f"{y:+.2f}" for y in ys), flush=True)
            for x in xs:
                cells = []
                for y in ys:
                    mm, deg = table[(x, y)]
                    cells.append("   . " if (mm < OK_MM and deg < OK_DEG) else f"{min(mm, 999):4.0f}{'t' if deg >= OK_DEG else ' '}")
                print(f"[reach] x {x:.2f}  " + " ".join(cells), flush=True)
        if a.json:
            with open(a.json, "w") as f:
                json.dump(rows, f)
        env.close()


if __name__ == "__main__":
    main()
