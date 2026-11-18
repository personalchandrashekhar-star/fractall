"""Resets, randomisation and the success test of the NWaves SO-101 pick-and-place task (Isaac Lab 3 API).

The task: put the cube on the tray. An episode is a success when the cube lies flat on the floor of the tray's
pocket, inside the walls (its centre inside the pocket, the cube level, its underside on the pocket's floor: a
cube that hangs on a wall or sits on top of the walls does not count), the gripper has let go of it, and this
has held for 0.5 s.

Every reset draws a new scene from a dict of ranges (see ``layout.OFFICIAL_RANDOMIZATION`` for the names):
cube position, yaw and colour, tray position and yaw, the look of the workspace (the table top, the colour of the
walls and of the arm), light brightness and colour temperature, a small jitter of the
scene camera, cube friction and mass, motor gains. What is drawn depends only on the seed and on the episode
number, so the same seed always gives the same list of scenes, whatever the policy does.

State lives on the env as ``env._hack`` (see ``state``): the scorer reads finished episodes from it, because
the env resets itself inside ``step()``.
"""
from __future__ import annotations

import math

import numpy as np
import torch

import isaaclab.utils.math as math_utils

from . import layout as L

REST_SPEED = 0.05          # m/s: the cube counts as resting below this
CONTACT_N = 1.0            # N: a jaw contact force above this counts as "the gripper holds the cube"
STABLE_STEPS = int(round(L.SUCCESS_HOLD_SECONDS * L.CONTROL_HZ))   # 0.5 s at 30 Hz


def _t(x):
    """Isaac Lab 3 data buffers expose ``.torch``; plain tensors pass through."""
    return x.torch if hasattr(x, "torch") else x


# ----------------------------------------------------------------------------------------- state
def state(env) -> dict:
    """Per-environment bookkeeping of the task (created on first use)."""
    st = getattr(env, "_hack", None)
    if st is None:
        n = env.num_envs
        seed = getattr(env.cfg, "seed", None)
        st = {
            "seed": int(seed) if seed is not None else int(np.random.SeedSequence().entropy % (2 ** 31)),
            "episode": torch.zeros(n, dtype=torch.long),        # number of the next episode of each env
            "layout": [dict() for _ in range(n)],               # what the last reset drew for each env
            "count": torch.zeros(n, dtype=torch.long),          # control ticks the success test has held
            "success": torch.zeros(n, dtype=torch.bool),        # confirmed (held for STABLE_STEPS)
            "success_t": torch.zeros(n),                        # time the cube came to rest on the tray [s]
            "started": torch.zeros(n, dtype=torch.bool),
            "table_z": torch.full((n,), L.TABLE_Z),             # height of the surface the cube and the tray lie on
            "finished": [[] for _ in range(n)],                 # results of the episodes that ended
            "force_layout": None,                               # a layout dict to use at the next resets instead of
        }                                                       # drawing one (replays, pictures); see sample_layout
        env._hack = st
    return st


# ---------------------------------------------------------------------------------- appearance
_shaders: dict = {}


def _env_prim_path(cfg_path: str, env_id: int) -> str:
    """``{ENV_REGEX_NS}/Cube`` or ``/World/envs/env_[^/]+/Cube`` -> the prim of one env."""
    import re
    cfg_path = cfg_path.replace("{ENV_REGEX_NS}", "/World/envs/env_.*").replace("env_[^/]+", "env_.*")
    return re.sub(r"/World/envs/env_(\.\*|\d+)", f"/World/envs/env_{env_id}", cfg_path)


def _shader(env, env_id: int, asset: str):
    """The preview-surface shader of a spawned shape (cached)."""
    key = (id(env), env_id, asset)
    if key not in _shaders:
        from pxr import Usd
        from isaaclab.sim.utils.stage import get_current_stage
        stage = get_current_stage()
        root = _env_prim_path(env.scene[asset].cfg.prim_path, env_id)
        shader = stage.GetPrimAtPath(root + "/geometry/material/Shader")       # Isaac Lab 3 PreviewSurface layout
        if not (shader.IsValid() and shader.GetAttribute("inputs:diffuseColor").IsValid()):
            shader = None
            prim = stage.GetPrimAtPath(root)
            if prim.IsValid():
                for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                    if p.GetTypeName() == "Shader" and p.GetAttribute("inputs:diffuseColor").IsValid():
                        shader = p
                        break
        _shaders[key] = shader
    return _shaders[key]


def set_color(env, env_id: int, asset: str, rgb) -> None:
    """Set the diffuse colour of a scene asset with a preview material (the cube, the mat)."""
    from pxr import Gf
    shader = _shader(env, env_id, asset)
    if shader is not None:
        shader.GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(*[float(min(max(c, 0.0), 1.0)) for c in rgb]))


_look_attrs: dict = {}


def _stage():
    from isaaclab.sim.utils.stage import get_current_stage
    return get_current_stage()


def _vec(rgb):
    from pxr import Gf
    return Gf.Vec3f(*[float(min(max(c, 0.0), 1.0)) for c in rgb])


def set_enclosure_color(env, env_id: int, rgb) -> None:
    """Set the colour of the enclosure's own material (its walls, and its floor where that shows; the logo and the
    letters keep theirs). ``None`` = the colour it was built with."""
    from pxr import Usd
    key = (id(env), env_id, "enclosure")
    if key not in _look_attrs:
        found = []
        prim = _stage().GetPrimAtPath(f"/World/envs/env_{env_id}/Enclosure")
        if prim.IsValid():
            for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                attr = p.GetAttribute("inputs:diffuseColor") if p.GetTypeName() == "Shader" else None
                if attr and attr.IsValid() and "/Looks/Diffuse/" in str(p.GetPath()) and attr.Get() is not None:
                    found.append(attr)
        _look_attrs[key] = found
    for attr in _look_attrs[key]:
        attr.Set(_vec(L.ENCLOSURE_COLOR if rgb is None else rgb))
        metallic = attr.GetPrim().GetAttribute("inputs:metallic")
        if metallic and metallic.IsValid():
            metallic.Set(0.5 if rgb is None else 0.0)        # as built it is sheet metal; in a colour it is painted


def set_arm_color(env, env_id: int, rgb) -> None:
    """Set the colour of the arm's printed parts (the servos, the camera and the screws keep theirs).
    ``None`` = the yellow the model was built with. The model shares one material between all printed parts and
    keeps it read-only (an instance); the first call makes this arm's copy of it editable."""
    from pxr import Usd
    key = (id(env), env_id, "arm")
    if key not in _look_attrs:
        found = []
        looks = _stage().GetPrimAtPath(f"/World/envs/env_{env_id}/Robot/Looks")
        if looks.IsValid():
            for material in looks.GetChildren():
                if material.IsInstanceable():
                    material.SetInstanceable(False)
                for p in Usd.PrimRange(material):
                    attr = p.GetAttribute("inputs:diffuse_color_constant") if p.GetTypeName() == "Shader" else None
                    value = attr.Get() if attr and attr.IsValid() else None
                    # the printed parts are the ones built in the model's yellow
                    if value is not None and all(abs(float(value[k]) - L.ARM_YELLOW[k]) < 0.02 for k in range(3)):
                        found.append(attr)
        _look_attrs[key] = found
    for attr in _look_attrs[key]:
        attr.Set(_vec(L.ARM_YELLOW if rgb is None else rgb))


def set_surface_look(env, env_id: int, scale, bias) -> None:
    """The table top's look: the wood image times ``scale`` plus ``bias`` (per colour channel)."""
    from pxr import Gf
    key = (id(env), env_id, "surface")
    if key not in _look_attrs:
        image = _stage().GetPrimAtPath(f"/World/envs/env_{env_id}/Surface/Looks/TableTop/Image")
        _look_attrs[key] = (image.GetAttribute("inputs:scale"), image.GetAttribute("inputs:bias")) if image.IsValid() else None
    if _look_attrs[key] is not None:
        _look_attrs[key][0].Set(Gf.Vec4f(float(scale[0]), float(scale[1]), float(scale[2]), 1.0))
        _look_attrs[key][1].Set(Gf.Vec4f(float(bias[0]), float(bias[1]), float(bias[2]), 0.0))


def _rgb(color):
    if isinstance(color, str):
        if color not in L.COLORS:
            raise ValueError(f"unknown colour '{color}'; known names: {', '.join(L.COLORS)} (or give [r, g, b])")
        return L.COLORS[color]
    rgb = tuple(float(c) for c in color)
    if len(rgb) != 3:
        raise ValueError(f"a colour is a name or [r, g, b], got {color!r}")
    return rgb


def luminance(rgb) -> float:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def bright_color(rng):
    """Any bright colour, the real cube's colour included: the real cube's colour in layout.REAL_CUBE_SHARE of the
    draws; otherwise any hue, any saturation, never dark (layout.BRIGHT_COLOR_VALUE, BRIGHT_COLOR_MIN_LUMINANCE)."""
    import colorsys
    if float(rng.random()) < L.REAL_CUBE_SHARE:
        j = L.REAL_CUBE_JITTER
        return tuple(min(max(c + j * (2.0 * float(rng.random()) - 1.0), 0.0), 1.0) for c in L.COLORS[L.REAL_CUBE_COLOR])
    for _ in range(200):
        saturation = float(rng.random()) ** 0.5            # strong colours a little more often than pale ones
        rgb = colorsys.hsv_to_rgb(float(rng.random()), saturation, _uniform(rng, L.BRIGHT_COLOR_VALUE))
        if luminance(rgb) >= L.BRIGHT_COLOR_MIN_LUMINANCE:
            return rgb
    return L.COLORS["white"]


def dark_color(rng):
    """Black or nearly black, any hue: see layout.DARK_COLOR_VALUE."""
    import colorsys
    return colorsys.hsv_to_rgb(float(rng.random()), float(rng.random()), _uniform(rng, L.DARK_COLOR_VALUE))


def any_color(rng):
    """Any colour, black and dark ones included: dark in layout.ANY_COLOR_DARK_SHARE of the draws, otherwise a
    bright_color() draw."""
    if float(rng.random()) < L.ANY_COLOR_DARK_SHARE:
        return dark_color(rng)
    return bright_color(rng)


COLOR_WORDS = {"bright": bright_color, "any": any_color, "black": lambda rng: L.COLORS["black"]}


def sample_color(rng, ranges: dict):
    """The cube colour of one episode: "bright", "any" or "black" (see layout.py), or one of a list (plus an
    optional jitter per channel)."""
    colors = ranges.get("cube_colors") or "any"
    if isinstance(colors, str):
        if colors not in COLOR_WORDS:
            raise ValueError(f"cube_colors is '{colors}'; use bright, any, black or a list of colours")
        return COLOR_WORDS[colors](rng)
    base = _rgb(colors[int(rng.integers(len(colors)))])
    jitter = float(ranges.get("cube_color_jitter", 0.0))
    return [min(max(c + jitter * (2.0 * float(rng.random()) - 1.0), 0.0), 1.0) for c in base]


def _hsv(rng, saturation, value):
    import colorsys
    return colorsys.hsv_to_rgb(float(rng.random()), _uniform(rng, saturation), _uniform(rng, value))


def sample_look(rng, ranges: dict) -> dict:
    """The look of the workspace for one episode (see layout.py): the surface, the colour of the walls, of the arm
    and of the mat. Colours of None mean "as built"."""
    surfaces = ranges.get("surface") or L.LOOKS["real"]["surface"]
    pick = surfaces[int(rng.integers(len(surfaces)))]
    scale, bias = None, None                                    # None: the table top is not shown
    if pick == "wood":
        scale, bias = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
    elif pick == "wood_any":
        bright = _uniform(rng, L.SURFACE_BRIGHTNESS)
        scale = tuple(bright * (1.0 + L.SURFACE_TINT * (2.0 * float(rng.random()) - 1.0)) for _ in range(3))
        bias = (0.0, 0.0, 0.0)
    elif pick == "plain_any":
        scale, bias = (0.0, 0.0, 0.0), _hsv(rng, L.PLAIN_SATURATION, L.PLAIN_VALUE)
    elif pick not in ("grey", "mat"):
        scale, bias = (0.0, 0.0, 0.0), _rgb(pick)               # a plain top in a given colour

    def one(name, any_color):
        colors = ranges.get(name)
        if isinstance(colors, str) and colors == "any":
            return any_color()
        choice = colors[int(rng.integers(len(colors)))]
        return None if choice == "as_built" else _rgb(choice)

    def arm_any():
        t = float(rng.random()) ** L.ARM_ANY_POWER
        return tuple(b + t * (y - b) for b, y in zip(L.ARM_BLACK, L.ARM_YELLOW))

    walls = one("wall_colors", lambda: _hsv(rng, L.WALL_SATURATION, L.WALL_VALUE))
    arm = one("arm_colors", arm_any)
    mats = ranges.get("mat_colors") or [L.MAT_COLOR]
    rounded = lambda c: None if c is None else [round(float(v), 3) for v in c]     # noqa: E731
    return {"surface": pick if isinstance(pick, str) else "plain", "surface_scale": rounded(scale),
            "surface_bias": rounded(bias), "wall_rgb": rounded(walls), "arm_rgb": rounded(arm),
            "mat_rgb": rounded(_rgb(mats[int(rng.integers(len(mats)))]))}


# --------------------------------------------------------------------------------------- reset
def _uniform(rng, rng_range) -> float:
    lo, hi = float(rng_range[0]), float(rng_range[1])
    return lo + (hi - lo) * float(rng.random())


def sample_layout(rng, ranges: dict) -> dict:
    """Draw the cube and the tray for one episode. They never start closer than ``layout.MIN_CUBE_TRAY``
    (centre to centre): the cube stands on the table with room for the gripper beside the tray."""
    for _ in range(2000):
        cube = (_uniform(rng, ranges["cube_x"]), _uniform(rng, ranges["cube_y"]))
        tray = (_uniform(rng, ranges["tray_x"]), _uniform(rng, ranges["tray_y"]))
        if math.hypot(cube[0] - tray[0], cube[1] - tray[1]) >= L.MIN_CUBE_TRAY:
            break
    else:
        raise RuntimeError(f"the cube and tray ranges leave no place where their centres are "
                           f"{L.MIN_CUBE_TRAY * 100:.0f} cm apart: move the ranges away from each other")
    rgb = sample_color(rng, ranges)
    lay = {
        "cube_xy": [round(cube[0], 4), round(cube[1], 4)],
        "cube_yaw_deg": round(_uniform(rng, ranges["cube_yaw_deg"]), 2),
        "cube_rgb": [round(c, 3) for c in rgb],
        "tray_xy": [round(tray[0], 4), round(tray[1], 4)],
        "tray_yaw_deg": round(_uniform(rng, ranges["tray_yaw_deg"]), 2),
    }
    lay.update(sample_look(rng, ranges))
    return lay


# what a stored scene holds (episode files keep it; replay.py lays the same scene out again)
LAYOUT_KEYS = ("cube_xy", "cube_yaw_deg", "cube_rgb", "tray_xy", "tray_yaw_deg", "surface", "surface_scale",
               "surface_bias", "wall_rgb", "arm_rgb", "mat_rgb")


def _finalize(st: dict, env_id: int, env) -> None:
    """Store the result of the episode that just ended."""
    if not bool(st["started"][env_id]):
        return
    ok = bool(st["success"][env_id])
    st["finished"][env_id].append({
        "episode": int(st["episode"][env_id]) - 1,
        "success": ok,
        "time_s": round(float(st["success_t"][env_id]), 3) if ok else None,
        "layout": dict(st["layout"][env_id]),
    })


_camera_base: dict = {}


def _randomize_lights(rng, ranges: dict, env_id: int, shared: bool) -> dict:
    """Brightness and colour temperature of the lights. The two roof strips belong to one enclosure and are drawn
    for every environment; the dome light and the key light outside the box are shared by all environments and
    follow the first one of a reset (``shared``)."""
    from isaaclab.sim.utils.stage import get_current_stage
    stage = get_current_stage()
    temperature = _uniform(rng, ranges["light_temperature"])
    scales = {name: _uniform(rng, ranges["light_intensity"]) for name in ("strips", "dome", "key")}
    lights = [(f"/World/envs/env_{env_id}/RoofStrip_{k}", L.STRIP_INTENSITY * scales["strips"], temperature) for k in (1, 2)]
    if shared:
        outside = L.OUTSIDE_TEMPERATURE * temperature / L.STRIP_TEMPERATURE
        lights += [("/World/DomeLight", L.DOME_INTENSITY * scales["dome"], outside),
                   ("/World/KeyLight", L.KEY_INTENSITY * scales["key"], outside)]
    for path, intensity, kelvin in lights:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        prim.GetAttribute("inputs:intensity").Set(float(intensity))
        if prim.GetAttribute("inputs:enableColorTemperature"):
            prim.GetAttribute("inputs:enableColorTemperature").Set(True)
            prim.GetAttribute("inputs:colorTemperature").Set(float(kelvin))
    return {"strips": round(scales["strips"], 3), "dome": round(scales["dome"], 3), "key": round(scales["key"], 3),
            "temperature": round(temperature)}


def _jitter_camera(env, rng, env_id: int, ranges: dict, camera: str = "scene_cam") -> None:
    """Move the scene camera a little around its mount: the real camera is never exactly where the sim has it."""
    if camera not in env.scene.keys():
        return
    sensor = env.scene[camera]
    key = (id(env), camera)
    if key not in _camera_base:
        _camera_base[key] = (_t(sensor.data.pos_w).clone(), _t(sensor.data.quat_w_world).clone())
    pos, quat = _camera_base[key]
    dp_max, da_max = float(ranges["camera_pos_jitter"]), math.radians(float(ranges["camera_rot_jitter_deg"]))
    dp = torch.tensor([[dp_max * (2.0 * float(rng.random()) - 1.0) for _ in range(3)]], device=pos.device, dtype=pos.dtype)
    da = [da_max * (2.0 * float(rng.random()) - 1.0) for _ in range(3)]
    one = lambda v: torch.tensor([v], device=pos.device, dtype=pos.dtype)   # noqa: E731
    dq = math_utils.quat_from_euler_xyz(one(da[0]), one(da[1]), one(da[2]))
    ids = torch.tensor([env_id], device=pos.device)
    sensor.set_world_poses(pos[env_id:env_id + 1] + dp, math_utils.quat_mul(quat[env_id:env_id + 1], dq), env_ids=ids,
                           convention="world")


def reset_scene(env, env_ids: torch.Tensor, ranges: dict | None = None) -> None:
    """Event (mode="reset"), first in the list: store the result of the episode that just ended, then draw and
    lay out the next scene from ``ranges``. Seeds the random numbers of the reset from (seed, episode number),
    so the events after this one (friction, mass, motor gains) are repeatable too."""
    ranges = ranges or L.OFFICIAL_RANDOMIZATION
    st = state(env)
    origins = env.scene.env_origins
    ids = env_ids.tolist() if torch.is_tensor(env_ids) else list(range(env.num_envs))
    cube, tray, mat, top = env.scene["cube"], env.scene["tray"], env.scene["mat"], env.scene["surface"]
    dev = env.device
    first = True
    for eid in ids:
        _finalize(st, eid, env)
        episode = int(st["episode"][eid])
        rng = np.random.default_rng([st["seed"], eid, episode])
        if first:
            torch.manual_seed(int(rng.integers(2 ** 31)))       # for the Isaac Lab events that follow this one
        lights = _randomize_lights(rng, ranges, eid, shared=first)
        first = False
        lay = sample_layout(rng, ranges)
        if st["force_layout"]:
            lay.update({k: st["force_layout"][k] for k in LAYOUT_KEYS if k in st["force_layout"]})
        idx = torch.tensor([eid], device=dev)
        zero = torch.zeros(1, device=dev)
        upright = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=dev)
        # the look: the table top (a picture on the floor) or the bare floor, with or without the mat; walls; arm
        has_mat = lay["surface"] == "mat"
        table_z = L.SURFACE_Z + (L.MAT_THICK if has_mat else 0.0)
        pos = torch.tensor([[L.MAT_CENTER[0], L.MAT_CENTER[1],
                             L.SURFACE_Z + L.MAT_THICK / 2 if has_mat else L.SURFACE_PARKED_Z]], device=dev) + origins[eid]
        mat.write_root_pose_to_sim_index(root_pose=torch.cat([pos, upright], dim=-1), env_ids=idx)
        set_color(env, eid, "mat", lay["mat_rgb"])
        shown = lay["surface_scale"] is not None
        pos = torch.tensor([[0.0, 0.0, L.SURFACE_SHEET_Z if shown else L.SURFACE_PARKED_Z - 0.01]], device=dev) + origins[eid]
        top.write_root_pose_to_sim_index(root_pose=torch.cat([pos, upright], dim=-1), env_ids=idx)
        if shown:
            set_surface_look(env, eid, lay["surface_scale"], lay["surface_bias"])
        set_enclosure_color(env, eid, lay["wall_rgb"])
        set_arm_color(env, eid, lay["arm_rgb"])
        # the cube
        pos = torch.tensor([[lay["cube_xy"][0], lay["cube_xy"][1], table_z + L.CUBE_SIZE / 2 + L.CUBE_DROP]], device=dev) + origins[eid]
        quat = math_utils.quat_from_euler_xyz(zero, zero, torch.full((1,), math.radians(lay["cube_yaw_deg"]), device=dev))
        cube.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=idx)
        cube.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=dev), env_ids=idx)
        set_color(env, eid, "cube", lay["cube_rgb"])
        # the tray (a fixed body: it is moved here and nowhere else)
        pos = torch.tensor([[lay["tray_xy"][0], lay["tray_xy"][1], table_z]], device=dev) + origins[eid]
        quat = math_utils.quat_from_euler_xyz(zero, zero, torch.full((1,), math.radians(lay["tray_yaw_deg"]), device=dev))
        tray.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=idx)
        _jitter_camera(env, rng, eid, ranges)
        lay["lights"] = lights
        st["layout"][eid] = lay
        st["table_z"][eid] = table_z
        st["count"][eid] = 0
        st["success"][eid] = False
        st["success_t"][eid] = 0.0
        st["started"][eid] = True
        st["episode"][eid] = episode + 1


# -------------------------------------------------------------------------------------- success
def cube_on_tray_parts(env) -> dict:
    """What the success test looks at, per environment: where the cube is in the tray's own frame and how level
    it is. ``x``, ``y``: cube centre from the tray centre [m], along the tray's own sides; ``gap``: height of the
    cube's underside above the pocket's floor [m]; ``tilt_deg``: angle between the cube's most upright edge and
    the vertical."""
    cube, tray = env.scene["cube"].data, env.scene["tray"].data
    rel = math_utils.quat_apply_inverse(_t(tray.root_quat_w), _t(cube.root_pos_w) - _t(tray.root_pos_w))
    up = math_utils.matrix_from_quat(_t(cube.root_quat_w))[:, 2, :].abs().max(dim=-1).values
    return {"x": rel[:, 0], "y": rel[:, 1], "gap": rel[:, 2] - L.TRAY_FLOOR - L.CUBE_SIZE / 2,
            "tilt_deg": torch.rad2deg(torch.acos(up.clamp(-1.0, 1.0)))}


def cube_on_tray(env) -> torch.Tensor:
    """(num_envs,) bool: the cube lies flat on the floor of the tray's pocket, inside the walls. Its centre is
    inside the pocket, it is level (a cube that hangs on a wall is not) and its underside is at the height of the
    pocket's floor (a cube sitting on top of the walls is 14 mm above it). See layout.py."""
    p = cube_on_tray_parts(env)
    return ((p["x"].abs() < L.TRAY_POCKET[0] / 2) & (p["y"].abs() < L.TRAY_POCKET[1] / 2)
            & (p["tilt_deg"] < L.ON_TRAY_LEVEL_DEG) & (p["gap"].abs() < L.ON_TRAY_HEIGHT_TOL))


def cube_held(env) -> torch.Tensor:
    """(num_envs,) bool: the moving jaw presses on the cube with more than CONTACT_N."""
    held = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if "contact_grasp" in env.scene.keys():
        fm = env.scene["contact_grasp"].data.force_matrix_w
        if fm is not None:
            held = _t(fm).norm(dim=-1).reshape(env.num_envs, -1).sum(dim=1) > CONTACT_N
    return held


def cube_resting(env, rest_speed: float = REST_SPEED) -> torch.Tensor:
    return _t(env.scene["cube"].data.root_lin_vel_w).norm(dim=-1) < rest_speed


def success_now(env, rest_speed: float = REST_SPEED) -> torch.Tensor:
    """(num_envs,) bool: right now the cube lies on the tray, at rest, and the gripper has let go."""
    return cube_on_tray(env) & cube_resting(env, rest_speed) & ~cube_held(env)


def _update(env) -> torch.Tensor:
    st = state(env)
    now_ok = success_now(env).cpu()
    st["count"] = torch.where(now_ok, st["count"] + 1, torch.zeros_like(st["count"]))
    confirmed = st["count"] >= STABLE_STEPS
    t = _t(env.episode_length_buf).cpu().float() * env.step_dt
    new = confirmed & ~st["success"]
    st["success_t"] = torch.where(new, t - (STABLE_STEPS - 1) * env.step_dt, st["success_t"])
    st["success"] = st["success"] | confirmed
    return st["success"]


def episode_success(env) -> torch.Tensor:
    """Termination term of the official task: the success test has held for 0.5 s. The time kept for ties is
    the moment the cube came to rest on the tray."""
    return _update(env).to(env.device)


def track_success(env) -> torch.Tensor:
    """The same bookkeeping without ending the episode (kit environments: teleop and the generator decide
    themselves when an episode is over)."""
    _update(env)
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def debug_state(env, env_id: int = 0) -> dict:
    origin = env.scene.env_origins[env_id]
    st = state(env)
    cube = env.scene["cube"].data
    parts = cube_on_tray_parts(env)
    return {
        "cube": [round(v, 4) for v in (_t(cube.root_pos_w)[env_id] - origin).tolist()],
        "tray": [round(v, 4) for v in (_t(env.scene["tray"].data.root_pos_w)[env_id] - origin).tolist()],
        "from_tray_centre_mm": [round(1000 * float(parts[k][env_id]), 1) for k in ("x", "y")],
        "gap_mm": round(1000 * float(parts["gap"][env_id]), 2), "tilt_deg": round(float(parts["tilt_deg"][env_id]), 2),
        "speed": round(float(_t(cube.root_lin_vel_w)[env_id].norm()), 4),
        "on_tray": bool(cube_on_tray(env)[env_id]), "held": bool(cube_held(env)[env_id]),
        "held_ticks": int(st["count"][env_id]), "success": bool(st["success"][env_id]),
    }


def view_image(env, env_id: int = 0):
    """Latest third-person frame (uint8 RGB) from the view camera, or None if the scene has none."""
    if "view_cam" not in env.scene.keys():
        return None
    img = _t(env.scene["view_cam"].data.output["rgb"])[env_id]
    return img[..., :3].cpu().numpy().astype("uint8")


# ------------------------------------------------------------------------------------- Mimic helpers
def success_settling(env, rest_speed: float = 0.15) -> torch.Tensor:
    """Success for Isaac Lab Mimic generation: on the tray and released, the cube may still be settling (the
    export replay applies the strict test after a pause)."""
    return success_now(env, rest_speed=rest_speed)


def subtask_signals(env) -> dict:
    """Subtask end signals for Isaac Lab Mimic: grasp = the cube is held by the jaw and lifted 12 mm or more."""
    z = _t(env.scene["cube"].data.root_pos_w)[:, 2] - env.scene.env_origins[:, 2]
    on_surface = state(env)["table_z"].to(z.device) + L.CUBE_SIZE / 2          # cube centre while it stands on the surface
    return {"grasp": (cube_held(env) & (z > on_surface + 0.012)).float()}
