"""Randomisation ranges: reading them from configs/datagen.yaml and putting them into an environment config.

One dict of ranges describes how a scene is drawn at every reset. The names are the same everywhere:

    layout.OFFICIAL_RANDOMIZATION      what the scorer uses (public)
    configs/datagen.yaml               what the generator, the teleop and Mimic use (yours to change)

    cube_x, cube_y            [lo, hi] m      cube centre
    cube_yaw_deg              [lo, hi] deg    cube rotation about the vertical
    cube_colors               word | list     bright (any bright colour), any (any colour, black and dark ones
                                              included) or black; or a list of names from layout.COLORS and
                                              [r, g, b] values
    cube_color_jitter         number          +- per colour channel around a listed colour
    tray_x, tray_y            [lo, hi] m      tray centre
    tray_yaw_deg              [lo, hi] deg    tray rotation about the vertical
    look                      word            real (black arm, wooden table top, white walls: the default) or
                                              classic (yellow arm, grey enclosure, dark mat); sets the next three
                                              entries where the file does not give them itself
    surface                   list            what the cube and the tray lie on, one per episode: wood, wood_any,
                                              plain_any, grey, mat, or a colour (see layout.py)
    wall_colors               word | list     colour of the walls: any (white to grey) or a list of colours; as_built =
                                              the grey of the enclosure model
    arm_colors                word | list     colour of the arm's printed parts: any (between black and the model's
                                              yellow) or a list of colours; as_built = that yellow
    mat_colors                list            colours of the mat in the episodes that have it
    light_intensity           [lo, hi]        scale on the brightness of the roof strips, the dome and the key light
    light_temperature         [lo, hi] K      colour temperature of the roof strips (4000 = nominal)
    camera_pos_jitter         number m        scene camera, +- on each axis
    camera_rot_jitter_deg     number deg      scene camera, +- about each axis
    cube_friction             [lo, hi]        static friction of the cube
    cube_mass_scale           [lo, hi]        scale on the cube's mass
    motor_stiffness_scale     [lo, hi]        scale on the servos' position gain
    motor_damping_scale       [lo, hi]        scale on the servos' damping
"""
from __future__ import annotations

import os

from . import layout as L

PAIRS = ("cube_x", "cube_y", "cube_yaw_deg", "tray_x", "tray_y", "tray_yaw_deg", "light_intensity", "light_temperature",
         "cube_friction", "cube_mass_scale", "motor_stiffness_scale", "motor_damping_scale")
NUMBERS = ("cube_color_jitter", "camera_pos_jitter", "camera_rot_jitter_deg")
COLOR_WORDS = ("bright", "any", "black")
RENAMED = {"cup_x": "tray_x", "cup_y": "tray_y"}          # names of the task before 0.3.0 (the target was a cup)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_config_path() -> str:
    """configs/datagen.yaml of this repository, or the file named by NWAVES_DATAGEN_CONFIG."""
    return os.environ.get("NWAVES_DATAGEN_CONFIG") or os.path.join(REPO, "configs", "datagen.yaml")


def find_config(name: str) -> str:
    """A settings file as typed (relative to where you are), or inside this repository: configs/my.yaml, my.yaml,
    or just the name ("my" finds configs/my.yaml or configs/datagen_my.yaml). Last, the lab's built-in configs."""
    name = os.path.expanduser(name)
    built_in = os.path.join("/opt/nwaves/hackathon", "configs")
    for candidate in (name, os.path.join(REPO, name), os.path.join(REPO, "configs", name),
                      os.path.join(REPO, "configs", name + ".yaml"), os.path.join(REPO, "configs", f"datagen_{name}.yaml"),
                      os.path.join(built_in, name + ".yaml"), os.path.join(built_in, f"datagen_{name}.yaml")):
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(f"no settings file '{name}' (looked here, in {REPO} and in {REPO}/configs)")


def load_config(path: str | None = None) -> dict:
    """Read a generator config file. Returns {"randomization": {...}, "behavior": {...}, ...}."""
    import yaml
    path = find_config(path) if path else default_config_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no generator config at {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    cfg["randomization"] = check(cfg.get("randomization") or {}, where=path)
    cfg.setdefault("behavior", {})
    cfg["_path"] = path
    return cfg


def check(ranges: dict, where: str = "randomization") -> dict:
    """Validate a dict of ranges and fill what is missing with "does not vary" (layout.FIXED_RANDOMIZATION)."""
    old = sorted(set(ranges) & set(RENAMED))
    if old:
        raise ValueError(f"{where}: " + ", ".join(f"{k} is now {RENAMED[k]}" for k in old)
                         + " (the target is a tray since version 0.3.0): rename the entries in this file")
    ranges = dict(ranges)
    look = ranges.pop("look", None)
    if look is not None and look not in L.LOOKS:
        raise ValueError(f"{where}: look is '{look}'; the looks are {', '.join(L.LOOKS)}")
    unknown = sorted(set(ranges) - set(L.OFFICIAL_RANDOMIZATION))
    if unknown:
        raise ValueError(f"{where}: unknown randomisation name(s) {unknown}; known: "
                         f"{sorted(set(L.OFFICIAL_RANDOMIZATION) | {'look'})}")
    out = dict(L.FIXED_RANDOMIZATION)
    out.update(L.LOOKS[look] if look else {})
    out.update(ranges)
    for key in PAIRS:
        value = out[key]
        if isinstance(value, (int, float)):
            value = [value, value]                      # one number = no variation
        if not (isinstance(value, (list, tuple)) and len(value) == 2):
            raise ValueError(f"{where}: {key} must be [low, high], got {value!r}")
        lo, hi = float(value[0]), float(value[1])
        if lo > hi:
            raise ValueError(f"{where}: {key} has low > high ({lo} > {hi})")
        out[key] = (lo, hi)
    for key in NUMBERS:
        out[key] = float(out[key])
        if out[key] < 0:
            raise ValueError(f"{where}: {key} must be 0 or more")
    out["mat_colors"] = _colors(out.get("mat_colors") or [L.MAT_COLOR], "mat_colors", where)
    out["surface"] = _colors(out.get("surface") or L.LOOKS["real"]["surface"], "surface", where, also=L.SURFACE_WORDS)
    for name, extra in (("wall_colors", ("as_built",)), ("arm_colors", ("as_built",))):
        value = out.get(name) or L.LOOKS["real"][name]
        out[name] = value if value == "any" else _colors(value, name, where, also=extra, words=("any",))
    if out["camera_pos_jitter"] > L.MAX_CAMERA_POS_JITTER:
        raise ValueError(f"{where}: camera_pos_jitter is in metres and at most {L.MAX_CAMERA_POS_JITTER} "
                         f"(further out the camera would sit inside its own housing and see black)")
    for key in ("cube_friction", "cube_mass_scale", "motor_stiffness_scale", "motor_damping_scale", "light_intensity"):
        if out[key][0] <= 0:
            raise ValueError(f"{where}: {key} must be above 0")
    colors = out.get("cube_colors") or "any"
    if not (isinstance(colors, str) and colors in COLOR_WORDS):
        colors = _colors(colors, "cube_colors", where, words=COLOR_WORDS)
    out["cube_colors"] = colors
    return out


def _colors(value, name: str, where: str, also: tuple = (), words: tuple = ()) -> list:
    """A list of colours: names from layout.COLORS (and the extra names in ``also``) and [r, g, b] values. A single
    colour given without the list is accepted."""
    if isinstance(value, str) or (len(value) == 3 and all(isinstance(c, (int, float)) for c in value)):
        value = [value]
    for c in value:
        if isinstance(c, str) and c not in L.COLORS and c not in also:
            hint = f"one of the words {', '.join(words)}, " if words else ""
            raise ValueError(f"{where}: {name}: unknown colour '{c}'; use {hint}a name "
                             f"({', '.join(tuple(L.COLORS) + tuple(also))}) or [r, g, b]")
        if not isinstance(c, str) and not (len(c) == 3 and all(isinstance(v, (int, float)) for v in c)):
            raise ValueError(f"{where}: {name}: a colour is a name or [r, g, b], got {c!r}")
    return list(value)


def reach_warnings(ranges: dict) -> list[str]:
    """Plain sentences about ranges that go beyond what the arm can serve or the scene camera can see
    (see layout.py)."""
    import math
    notes = []
    for name, limit in (("cube", L.CUBE_REACH), ("tray", L.TRAY_REACH)):
        xs, ys = ranges[f"{name}_x"], ranges[f"{name}_y"]
        corners = [(x, y) for x in xs for y in ys]
        dist = [math.hypot(x, y) for x, y in corners]
        if min(dist) < limit[0] - 1e-9 or max(dist) > limit[1] + 1e-9:
            notes.append(f"the {name} range reaches {min(dist):.2f} .. {max(dist):.2f} m from the arm's base; the arm "
                         f"serves {limit[0]} .. {limit[1]} m well: expect failed episodes outside")
        margin = 0.02 if name == "cube" else 0.0                           # the cube whole, the tray more than half
        if any(y + margin > 0.75 * (x + 0.305) - 0.28 for x, y in corners):
            notes.append(f"in part of the {name} range the arm hides the {name} from the scene camera at the start")
    return notes


def apply(cfg, ranges: dict):
    """Put a dict of ranges into an environment config (before gym.make). Returns the config."""
    ranges = check(ranges)
    cfg.randomization = ranges
    ev = cfg.events
    ev.layout.params["ranges"] = ranges
    lo, hi = ranges["cube_friction"]
    ev.cube_friction.params["static_friction_range"] = (lo, hi)
    ev.cube_friction.params["dynamic_friction_range"] = (0.85 * lo, 0.85 * hi)
    ev.cube_mass.params["mass_distribution_params"] = ranges["cube_mass_scale"]
    ev.motor_gains.params["stiffness_distribution_params"] = ranges["motor_stiffness_scale"]
    ev.motor_gains.params["damping_distribution_params"] = ranges["motor_damping_scale"]
    return cfg
