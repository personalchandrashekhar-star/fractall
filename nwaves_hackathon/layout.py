"""Geometry, units and the official randomisation of the NWaves SO-101 hackathon scene (Isaac Lab 3).

The task: pick up one cube and put it on one tray (a low open box).

Frame: the robot base is at the env origin and faces +x, +y is to the robot's left, z = 0 is the work
surface (the table top inside the enclosure). All lengths are in metres. The same numbers describe the real setup, so the
real cube, tray and cameras are placed where the simulation has them.
"""
import math

# ------------------------------------------------------------------------------------------------ the task
TASK = "pickplace"
INSTRUCTION = "Put the cube on the tray."
EPISODE_SECONDS = 30.0
CONTROL_HZ = 30.0
SUCCESS_HOLD_SECONDS = 0.5          # the cube must lie on the tray, released, for this long

# ------------------------------------------------------------------------------- the look of the workspace
# The real workspace (photo of 15 Nov 2026): a black arm on a light wooden table top, white walls around it, no mat.
# The simulated scene looks like that by default. Four things can change from episode to episode (the names are
# those of the randomisation, see OFFICIAL_RANDOMIZATION):
#   surface       what the cube and the tray lie on, one of
#                   wood       the light wooden table top
#                   wood_any   the wooden top with another brightness and tint (SURFACE_BRIGHTNESS, SURFACE_TINT)
#                   plain_any  a plain top without grain, a random light colour (PLAIN_SATURATION, PLAIN_VALUE)
#                   grey       the bare floor of the enclosure (it has the colour of the walls)
#                   mat        the dark mat on the bare floor of the enclosure
#                   a colour   a plain top in that colour (a name from COLORS or [r, g, b])
#   wall_colors   colour of the enclosure's own material (its walls; also its floor where that shows):
#                 a list of colours, or the word  any  (white to grey: WALL_VALUE, WALL_SATURATION);
#                 as_built  in a list = the grey sheet metal the enclosure model was built with
#   arm_colors    colour of the arm's printed parts: a list of colours, or the word  any  (between black and the
#                 yellow the model was built with, dark ones more often: ARM_YELLOW, ARM_ANY_POWER);
#                 as_built  in a list = that yellow
#   light         see light_intensity and light_temperature
# LOOKS names two complete looks. "real" is the default. "classic" is the scene of versions 0.1 and 0.2: the yellow
# arm, the grey enclosure, the dark mat. A settings file selects one with  look: classic.
SURFACE_Z = 0.0                                       # the enclosure's floor: where the cube and the tray lie
SURFACE_SHEET = ((-0.118, 0.578), (-0.398, 0.398))    # the table top covers the enclosure's floor: x and y extent [m]
SURFACE_SHEET_Z = 0.0002                              # it is a picture, 0.2 mm above the floor: nothing rests on it
SURFACE_PARKED_Z = -0.03                              # where the top and the mat are kept, out of sight, when not shown
WOOD_COLOR = (0.65, 0.43, 0.20)                       # mean colour of the wooden top (assets/wood_light.png)
SURFACE_BRIGHTNESS = (0.65, 1.15)                     # wood_any: scale on the wood's brightness
SURFACE_TINT = 0.08                                   # wood_any: each colour channel scaled by 1 +- this
PLAIN_SATURATION = (0.0, 0.35)
PLAIN_VALUE = (0.35, 0.90)
MAT_SIZE, MAT_THICK, MAT_CENTER = (0.31, 0.46), 0.002, (0.25, 0.0)
MAT_COLOR = (0.05, 0.05, 0.055)                       # the dark mat
ENCLOSURE_COLOR = (0.627, 0.627, 0.627)               # the enclosure's own material as built: "grey" in COLORS
WALL_VALUE = (0.55, 0.95)
WALL_SATURATION = (0.0, 0.10)
ARM_YELLOW = (1.0, 0.82, 0.12)                        # the printed parts of the arm as the model was built
ARM_BLACK = (0.03, 0.03, 0.035)                       # the real arm
ARM_ANY_POWER = 2.0                                   # any: black + t ** ARM_ANY_POWER * (yellow - black), t drawn evenly in 0..1
LOOKS = {
    "real": {"arm_colors": ["black"], "wall_colors": ["white"], "surface": ["wood"]},
    "classic": {"arm_colors": ["as_built"], "wall_colors": ["as_built"], "surface": ["mat"]},
}
SURFACE_WORDS = ("wood", "wood_any", "plain_any", "grey", "mat")
TABLE_Z = SURFACE_Z                                   # where the cube and the tray lie (on the mat: MAT_THICK higher)

# ------------------------------------------------------------------------------------------------ the cube
CUBE_SIZE = 0.025                                     # the real cube: 25.00 mm (cube_25mm.stl)
# Mass [kg]. A 25 mm cube printed solid in PLA would weigh 19.4 g; real prints are mostly air inside and weigh
# 6 to 12 g. The nominal value is a printed cube with thick walls; the official randomisation scales it over
# 6 to 20 g (cube_mass_scale below), so every real print is covered.
CUBE_MASS = 0.012
CUBE_DROP = 0.001                                     # a new scene sets the cube down from 1 mm above the surface
CUBE_Z = TABLE_Z + CUBE_SIZE / 2 + CUBE_DROP          # centre height of the cube at the start
COLORS = {                                            # names you can use in a settings file
    "red": (0.78, 0.07, 0.07),
    "blue": (0.06, 0.18, 0.80),
    "green": (0.08, 0.60, 0.18),
    "yellow": (0.92, 0.76, 0.05),
    "orange": (0.90, 0.38, 0.04),
    "purple": (0.45, 0.12, 0.70),
    "pink": (0.90, 0.35, 0.60),
    "cyan": (0.05, 0.65, 0.75),
    "white": (0.85, 0.85, 0.82),
    "black": (0.02, 0.02, 0.022),                     # the same black as the tray
    "grey": (0.627, 0.627, 0.627),                    # the enclosure as it was built
}
# The cube has no fixed colour: the scorer draws one for every episode. Instead of a list of colours, a settings
# file (and OFFICIAL_CUBE_COLORS below) may say one of three words:
#   "bright"  any bright colour, the real cube's orange included. In REAL_CUBE_SHARE of the episodes the cube has
#             the colour of the real cube (REAL_CUBE_COLOR, each channel moved by up to REAL_CUBE_JITTER). In the
#             others: any hue, any saturation (grey and white included), never dark: brightness (the V of HSV) in
#             BRIGHT_COLOR_VALUE and luminance of at least BRIGHT_COLOR_MIN_LUMINANCE.
#   "any"     any colour, black and dark ones included: a dark cube (brightness in DARK_COLOR_VALUE, any hue) in
#             ANY_COLOR_DARK_SHARE of the episodes, a "bright" draw in the others.
#   "black"   always the black of COLORS["black"].
REAL_CUBE_COLOR = "orange"                            # the cube used on the real arm (a name from COLORS)
REAL_CUBE_SHARE = 0.3
REAL_CUBE_JITTER = 0.06
BRIGHT_COLOR_VALUE = (0.55, 1.0)
BRIGHT_COLOR_MIN_LUMINANCE = 0.10
DARK_COLOR_VALUE = (0.0, 0.12)
ANY_COLOR_DARK_SHARE = 0.3
# What the official evaluation uses: "bright", "any" or "black" (the one place to change it).
OFFICIAL_CUBE_COLORS = "bright"

# ------------------------------------------------------------------------------------------------ the tray
# The target: a printed tray, a low open box. The cube is put on its floor, inside the walls.
# Size: from the print file (cube_tray_80mm.3mf). The model in that file is a plate of 80 x 80 x 2 mm with a rim;
# the print project scales it by 0.9472 (x), 0.8968 (y) and 14.4374 (z), and that is what gets printed:
#   outside 75.8 x 71.7 mm, 28.9 mm high; the floor is 14.4 mm thick, so the cube stands 14.4 mm above the table;
#   the walls rise 14.4 mm above the floor and are 1.9 mm (across x) and 1.8 mm (across y) thick;
#   the pocket inside the walls is 72.0 x 68.1 mm and 14.4 mm deep. A 25 mm cube in it stands 10.6 mm above the walls.
# Change the numbers below and the simulated tray (tray.py), the success test, the generator's place motion and the
# cube-to-tray spacing all follow.
TRAY_SIZE = (0.07578, 0.07174)                        # outside, along the tray's own x and y
TRAY_HEIGHT = 0.02887                                 # in all
TRAY_FLOOR = 0.01444                                  # thickness of the floor = height of the pocket's floor above the table
TRAY_WALL = (0.00189, 0.00179)                        # thickness of the walls: the two across x, the two across y
TRAY_CORNER_RADIUS = (0.00568, 0.00538)               # of the outline's corners (the model's 6 mm, scaled like the rest)
TRAY_COLOR = (0.02, 0.02, 0.022)                      # black (the one place to change the tray's colour)

TRAY_WALL_HEIGHT = TRAY_HEIGHT - TRAY_FLOOR           # the walls above the pocket's floor
TRAY_POCKET = (TRAY_SIZE[0] - 2 * TRAY_WALL[0], TRAY_SIZE[1] - 2 * TRAY_WALL[1])     # inside the walls
TRAY_FLOOR_Z = TABLE_Z + TRAY_FLOOR                   # height of the surface the cube is put on
# Success: the cube lies flat on the pocket's floor, inside the walls. Measured as three things, all needed:
#   the cube's centre is inside the pocket;
#   the cube is level (lying flat it is level to 0.05 degrees; a cube that hangs on a wall is tilted by tens of degrees);
#   its underside is at the height of the pocket's floor (a cube that sits on top of the walls in a corner is level
#   too, but 14.4 mm higher).
# The walls are higher than half the cube, so a cube that passes this test is inside the walls with all of itself:
# its centre is within 23.5 mm (along x) and 21.6 mm (along y) of the tray's centre, less when it is turned
# (18.3 and 16.4 mm at 45 degrees).
ON_TRAY_LEVEL_DEG = 2.0                               # the cube counts as level below this tilt
ON_TRAY_HEIGHT_TOL = 0.0015                           # m: how far the cube's underside may be from the pocket's floor

# ------------------------------------------------------------------------------------- where things can be
# What the arm can serve, measured with the IK (scripts/reach.py), as distances from the scene origin (x = y = 0; the
# arm's own turning axis stands 3 cm from it, see PAN_AXIS below).
# REACH_NOTE
# The scene camera sits behind the arm's right shoulder: at the home pose the arm hides what stands to the left of
# the line y = 0.75 (x + 0.305) - 0.28. So the cube lives on the right, the tray in the middle, both in sight.
CUBE_REACH = (0.15, 0.27)          # m from the scene origin: where a cube can be picked up
TRAY_REACH = (0.15, 0.27)          # m from the scene origin: where the tray's centre can be
# The cube never starts in the tray or against it, and the gripper needs room to come down beside the tray: its
# open jaws reach about 4 cm from the cube's centre and go lower than the tray's walls are high.
# Measured (the generator's own grasp, the tray at 60 to 110 mm from the cube in 8 directions): the gripper touches
# the tray at 60 mm in 5 directions, at 70 mm in 2, at 80 and 90 mm in 1 (a graze of 0.4 N at 90 mm), from 100 mm
# on in none. Scenes are drawn with 105 mm or more between the two centres.
TRAY_OUTER_REACH = (((TRAY_SIZE[0] / 2 - TRAY_CORNER_RADIUS[0]) ** 2 + (TRAY_SIZE[1] / 2 - TRAY_CORNER_RADIUS[1]) ** 2) ** 0.5
                    + max(TRAY_CORNER_RADIUS))                              # centre to the corner's arc: 50 mm
GRIPPER_ROOM = 0.055               # m beyond the tray's corner that the gripper needs clear around the cube
MIN_CUBE_TRAY = TRAY_OUTER_REACH + GRIPPER_ROOM     # smallest centre-to-centre distance at the start

# The official randomisation: what the scorer (nwaves-eval, the jury) draws for every episode. The ranges are
# public, the jury's seed is not. The generator's own ranges live in configs/datagen.yaml with the same names.
OFFICIAL_RANDOMIZATION = {
    "cube_x": (0.14, 0.21),                 # cube centre [m]
    "cube_y": (-0.16, -0.08),
    "cube_yaw_deg": (-45.0, 45.0),          # a cube looks the same every 90 deg
    "cube_colors": OFFICIAL_CUBE_COLORS,    # "bright", "any", "black", or a list of names from COLORS and [r, g, b]
    "cube_color_jitter": 0.0,               # +- per channel around a listed colour (not used with the three words)
    "tray_x": (0.20, 0.25),                 # tray centre [m]
    "tray_y": (-0.01, 0.05),
    "tray_yaw_deg": (-10.0, 10.0),          # the tray is laid down by hand: never exactly square to the arm
    # the look of the workspace, around the real one (see "the look of the workspace" above)
    "surface": ["wood", "wood_any", "wood_any", "wood_any", "plain_any", "plain_any", "grey", "mat"],
    "wall_colors": "any",
    "arm_colors": "any",
    "mat_colors": [MAT_COLOR],              # colour of the mat in the episodes that have it
    "light_intensity": (0.7, 1.3),          # scale on the brightness of the roof strips, the dome and the key light
    "light_temperature": (3600.0, 6000.0),  # colour temperature of the roof strips [K] (4800 = nominal, neutral white)
    "camera_pos_jitter": 0.02,              # scene camera: +- metres on each axis around its mount
    "camera_rot_jitter_deg": 2.0,           # scene camera: +- degrees about each axis
    "cube_friction": (0.6, 1.2),            # static friction of the cube (dynamic = 0.85 x static)
    "cube_mass_scale": (0.5, 1.7),          # scale on CUBE_MASS: 6 to 20 g, an airy print to a solid one
    "motor_stiffness_scale": (0.8, 1.2),    # scale on every servo's position gain
    "motor_damping_scale": (0.8, 1.25),     # scale on every servo's damping
}
# Nothing varies: the orange cube and the tray in the middle of their official regions, the real look (black arm,
# wooden top, white walls), nominal light and physics.
FIXED_RANDOMIZATION = {
    "cube_x": (0.175, 0.175), "cube_y": (-0.12, -0.12), "cube_yaw_deg": (0.0, 0.0),
    "cube_colors": [REAL_CUBE_COLOR], "cube_color_jitter": 0.0,
    "tray_x": (0.225, 0.225), "tray_y": (0.02, 0.02), "tray_yaw_deg": (0.0, 0.0),
    "surface": list(LOOKS["real"]["surface"]), "wall_colors": list(LOOKS["real"]["wall_colors"]),
    "arm_colors": list(LOOKS["real"]["arm_colors"]), "mat_colors": [MAT_COLOR],
    "light_intensity": (1.0, 1.0), "light_temperature": (4800.0, 4800.0),
    "camera_pos_jitter": 0.0, "camera_rot_jitter_deg": 0.0,
    "cube_friction": (1.0, 1.0), "cube_mass_scale": (1.0, 1.0),
    "motor_stiffness_scale": (1.0, 1.0), "motor_damping_scale": (1.0, 1.0),
}
PRACTICE_SEED = 2026                # public seed of nwaves-eval; the jury uses a hidden one
PRACTICE_EPISODES, JURY_EPISODES = 10, 10

# ------------------------------------------------------------------------------------------- the enclosure
# Mount in the NWaves workspace enclosure: in the enclosure's own frame the arm base
# sits at (0, 0.455) on the work surface turned -90 deg, just inside the opening and working toward the back wall.
# Seen from the robot (x forward, y left, z up from the work surface) the enclosure origin is therefore at
# (0.455, 0, -0.49846) turned +90 deg. Enclosure interior in this frame: x -0.12 (opening) .. 0.58, y -0.40 .. 0.40.
ENCLOSURE_POS = (0.455, 0.0, -0.49846)
ENCLOSURE_ROT_XYZW = (0.0, 0.0, 0.70710678, 0.70710678)

# Fixed scene camera (behind the arm's right shoulder, outside the opening),
# 20 mm on a 4:3 sensor (about 55 deg horizontal). The real camera is mounted here.
SCENE_EYE = (-0.305, -0.28, 0.52)
SCENE_TARGET = (0.175, 0.0, 0.0)
SCENE_FOCAL_MM, SCENE_H_APERTURE_MM, SCENE_V_APERTURE_MM = 20.0, 20.955, 15.71625
IMAGE_W, IMAGE_H = 640, 480

# The table under the enclosure (enclosure frame: top 1.15 x 0.95 m centred at (0, 0.22), legs at x +-0.49,
# y -0.17 / 0.61, floor 0.44 m below the work surface), converted to the robot frame
TABLE_CENTER, TABLE_SIZE, TABLE_TOP_Z, FLOOR_Z = (0.235, 0.0), (0.95, 1.15), -0.003, -0.44
TABLE_LEGS = ((0.625, -0.49), (-0.155, -0.49), (0.625, 0.49), (-0.155, 0.49))
_d = [t - e for t, e in zip(SCENE_TARGET, SCENE_EYE)]
_n = sum(v * v for v in _d) ** 0.5
# The camera's housing (a small dark box, for looks only) sits 8 cm behind the lens: its front face is 5.5 cm
# behind, so a camera moved by the jitter below never ends up inside it (a camera inside the box sees black).
SCENE_CAM_BODY_POS = tuple(e - 0.08 * v / _n for e, v in zip(SCENE_EYE, _d))
MAX_CAMERA_POS_JITTER = 0.03       # m per axis: the largest scene-camera jitter a settings file may ask for

# Lights: two neutral white strips under the enclosure's roof, a dome and a soft key light outside
ROOF_STRIPS = ((0.255, 0.20), (0.255, -0.20))
ROOF_STRIP_Z, ROOF_STRIP_LEN, ROOF_STRIP_W = 0.618, 0.44, 0.035
DOME_INTENSITY, KEY_INTENSITY, STRIP_INTENSITY = 800.0, 650.0, 950.0
STRIP_TEMPERATURE, OUTSIDE_TEMPERATURE = 4800.0, 6500.0     # K: the roof strips (neutral white); the dome and key light
STRIP_COLOR = (1.0, 0.96, 0.90)

# Third-person view for the Isaac Lab window and the documentation renders (never an observation)
VIEW_EYE, VIEW_TARGET, VIEW_W, VIEW_H = (-0.62, -0.46, 0.60), (0.22, 0.0, 0.08), 1024, 576

# ---------------------------------------------------------------------------------------------- the robot
# The robot model is NVIDIA's SO-101 of the Isaac Sim 6.2 asset pack. The lab holds a copy (ROBOT_ASSET_DIRS, first
# one found); where there is none, Isaac Lab fetches the model from NVIDIA's asset server, which needs the internet.
ROBOT_ASSET_FILE = "so101_new_calib_SysID/so101_new_calib.usda"
ROBOT_ASSET_DIRS = ("/opt/nwaves/assets",)            # NWAVES_ASSETS in the environment names another folder
# Joint units of the policy interface (LeRobot SO-101 follower with use_degrees=True):
# shoulder_pan .. wrist_roll in degrees (0 = the LeRobot calibration middle pose = the model's zero),
# gripper 0..100 over its range (0 closed, 100 open).
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
GRIPPER_RANGE_RAD = (-0.174533, 1.74533)  # so101_new_calib gripper limits (-10 .. 100 deg)
# Home pose = the original SO-101 home: the five arm joints at 0 deg, the LeRobot calibration middle pose.
# Every episode starts there, in simulation and on the real arm.
HOME_RAD = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0, "wrist_flex": 0.0, "wrist_roll": 0.0}
HOME_GRIPPER = 22.4                       # gripper at the start of an episode, on the 0..100 scale (as on the real arm)
# How fast joint targets may move, the same as the real arm's runner allows. A target (arm joints) never changes by
# more than MAX_STEP_DEG from one 30 Hz tick to the next and is never more than MAX_LEAD_DEG away from where the
# joint was at the start of the tick. The generator and the teleop are held to this, so the data never asks the
# real arm for more; the official task applies the same limit to what a policy commands.
MAX_STEP_DEG = 4.0
MAX_LEAD_DEG = 10.0
GRIPPER_EFFORT_NM = 1.5                   # STS3215 at 7.4 V: ~1.9 N m stall, ~1.5 N m usable
# Measured on 15 Nov 2026: this limit is not what holds the cube. The robot asset has no motor inertia on its joints
# in PhysX, and the drive of the 12 g jaw then presses with about 0.1 N m (1.5 N at the jaw) whatever the limit is.
# That grip carries the cube with a margin of about 10 and the scripted grasp is tuned to it; with the identified
# motor inertia on the gripper the jaw would press with the full limit (21 N) and grasps get less reliable.
# The arm turns about a vertical axis that does not pass through the scene origin: it stands here (x, y) [m].
PAN_AXIS = (0.0231, 0.0208)
# Fingertip point ("gripper_frame_link" of the LeRobot URDF) in the gripper link frame; the jaws close along
# the link's x axis and approach along -z, so "pointing down" = link z up.
TCP_OFFSET = (-0.0079, -0.000218, -0.0981274)
JAW_OPEN_RAD, JAW_CLOSED_RAD = 0.55, -0.05   # ~44 mm between the tips when open: clears the 25 mm cube


def rad_to_units(q):
    """Model joint positions [rad] (6) -> interface units: 5 x degrees, gripper 0..100."""
    lo, hi = GRIPPER_RANGE_RAD
    return [math.degrees(v) for v in q[:5]] + [100.0 * (q[5] - lo) / (hi - lo)]


def units_to_rad(u):
    lo, hi = GRIPPER_RANGE_RAD
    return [math.radians(v) for v in u[:5]] + [lo + (hi - lo) * min(max(u[5], 0.0), 100.0) / 100.0]


HOME_GRIPPER_RAD = GRIPPER_RANGE_RAD[0] + (GRIPPER_RANGE_RAD[1] - GRIPPER_RANGE_RAD[0]) * HOME_GRIPPER / 100.0


def local_robot_usd():
    """The copy of the robot model that the lab holds, or None when there is none (the server is used then)."""
    import os
    for folder in (os.environ.get("NWAVES_ASSETS"),) + ROBOT_ASSET_DIRS:
        if folder and os.path.isfile(os.path.join(folder, ROBOT_ASSET_FILE)):
            return os.path.join(folder, ROBOT_ASSET_FILE)
    return None
