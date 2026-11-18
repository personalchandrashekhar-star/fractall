"""Run an Isaac Lab Mimic script (annotate_demos.py, generate_dataset.py) with the hackathon environments
registered, plus any of your own modules (--import my_module) that register more environments.

Run it from your clone (nwaves-mimic annotate / generate are the short forms):

    isaaclab -p scripts/mimic_run.py annotate --task NWaves-SO101-PickPlace-Kit-Mimic-v0 \\
        --input_file ~/workspace/mimic/src.hdf5 --output_file ~/workspace/mimic/annotated.hdf5 --auto
    isaaclab -p scripts/mimic_run.py generate --task NWaves-SO101-PickPlace-Kit-Mimic-v0 \\
        --input_file ~/workspace/mimic/annotated.hdf5 --output_file ~/workspace/mimic/generated.hdf5 \\
        --generation_num_trials 200 --num_envs 4

Everything after annotate / generate goes to Isaac Lab's own script. The Isaac Sim window opens by default;
--headless runs without it.

The order: record a few demonstrations (scripts/teleop.py --hdf5 FILE, or scripts/generate.py --hdf5 FILE), annotate
them, generate, then scripts/mimic_export.py. Each step reads the file the step before wrote; when that file is not
there, the script says which step comes first.
"""
import importlib
import os
import runpy
import sys

import _start  # noqa: F401  (first: this clone's nwaves_hackathon goes to the front of Python's path)

SCRIPTS = os.path.join(os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab"), "scripts/imitation_learning/isaaclab_mimic")
NAMES = {"annotate": "annotate_demos.py", "generate": "generate_dataset.py"}

USAGE = (f"usage: isaaclab -p scripts/mimic_run.py annotate|generate --task {_start.MIMIC_TASK} --input_file FILE "
         f"--output_file FILE [--auto] [--generation_num_trials N] [--num_envs K] [--import module ...] [--headless]")
args = sys.argv[1:]
if not args or args[0] not in NAMES:
    _start.stop("mimic", USAGE)
for needed in ("--task", "--input_file", "--output_file"):
    if _start._value(args, needed) is None and not any(v in ("-h", "--help") for v in args):
        _start.stop("mimic", f"{needed} is missing. {USAGE}")
if _start._value(args, "--input_file"):
    _start.checked("mimic", _start.need_demos, _start._value(args, "--input_file"), args[0])
    os.makedirs(os.path.dirname(_start.full_path(_start._value(args, "--output_file"))), exist_ok=True)
script = os.path.join(SCRIPTS, NAMES[args[0]])
rest, imports, i = [], [], 1
while i < len(args):
    if args[i] == "--import":
        imports.append(args[i + 1]); i += 2
    elif args[i] in ("--input_file", "--output_file") and i + 1 < len(args):
        rest += [args[i], os.path.abspath(os.path.expanduser(args[i + 1]))]; i += 2
    else:
        rest.append(args[i]); i += 1
rest = _start.window_argv(rest)
print(f"[mimic] code: {_start.REPO}", flush=True)
user_dir = os.environ.get("NWAVES_USER_DIR") or os.environ.get("WORKSPACE")
if user_dir and user_dir not in sys.path:
    sys.path.append(user_dir)
import nwaves_hackathon  # noqa: E402,F401  (registers the gym ids)
for m in imports:
    importlib.import_module(m)
sys.argv = [script] + rest
runpy.run_path(script, run_name="__main__")
