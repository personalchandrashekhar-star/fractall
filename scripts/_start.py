"""Start-up shared by the scripts of this folder.

Every script here is a normal Isaac Lab script. Run it from your clone with the lab's Isaac Lab launcher:

    cd ~/workspace/so101-sim2real-hackathon-blr
    isaaclab -p scripts/view.py

Importing this module puts this clone first on Python's path, so ``import nwaves_hackathon`` is the code next to
these scripts (yours, with your changes) and not a copy installed somewhere else.

``window_args`` and ``window`` handle the Isaac Sim window: it opens by default on the lab's virtual display, which
the simulation panel shows; ``--headless`` runs without it (faster); ``--viz ...`` is Isaac Lab's own option and
is respected when given.

``stop`` and the ``need_...`` functions end a script with one line that says what to do first when a file or a
folder it was asked to read is not there, before Isaac Sim starts. ``python3 scripts/_start.py --check SCRIPT ARGS``
runs the same checks without starting anything (the lab's isaaclab command does that first).
"""
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[:1] != [REPO]:
    sys.path.insert(0, REPO)

LAB_DISPLAY = ":99"                                   # the lab's virtual display (what the simulation panel shows)
OFFICIAL_COPY = "/opt/nwaves/hackathon"               # the copy built into the lab; the official scorer runs from there


def is_official_copy() -> bool:
    return os.path.realpath(REPO) == os.path.realpath(OFFICIAL_COPY)


def window_args(parser):
    """Add --headless to a script's arguments (next to Isaac Lab's own --viz)."""
    parser.add_argument("--headless", action="store_true", help="no Isaac Sim window (faster)")


def _lab_display():
    """Use the lab's virtual display when the shell has none set."""
    if not os.environ.get("DISPLAY") and os.path.exists("/tmp/.X11-unix/X" + LAB_DISPLAY.lstrip(":")):
        os.environ["DISPLAY"] = LAB_DISPLAY
    return os.environ.get("DISPLAY")


def fit_window():
    """Stretch the Isaac Sim window over the whole virtual display as soon as it appears (the simulation panel
    shows the whole display). Runs beside the script; does nothing where xdotool is missing."""
    if not shutil.which("xdotool"):
        return
    loop = ('for i in $(seq 1 1200); do wid=$(xdotool search --onlyvisible --name "Isaac" 2>/dev/null | head -1); '
            'if [ -n "$wid" ]; then size=$(xdotool getdisplaygeometry 2>/dev/null); size=${size:-1920 1080}; '
            'for d in 0.3 1 2 4 8 15; do sleep $d; '
            'xdotool windowmove "$wid" 0 0 windowsize "$wid" $size >/dev/null 2>&1 || exit 0; done; exit 0; fi; '
            'sleep 0.5; done')
    subprocess.Popen(["bash", "-c", loop], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def window(args, tag: str) -> bool:
    """Call right after parse_args(). Decides whether the Isaac Sim window opens and says which code runs."""
    display = _lab_display()
    asked = getattr(args, "visualizer", None)
    if getattr(args, "headless", False):
        args.visualizer = None
    elif asked is None and display:
        args.visualizer = "kit"                       # the default: show the window
    shown = "kit" in str(args.visualizer or "")
    if shown and display:
        fit_window()
    print(f"[{tag}] code: {REPO}", flush=True)
    print(f"[{tag}] Isaac Sim window: " + (f"on (display {display})" if shown else "off"), flush=True)
    return shown


# ---------------------------------------------------------------------------------- what to do first
MIMIC_TASK = "NWaves-SO101-PickPlace-Kit-Mimic-v0"
CHECK_FAILED = 64                                     # exit code of --check when something is missing


class Missing(Exception):
    """Something a script was asked to read is not there. The text says what to do first."""


def stop(tag: str, message: str, code: int = 2):
    """End the script with one line."""
    print(f"[{tag}] {message}", flush=True)
    sys.exit(code)


def full_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def workspace() -> str:
    return os.environ.get("WORKSPACE") or os.path.expanduser("~/workspace")


def dataset_folder(name: str) -> str:
    """A dataset given by name is a folder under ~/workspace/datasets; a folder that exists is taken as it is."""
    folder = os.path.expanduser(name)
    if os.path.isdir(folder) or os.path.isabs(folder):
        return os.path.abspath(folder)
    return os.path.join(workspace(), "datasets", name)


def _demo_count(path: str):
    """Number of demonstrations in an Isaac Lab HDF5 file, or None when that cannot be read here."""
    try:
        import h5py
    except ImportError:
        return None
    try:
        with h5py.File(path, "r") as f:
            return len(f["data"]) if "data" in f else 0
    except OSError:
        return 0


def need_demos(path: str, step: str):
    """The input of a Mimic step: recorded demonstrations (annotate), annotated ones (generate), generated ones
    (export)."""
    path = full_path(path)
    todo = {
        "annotate": (f"Record the demonstrations first:  isaaclab -p scripts/teleop.py --hdf5 {path}   "
                     f"(Enter starts an episode, Enter again saves it; or let the generator make them:  "
                     f"isaaclab -p scripts/generate.py --episodes 10 --hdf5 {path})"),
        "generate": (f"Annotate your recorded demonstrations first:  isaaclab -p scripts/mimic_run.py annotate --task "
                     f"{MIMIC_TASK} --input_file ~/workspace/mimic/src.hdf5 --output_file {path} --auto"),
        "export": (f"Let Mimic generate the demonstrations first:  isaaclab -p scripts/mimic_run.py generate --task "
                   f"{MIMIC_TASK} --input_file ~/workspace/mimic/ann.hdf5 --output_file {path} "
                   f"--generation_num_trials 200 --num_envs 4"),
    }[step]
    if not os.path.isfile(path):
        raise Missing(f"There is no file {path}. {todo}")
    if _demo_count(path) == 0:
        raise Missing(f"The file {path} holds no demonstration. {todo}")
    return path


def need_episodes(name: str):
    """A folder of recorded episodes (ep_*.npz)."""
    import glob
    folder = dataset_folder(name)
    if not glob.glob(os.path.join(folder, "ep_*.npz")):
        raise Missing(f"There are no episodes (ep_*.npz) in {folder}. Make some first:  isaaclab -p scripts/generate.py "
                      f"--episodes 10 --name {os.path.basename(folder.rstrip('/'))}   (or record them:  "
                      f"isaaclab -p scripts/teleop.py --record {os.path.basename(folder.rstrip('/'))})")
    return folder


def need_policy(path: str):
    """A policy folder: policy.py and model files, or a training run (its last checkpoint)."""
    path = full_path(path)
    last = os.path.join(path, "checkpoints", "last", "pretrained_model")
    is_policy = lambda d: os.path.isfile(os.path.join(d, "policy.py")) or os.path.isfile(os.path.join(d, "config.json"))  # noqa: E731
    train = "Train a policy first:  nwaves-train <dataset> --policy act   (it writes ~/workspace/outputs/<run>)"
    if not os.path.isdir(path):
        raise Missing(f"There is no folder {path}. {train}. To try the scorer without a trained policy:  --policy policy/hold")
    if is_policy(path):
        return path
    if os.path.isdir(last) and is_policy(last):
        return last
    raise Missing(f"There is no policy in {path}: no policy.py, no trained model and no finished checkpoint "
                  f"(checkpoints/last/pretrained_model). {train}, or wait for the first checkpoint of a run that is still training.")


def _value(argv: list, name: str):
    """The value of an option in a list of arguments (--name VALUE or --name=VALUE), or None."""
    for k, arg in enumerate(argv):
        if arg == name and k + 1 < len(argv):
            return argv[k + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def check(script: str, argv: list):
    """The checks a script makes before it starts, from its arguments alone. Only what is certain: a file or a
    folder that is named and not there. Raises Missing."""
    name = os.path.basename(script)
    first = argv[0] if argv and not argv[0].startswith("-") else None
    if name == "mimic_run.py" and first in ("annotate", "generate"):
        given = _value(argv, "--input_file")
        if given:
            need_demos(given, first)
    elif name == "mimic_export.py" and first:
        need_demos(first, "export")
    elif name == "replay.py" and first:
        need_episodes(first)
    elif name == "evaluate.py" and _value(argv, "--policy"):
        need_policy(_value(argv, "--policy"))


def checked(tag: str, call, *args):
    """Run one of the need_... functions; end the script with its line when something is missing."""
    try:
        return call(*args)
    except Missing as exc:
        stop(tag, str(exc))


def load_settings(tag: str, name):
    """Read a generator settings file (nwaves_hackathon.randomization.load_config). A file that is not there or
    has a wrong entry ends the script with one line instead of a traceback."""
    from nwaves_hackathon import randomization
    try:
        return randomization.load_config(name)
    except FileNotFoundError as exc:
        stop(tag, f"{exc}. The shipped settings are configs/datagen.yaml: copy that file, change the copy and pass it "
                  f"with --config.")
    except (ValueError, OSError) as exc:
        stop(tag, f"{exc}")
    except Exception as exc:                          # a YAML syntax error
        if type(exc).__module__.startswith("yaml"):
            stop(tag, f"the settings file is not valid YAML: {' '.join(str(exc).split())}")
        raise


def window_argv(argv: list) -> list:
    """The same decision for scripts that hand their arguments on to an Isaac Lab script (the Mimic tools):
    returns argv with --headless removed, or with '--viz kit' added when a display is there and nothing was asked."""
    display = _lab_display()
    if "--headless" in argv:
        return [v for v in argv if v != "--headless"]
    if not any(v in ("--viz", "--visualizer") or v.startswith(("--viz=", "--visualizer=")) for v in argv) and display:
        fit_window()
        return list(argv) + ["--viz", "kit"]
    return list(argv)


if __name__ == "__main__":
    # python3 scripts/_start.py --check SCRIPT [ARGS...]: exit code CHECK_FAILED and one line when something the
    # script would read is missing; exit code 0 and no output otherwise. Starts nothing.
    if len(sys.argv) >= 3 and sys.argv[1] == "--check":
        try:
            check(sys.argv[2], sys.argv[3:])
        except Missing as exc:
            tags = {"mimic_run.py": "mimic", "mimic_export.py": "mimic-export", "replay.py": "replay", "evaluate.py": "eval"}
            print(f"[{tags.get(os.path.basename(sys.argv[2]), 'nwaves')}] {exc}", flush=True)
            sys.exit(CHECK_FAILED)
        except Exception:
            pass                                      # a check that cannot run never stops a script
    sys.exit(0)
