"""Merge several LeRobot datasets into one, so a policy can be trained on all of them together
(for example your simulated episodes and the official real-robot episodes).

    python -m nwaves_hackathon.merge OUT_DIR DATASET_DIR DATASET_DIR [...]

Runs in the LeRobot venv; nwaves-train calls it when it is given more than one dataset. The datasets must have
the same frame rate and the same features: state and action with six values, the cameras ``scene`` and ``wrist``
at the same size, videos in the same codec. That is the case for everything made with nwaves-dataset. Harmless
differences (the robot type's name, the names of the six joints) are taken from the first dataset.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

SKIP = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def info(root):
    with open(os.path.join(root, "meta", "info.json")) as f:
        return json.load(f)


def compare(roots):
    """Plain sentences about what keeps the datasets from being merged (empty = fine)."""
    first, problems = info(roots[0]), []
    for root in roots[1:]:
        other = info(root)
        if other.get("fps") != first.get("fps"):
            problems.append(f"{root}: {other.get('fps')} frames per second, but {roots[0]} has {first.get('fps')}")
        a, b = first.get("features", {}), other.get("features", {})
        for key in sorted((set(a) | set(b)) - SKIP):
            if key not in a or key not in b:
                problems.append(f"'{key}' is in {roots[0] if key in a else root} only")
                continue
            if a[key].get("dtype") != b[key].get("dtype") or list(a[key].get("shape", [])) != list(b[key].get("shape", [])):
                problems.append(f"'{key}' is {b[key].get('dtype')} {b[key].get('shape')} in {root} but "
                                f"{a[key].get('dtype')} {a[key].get('shape')} in {roots[0]}")
                continue
            ia, ib = a[key].get("info") or {}, b[key].get("info") or {}
            for field in ("video.codec", "video.pix_fmt", "video.fps"):
                if ia.get(field) != ib.get(field):
                    problems.append(f"'{key}': {field} is {ib.get(field)} in {root} but {ia.get(field)} in {roots[0]} "
                                    f"(videos in different encodings cannot be joined)")
    return problems


def shadow(root, reference, folder):
    """A stand-in for a dataset folder whose info.json agrees with the first dataset in the details that do not
    change the data (robot type, feature names, video details already checked). Data and videos are linked."""
    meta, ref = info(root), reference
    patched = json.loads(json.dumps(meta))
    patched["robot_type"] = ref.get("robot_type")
    for key, feat in ref.get("features", {}).items():
        if key in patched["features"]:
            patched["features"][key] = json.loads(json.dumps(feat))
    if patched == meta:
        return root
    out = os.path.join(folder, os.path.basename(root))
    os.makedirs(out)
    shutil.copytree(os.path.join(root, "meta"), os.path.join(out, "meta"))
    with open(os.path.join(out, "meta", "info.json"), "w") as f:
        json.dump(patched, f, indent=4)
    for sub in os.listdir(root):
        if sub != "meta":
            os.symlink(os.path.join(root, sub), os.path.join(out, sub))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", help="folder of the merged dataset (replaced if it exists)")
    ap.add_argument("datasets", nargs="+", help="LeRobot dataset folders")
    a = ap.parse_args()
    roots = [os.path.abspath(d) for d in a.datasets]
    for root in roots:
        if not os.path.isfile(os.path.join(root, "meta", "info.json")):
            sys.exit(f"[merge] {root} is not a LeRobot dataset (no meta/info.json)")
    problems = compare(roots)
    if problems:
        print("[merge] these datasets cannot be trained together:")
        for line in problems:
            print(f"[merge]   {line}")
        sys.exit(1)
    out = os.path.abspath(a.out)
    if os.path.exists(out):
        shutil.rmtree(out)
    from lerobot.datasets.aggregate import aggregate_datasets
    with tempfile.TemporaryDirectory(prefix="nwaves-merge-") as tmp:
        reference = info(roots[0])
        use = [roots[0]] + [shadow(r, reference, tmp) for r in roots[1:]]
        ids = [f"local/{os.path.basename(r)}" for r in roots]
        aggregate_datasets(repo_ids=ids, aggr_repo_id=f"local/{os.path.basename(out)}", roots=[Path(r) for r in use],
                           aggr_root=Path(out))
    merged = info(out)
    print(f"[merge] {len(roots)} datasets -> {out}: {merged.get('total_episodes')} episodes, "
          f"{merged.get('total_frames')} frames", flush=True)


if __name__ == "__main__":
    main()
