#!/usr/bin/env python3
"""Validator-clean Stage-2 ACT packager + dummy-obs smoke test for BIEC real-arm deployment.

Copies model.safetensors, config.json (scrubbing pretrained_path=None), and ALL new
policy_preprocessor* / policy_postprocessor* files together from --ckpt_dir, adds
policy.py (ACCEPTED=("act",), cuda->mps->cpu) and inference.json from --template_dir,
excludes train_config.json and optimizer state, runs a (25, 6) dummy-obs smoke check,
and writes --out_zip.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

REQUIRED_CKPT_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)
FORBIDDEN_FILES = {
    "train_config.json",
    "checkpoint_provenance.json",
    "checkpoint_origin.json",
    "continuation_provenance.json",
    "optimizer.pt",
    "scheduler.pt",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a 100%-real fine-tuned SO-101 ACT checkpoint.")
    parser.add_argument("--ckpt_dir", required=True, help="Path to checkpoints/<step>/pretrained_model")
    parser.add_argument("--template_dir", required=True, help="Path containing reference policy.py and inference.json")
    parser.add_argument("--out_zip", required=True, help="Output .zip path for NWaves upload")
    parser.add_argument("--n_action_steps", type=int, default=25, help="Open-loop chunk steps (default: 25)")
    parser.add_argument("--gripper_min", type=float, default=6.0, help="Safe closed clamp (default: 6.0)")
    parser.add_argument("--gripper_max", type=float, default=58.0, help="Safe open clamp (default: 58.0; use 68.0 if tight)")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    if (ckpt_dir / "checkpoints" / "last" / "pretrained_model").is_dir():
        ckpt_dir = ckpt_dir / "checkpoints" / "last" / "pretrained_model"
    elif (ckpt_dir / "pretrained_model").is_dir():
        ckpt_dir = ckpt_dir / "pretrained_model"

    template_dir = Path(args.template_dir).expanduser().resolve()
    out_zip = Path(args.out_zip).expanduser().resolve()

    for req in REQUIRED_CKPT_FILES:
        p = ckpt_dir / req
        if not p.is_file():
            raise FileNotFoundError(f"Missing required file in ckpt_dir ({ckpt_dir}): {req}")

    policy_src = template_dir / "policy.py"
    if not policy_src.is_file():
        raise FileNotFoundError(f"Missing policy.py in template_dir ({template_dir})")

    with tempfile.TemporaryDirectory(prefix="so101_stage2_pack_") as tmp:
        stage_dir = Path(tmp)
        for req in REQUIRED_CKPT_FILES:
            shutil.copy2(ckpt_dir / req, stage_dir / req)

        # Scrub pretrained_path in config.json and verify type == "act"
        cfg_path = stage_dir / "config.json"
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("type") != "act":
            raise ValueError(f"Expected config.json type='act', got {cfg.get('type')!r}")
        cfg["pretrained_path"] = None
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

        # Copy policy.py and write inference.json (snap_gripper strictly False for physical servos)
        shutil.copy2(policy_src, stage_dir / "policy.py")
        inference_cfg = {
            "n_action_steps": int(args.n_action_steps),
            "gripper_min": float(args.gripper_min),
            "gripper_max": float(args.gripper_max),
            "snap_gripper": False,
        }
        (stage_dir / "inference.json").write_text(json.dumps(inference_cfg, indent=2) + "\n")

        # Verify no forbidden files exist
        staged_names = {p.name for p in stage_dir.iterdir()}
        leaked = staged_names & FORBIDDEN_FILES
        if leaked:
            raise RuntimeError(f"Forbidden training metadata found in stage_dir: {sorted(leaked)}")

        # Run dummy-obs smoke check
        spec = importlib.util.spec_from_file_location("stage2_policy", str(stage_dir / "policy.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        pol = mod.Policy(str(stage_dir))
        obs = {
            "scene": np.zeros((480, 640, 3), dtype=np.uint8),
            "wrist": np.zeros((480, 640, 3), dtype=np.uint8),
            "joint_state": np.zeros((6,), dtype=np.float32),
        }
        t0 = time.perf_counter()
        out = pol.act(obs)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        expected_shape = (args.n_action_steps,) if args.n_action_steps == 1 else (args.n_action_steps, 6)
        if out.shape != expected_shape:
            raise AssertionError(f"Smoke test shape mismatch: expected {expected_shape}, got {out.shape}")

        # Build clean zip
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in sorted(stage_dir.iterdir()):
                zf.write(item, item.name)
                if item.name in FORBIDDEN_FILES:
                    raise RuntimeError(f"Refusing to zip forbidden file: {item.name}")

        weights_sha = sha256_file(stage_dir / "model.safetensors")
        pre_sha = sha256_file(stage_dir / "policy_preprocessor_step_3_normalizer_processor.safetensors")
        post_sha = sha256_file(stage_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
        print(f"[pack_stage2_real] SUCCESS -> {out_zip} ({out_zip.stat().st_size / (1024*1024):.1f} MB)")
        print(f"  device={pol.device} | shape={out.shape} | smoke_latency={dt_ms:.2f} ms | train_config.json=ABSENT")
        print(f"  model.safetensors sha256 : {weights_sha}")
        print(f"  preprocessor      sha256 : {pre_sha[:16]}... | postprocessor sha256 : {post_sha[:16]}...")


if __name__ == "__main__":
    main()
