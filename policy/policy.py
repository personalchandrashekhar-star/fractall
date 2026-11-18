import json
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

ACCEPTED = ("act",)
CAMERAS = ("scene", "wrist")

JAW_CLOSED_UNITS = 6.49
JAW_OPEN_UNITS = 42.0
GRIPPER_MIN_SAFE = 6.0
GRIPPER_MAX_SAFE = 58.0
ARM_ABS_LIMIT_DEG = 175.0


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Policy:
    def __init__(self, model_dir: str):
        self.device = _select_device()
        settings_path = Path(model_dir) / "inference.json"
        settings = json.loads(settings_path.read_text()) if settings_path.is_file() else {}
        self.chunk_steps = int(settings.get("n_action_steps", 25))
        self.snap_gripper = bool(settings.get("snap_gripper", False))
        self.gripper_min = float(settings.get("gripper_min", GRIPPER_MIN_SAFE))
        self.gripper_max = float(settings.get("gripper_max", GRIPPER_MAX_SAFE))

        config = PreTrainedConfig.from_pretrained(model_dir)
        if config.type not in ACCEPTED:
            raise ValueError(f"Only ACT is accepted, got {config.type}")
        config.device = self.device
        self.model_type = config.type
        self.policy = get_policy_class(config.type).from_pretrained(model_dir, config=config)
        self.policy.to(self.device)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=model_dir,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )
        self.instruction = ""
        if self.device in ("cuda", "mps"):
            dummy_obs = {
                "scene": np.zeros((480, 640, 3), dtype=np.uint8),
                "wrist": np.zeros((480, 640, 3), dtype=np.uint8),
                "joint_state": np.zeros((6,), dtype=np.float32),
            }
            for _ in range(3):
                self.act(dummy_obs)
            if self.device == "cuda":
                torch.cuda.synchronize()

    def reset(self, task: str = "", instruction: str = ""):
        self.policy.reset()
        self.instruction = instruction

    def _guard_actions(self, actions: np.ndarray) -> np.ndarray:
        arr = np.asarray(actions, dtype=np.float32).copy()
        arr[..., :5] = np.clip(arr[..., :5], -ARM_ABS_LIMIT_DEG, ARM_ABS_LIMIT_DEG)
        arr[..., 5] = np.clip(arr[..., 5], self.gripper_min, self.gripper_max)
        if self.snap_gripper:
            open_mask = arr[..., 5] > 28.0
            close_mask = arr[..., 5] < 11.0
            arr[..., 5] = np.where(open_mask, np.maximum(arr[..., 5], JAW_OPEN_UNITS), arr[..., 5])
            arr[..., 5] = np.where(close_mask, np.minimum(arr[..., 5], JAW_CLOSED_UNITS), arr[..., 5])
        return arr

    def act(self, obs: dict) -> np.ndarray:
        batch = {
            "observation.state": torch.as_tensor(obs["joint_state"], dtype=torch.float32, device=self.device).view(1, 6),
            "task": [self.instruction],
        }
        non_blocking = self.device == "cuda"
        for name in CAMERAS:
            rgb = np.ascontiguousarray(obs[name])
            t = torch.from_numpy(rgb).to(device=self.device, non_blocking=non_blocking)
            batch[f"observation.images.{name}"] = t.permute(2, 0, 1).float().mul_(1.0 / 255.0).unsqueeze(0)
        batch = self.pre(batch)
        use_amp = self.device == "cuda"
        with torch.inference_mode(), torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=torch.float16, enabled=use_amp):
            chunk_t = self.policy.predict_action_chunk(batch)
            chunk_np = self.post(chunk_t).detach().cpu().numpy().reshape(-1, 6)
        out = self._guard_actions(chunk_np[: self.chunk_steps])
        return out[0] if self.chunk_steps == 1 else out
