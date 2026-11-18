"""Episode recorder: writes one compressed .npz per episode (JPEG frames, joint units, actions).

The generator, the keyboard teleop and the Mimic export all write this format; ``nwaves-dataset`` (to_lerobot.py)
turns a folder of episodes into a LeRobot dataset. Layout of one file:

    state      float32 (T, 6)   joints in interface units (degrees, gripper 0..100), observation at t
    action     float32 (T, 6)   joint targets commanded at t, same units
    timestamp  float32 (T,)     seconds from the start of the episode
    scene      object  (T,)     JPEG bytes, 640x480 RGB
    wrist      object  (T,)     JPEG bytes, 640x480 RGB
    meta       str              JSON: task, instruction, fps, success, time_s, source, layout (where the cube
                                and the tray were), ...
"""
import json
import os

import numpy as np

from . import layout as L


def _jpeg(rgb_uint8, quality=90):
    import cv2
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(rgb_uint8), cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


_pool = None


def _encoder():
    """A few threads that turn frames into JPEG while the simulation goes on (OpenCV releases the GIL)."""
    global _pool
    if _pool is None:
        from concurrent.futures import ThreadPoolExecutor
        _pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jpeg")
    return _pool


class EpisodeRecorder:
    """Collects one episode frame by frame and writes it as ep_<number>.npz into ``root``."""

    def __init__(self, root: str, fps: float = L.CONTROL_HZ):
        self.root, self.fps = root, fps
        os.makedirs(root, exist_ok=True)
        self.index = len([f for f in os.listdir(root) if f.endswith(".npz")])
        self.clear()

    def clear(self):
        self.state, self.action, self.ts, self.scene, self.wrist = [], [], [], [], []

    def __len__(self):
        return len(self.state)

    def add(self, joint_pos_rad, target_rad, scene_rgb, wrist_rgb, t):
        """One control tick: the joint positions and camera images the policy would see, and the joint targets
        that were commanded for this tick (all joints in rad; images uint8 RGB)."""
        self.state.append(np.asarray(L.rad_to_units(list(joint_pos_rad)), dtype=np.float32))
        self.action.append(np.asarray(L.rad_to_units(list(target_rad)), dtype=np.float32))
        self.ts.append(float(t))
        self.scene.append(_encoder().submit(_jpeg, np.ascontiguousarray(scene_rgb)))
        self.wrist.append(_encoder().submit(_jpeg, np.ascontiguousarray(wrist_rgb)))

    def save(self, meta: dict) -> str:
        path = os.path.join(self.root, f"ep_{self.index:05d}.npz")
        while os.path.exists(path):                      # several recorders may write into the same folder
            self.index += 1
            path = os.path.join(self.root, f"ep_{self.index:05d}.npz")
        meta = dict(meta, fps=self.fps, frames=len(self), joints=list(L.JOINTS),
                    units="deg, gripper 0..100", image_hw=(L.IMAGE_H, L.IMAGE_W))
        np.savez_compressed(path, state=np.stack(self.state), action=np.stack(self.action),
                            timestamp=np.asarray(self.ts, dtype=np.float32),
                            scene=np.array([f.result() for f in self.scene], dtype=object),
                            wrist=np.array([f.result() for f in self.wrist], dtype=object),
                            meta=json.dumps(meta))
        self.index += 1
        self.clear()
        return path


def enable_hdf5_recorder(cfg, path: str):
    """Isaac Lab's demo recorder (actions + states + initial state per episode, successful episodes only), the input
    format of Isaac Lab Mimic. Call on the env config before gym.make; mark success with export_success(env)."""
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cfg.recorders = ActionStateRecorderManagerCfg()
    cfg.recorders.dataset_export_dir_path = os.path.dirname(path)
    cfg.recorders.dataset_filename = os.path.splitext(os.path.basename(path))[0]
    cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    return cfg


def export_success(env):
    """Store the current episode of env 0 in the HDF5 file as a successful demonstration."""
    import torch
    u = env.unwrapped
    u.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
    u.recorder_manager.set_success_to_episodes([0], torch.tensor([[True]], dtype=torch.bool, device=u.device))
    u.recorder_manager.export_episodes([0])
