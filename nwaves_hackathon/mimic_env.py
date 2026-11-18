"""Isaac Lab Mimic interface for the SO-101 hackathon scene.

Mimic stores and transforms poses relative to the objects. Here every pose is in the scene frame (env origin, x
forward from the base, y left, z up), the same frame as Isaac Lab's object poses. The arm's action is a fingertip
pose in the robot root frame, which is turned 90 deg in the scene; the conversions happen at the action boundary.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv

from . import layout as L
from . import mdp as H


class NWavesSO101MimicEnv(ManagerBasedRLMimicEnv):

    def _root_pose(self, env_ids):
        robot = self.scene["robot"]
        pos = H._t(robot.data.root_pos_w)[env_ids] - self.scene.env_origins[env_ids]
        return pos, H._t(robot.data.root_quat_w)[env_ids]

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)
        robot = self.scene["robot"]
        body = robot.body_names.index("gripper")
        pw = H._t(robot.data.body_pos_w)[env_ids, body]
        qw = H._t(robot.data.body_quat_w)[env_ids, body]
        tcp = torch.tensor(L.TCP_OFFSET, device=pw.device, dtype=pw.dtype).expand_as(pw)
        pos = pw + PoseUtils.quat_apply(qw, tcp) - self.scene.env_origins[env_ids]
        return PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(qw))

    def target_eef_pose_to_action(self, target_eef_pose_dict: dict, gripper_action_dict: dict,
                                  noise: float | None = None, env_id: int = 0) -> torch.Tensor:
        (target_pose,) = target_eef_pose_dict.values()
        pos_e, rot_e = PoseUtils.unmake_pose(target_pose)
        quat_e = PoseUtils.quat_from_matrix(rot_e)
        root_p, root_q = self._root_pose([env_id])
        pos_b, quat_b = PoseUtils.subtract_frame_transforms(root_p, root_q, pos_e.reshape(1, 3), quat_e.reshape(1, 4))
        pose_action = torch.cat([pos_b[0], quat_b[0]], dim=0)
        if noise is not None:
            pose_action[:3] += noise * torch.randn(3, device=pose_action.device)
        (gripper_action,) = gripper_action_dict.values()
        return torch.cat([pose_action, gripper_action.reshape(-1)], dim=0).unsqueeze(0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        eef_name = list(self.cfg.subtask_configs.keys())[0]
        n = action.shape[0]
        root_p, root_q = self._root_pose([0])           # fixed base: the same for every step of a demo
        pos_e, quat_e = PoseUtils.combine_frame_transforms(root_p.expand(n, 3), root_q.expand(n, 4),
                                                           action[:, :3], action[:, 3:7])
        return {eef_name: PoseUtils.make_pose(pos_e, PoseUtils.matrix_from_quat(quat_e)).clone()}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        return {list(self.cfg.subtask_configs.keys())[0]: actions[:, -1:]}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        # only the signals this task's subtasks use: the annotator requires every reported signal to fire
        used = {sc.subtask_term_signal for scs in self.cfg.subtask_configs.values() for sc in scs if sc.subtask_term_signal}
        return {name: sig[env_ids] for name, sig in H.subtask_signals(self).items() if name in used}
