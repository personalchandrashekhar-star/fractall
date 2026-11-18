"""Fingertip pose IK with an integral term, so the SO-101 actually reaches the commanded pose.

Isaac Lab's differential IK sets each joint target to (current joint angle + IK correction). With the system-identified
gains the arm sags under gravity, and that law settles where the PD drive's pull equals the sag: the fingertip stays
8-10 mm (up to 25 mm) off the command. Here a per-joint offset accumulates the remaining correction, so the error
goes to zero; it is clamped, and cleared on reset, so a blocked arm cannot wind it up.

The term can also be told to drive the joints directly for a while (``joint_override``): the teleop's h key and the
generator's "back to home" glide the joints to the home pose, like the real arm does, instead of dragging the
fingertip through space the 5-joint arm cannot always follow.

Every joint target, here and in the official task, passes a limiter that keeps it inside what the real arm's
runner allows (see TargetLimiter).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab_tasks.contrib.stack.config.so101.pose_ik_action_term import SO101PoseIKAction

from . import layout as L


class TargetLimiter:
    """Keeps joint targets inside what the real arm's runner allows (layout.MAX_STEP_DEG, layout.MAX_LEAD_DEG):
    a target moves at most MAX_STEP_DEG per control tick and stays within MAX_LEAD_DEG of where the joint was when
    the tick began."""

    def __init__(self, num_envs: int, num_joints: int, substeps: int, device):
        self.step = math.radians(L.MAX_STEP_DEG) / max(int(substeps), 1)       # per physics step
        self.lead = math.radians(L.MAX_LEAD_DEG)
        self.prev = torch.zeros(num_envs, num_joints, device=device)
        self.anchor = torch.zeros(num_envs, num_joints, device=device)
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=device)

    def begin_tick(self, joint_pos: torch.Tensor):
        """Call once per control tick, before the physics steps, with the measured joint positions."""
        self.anchor = joint_pos.clone()
        if bool(self.fresh.any()):                       # first tick after a reset: start from where the arm is
            self.prev = torch.where(self.fresh[:, None], joint_pos, self.prev)
            self.fresh[:] = False

    def __call__(self, target: torch.Tensor) -> torch.Tensor:
        target = self.anchor + (target - self.anchor).clamp(-self.lead, self.lead)
        target = self.prev + (target - self.prev).clamp(-self.step, self.step)
        self.prev = target.clone()
        return target

    def reset(self, env_ids=None):
        if env_ids is None:
            self.fresh[:] = True
        else:
            self.fresh[env_ids] = True


class SO101PoseIKIntegralAction(SO101PoseIKAction):

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._ik_integral = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self._gain = float(getattr(cfg, "integral_gain", 0.03))
        self._limit = float(getattr(cfg, "integral_limit", 0.12))
        # Joint targets [num_envs, 5] that replace the IK while set. ``joint_override_mask`` ([num_envs] bool, or
        # None = every env) says for which environments.
        self.joint_override = None
        self.joint_override_mask = None
        self._last_override = None
        self._last_mask = None
        self._limiter = TargetLimiter(self.num_envs, len(self._joint_ids), env.cfg.decimation, self.device)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._limiter.begin_tick(self._asset.data.joint_pos.torch[:, self._joint_ids])

    def apply_actions(self):
        override, mask = self.joint_override, None
        if override is not None:
            override = override.to(device=self.device, dtype=torch.float32)
            mask = self.joint_override_mask
            mask = (torch.ones(self.num_envs, dtype=torch.bool, device=self.device) if mask is None
                    else mask.to(device=self.device, dtype=torch.bool))
        joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
        if self._last_override is not None:
            # back from an override: start the integral at the offset the override needed against gravity, so the
            # arm does not sag while the IK takes over
            released = self._last_mask if mask is None else (self._last_mask & ~mask)
            if bool(released.any()):
                seed = (self._last_override - joint_pos).clamp(-self._limit, self._limit)
                self._ik_integral = torch.where(released[:, None], seed, self._ik_integral)
        if mask is not None and bool(mask.all()):
            # every arm is driven by joint targets in this step (the keyboard teleop always is): no IK to compute
            self._ik_integral.zero_()
            self._last_override, self._last_mask = override.clone(), mask.clone()
            self._asset.set_joint_position_target_index(target=self._limiter(override), joint_ids=self._joint_ids)
            return
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        if not self._limits_injected and getattr(self.cfg.controller, "joint_limit_avoidance_gain", 0.0) > 0.0:
            limits = self._asset.data.soft_joint_pos_limits.torch[0, self._joint_ids, :]
            self._ik_controller.set_joint_pos_limits(limits[:, 0].clone(), limits[:, 1].clone())
            self._limits_injected = True
        jacobian = self._compute_frame_jacobian()
        joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        joint_pos_des = torch.where(ee_quat_curr.norm() != 0, joint_pos_des, joint_pos)
        delta = joint_pos_des - joint_pos
        self._ik_integral = (self._ik_integral + self._gain * delta).clamp(-self._limit, self._limit)
        target = joint_pos_des + self._ik_integral
        if mask is not None:
            target = torch.where(mask[:, None], override, target)
            self._ik_integral = torch.where(mask[:, None], torch.zeros_like(self._ik_integral), self._ik_integral)
            self._last_override, self._last_mask = override.clone(), mask.clone()
        else:
            self._last_override, self._last_mask = None, None
        target = self._limiter(target)
        self._asset.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._limiter.reset(env_ids)
        if env_ids is None:
            self._ik_integral.zero_()
        else:
            self._ik_integral[env_ids] = 0.0


class LimitedJointPositionAction(JointPositionAction):
    """Joint position targets of the official task (rad, the order of layout.JOINTS). The five arm joints pass
    through the same limiter as above; the gripper target is taken as it is."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        names = list(self._joint_names)                                    # the action's joints, in action order
        self._arm_cols = [k for k, n in enumerate(names) if n in L.HOME_RAD]
        self._arm_ids = [self._asset.joint_names.index(names[k]) for k in self._arm_cols]
        self._limiter = TargetLimiter(self.num_envs, len(self._arm_cols), env.cfg.decimation, self.device)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._limiter.begin_tick(self._asset.data.joint_pos.torch[:, self._arm_ids])

    def apply_actions(self):
        target = self.processed_actions.clone()
        target[:, self._arm_cols] = self._limiter(target[:, self._arm_cols])
        self._asset.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._limiter.reset(env_ids)
