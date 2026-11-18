"""Kinematics of the SO-101 for the keyboard teleop (numpy only, no simulator needed).

The SO-101 has five arm joints: the base turns the whole arm (shoulder_pan), three parallel joints move it in its
own vertical plane (shoulder_lift, elbow_flex, wrist_flex) and the last one spins the gripper (wrist_roll). So what
the arm can really be asked for is: where the fingertip is (x, y, z), how far the gripper is tilted in the arm's
plane, and the wrist roll. ``solve`` turns exactly that into joint angles; the fingertip position comes first, the
tilt is met as far as the joint limits allow.

Everything is in the scene frame (x forward from the base, y left, z up from the work surface), angles in radians.
"""
from __future__ import annotations

import math

import numpy as np

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def _rot_wxyz(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _frame(rot, pos):
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = pos
    return m


class ArmKinematics:
    """Forward and inverse kinematics of the five arm joints.

    ``frames``: for every arm joint, in chain order, (pos0, rot0_wxyz, pos1, rot1_wxyz): the joint frame in the
    parent link and in the child link, as the robot's USD stores them (physics:localPos0 / localRot0 / localPos1 /
    localRot1). Every joint turns about the z axis of its frame.
    ``root``: 4x4 pose of the robot's base link in the scene frame.  ``tcp``: the fingertip point in the gripper link.
    ``limits``: [5, 2] joint limits in radians.
    """

    def __init__(self, frames, root, tcp, limits):
        self._pre = [_frame(_rot_wxyz(r0), p0) for p0, r0, _, _ in frames]
        self._post = [np.linalg.inv(_frame(_rot_wxyz(r1), p1)) for _, _, p1, r1 in frames]
        self._root = np.asarray(root, dtype=float)
        self._tcp = np.asarray(tcp, dtype=float)
        self.limits = np.asarray(limits, dtype=float)
        # the tilt: how far the gripper points below the horizontal, 0 at the home pose (pointing forward), +90 deg =
        # straight down. The three plane joints are parallel, so it is (+-) their sum.
        zero = np.zeros(5)
        down0 = self._down(zero)
        probe = zero.copy()
        probe[1] = 0.2
        self._tilt_sign = 1.0 if self._down(probe) > down0 else -1.0
        self._tilt_zero = down0

    @classmethod
    def from_stage(cls, stage, robot_prim_path, root_pos, root_quat_xyzw, tcp, limits):
        """Read the joint frames of the robot at ``robot_prim_path`` from the USD stage."""
        frames = []
        for name in ARM_JOINTS:
            prim = stage.GetPrimAtPath(f"{robot_prim_path}/joints/{name}")
            if not prim or not prim.IsValid():
                raise RuntimeError(f"joint {name} not found under {robot_prim_path}/joints")

            def get(attr, default):
                value = prim.GetAttribute(attr).Get()
                return default if value is None else value
            p0, p1 = get("physics:localPos0", (0, 0, 0)), get("physics:localPos1", (0, 0, 0))
            r0, r1 = prim.GetAttribute("physics:localRot0").Get(), prim.GetAttribute("physics:localRot1").Get()
            q0 = (1, 0, 0, 0) if r0 is None else (r0.GetReal(), *r0.GetImaginary())
            q1 = (1, 0, 0, 0) if r1 is None else (r1.GetReal(), *r1.GetImaginary())
            frames.append((tuple(p0), q0, tuple(p1), q1))
        x, y, z, w = root_quat_xyzw
        return cls(frames, _frame(_rot_wxyz((w, x, y, z)), root_pos), tcp, limits)

    # ------------------------------------------------------------------------------------------------ forward
    def _chain(self, q):
        """Pose of the gripper link, and the origin and axis of every joint, in the scene frame."""
        m = self._root
        origins, axes = [], []
        for k in range(5):
            m = m @ self._pre[k]
            origins.append(m[:3, 3].copy())
            axes.append(m[:3, 2].copy())
            c, s = math.cos(q[k]), math.sin(q[k])
            turn = np.eye(4)
            turn[0, 0], turn[0, 1], turn[1, 0], turn[1, 1] = c, -s, s, c
            m = m @ turn @ self._post[k]
        return m, origins, axes

    def pose(self, q):
        """Fingertip position [3] and gripper link rotation [3, 3] for joint angles ``q`` [5]."""
        m, _, _ = self._chain(q)
        return m[:3, :3] @ self._tcp + m[:3, 3], m[:3, :3]

    def tip(self, q):
        return self.pose(q)[0]

    def _down(self, q):
        rot = self.pose(q)[1]
        tool = -rot[:, 2]                                  # the gripper points along the link's -z
        return math.atan2(-tool[2], math.hypot(tool[0], tool[1]))

    def tilt(self, q):
        """How far the gripper is tilted down in the arm's plane [rad]: 0 = level (the home pose), pi/2 = straight
        down. It keeps counting past straight down."""
        return self._tilt_zero + self._tilt_sign * float(q[1] + q[2] + q[3])

    def pan_axis(self):
        """(x, y) of the vertical axis the whole arm turns about."""
        _, origins, _ = self._chain(np.zeros(5))
        return origins[0][:2].copy()

    # ------------------------------------------------------------------------------------------------ inverse
    def solve(self, pos, tilt, roll, seed, iterations=40):
        """Joint angles [5] that put the fingertip at ``pos`` with the gripper tilted by exactly ``tilt`` and the
        wrist rolled to ``roll``, starting from ``seed`` (the solution stays on the seed's side: elbow up stays
        elbow up). Returns (q, position error [m], tilt error [rad]); the errors are zero when the pose can be
        reached inside the joint limits."""
        lo, hi = self.limits[:, 0], self.limits[:, 1]
        q = np.clip(np.asarray(seed, dtype=float).copy(), lo, hi)
        q[4] = min(max(float(roll), lo[4]), hi[4])
        pos = np.asarray(pos, dtype=float)
        scale = 0.2                                         # one radian of tilt counts like 0.2 m: same order as the arm
        row = np.array([0.0, 1.0, 1.0, 1.0]) * self._tilt_sign * scale
        last = float("inf")
        for it in range(iterations):
            m, origins, axes = self._chain(q)
            tip = m[:3, :3] @ self._tcp + m[:3, 3]
            err = np.concatenate([pos - tip, [scale * (tilt - self.tilt(q))]])
            size = float(np.linalg.norm(err))
            if size < 1e-7 or (it >= 6 and size > 0.98 * last):     # there, or no longer getting closer (out of reach)
                break
            last = size
            jac = np.zeros((4, 4))
            for k in range(4):
                jac[:3, k] = np.cross(axes[k], tip - origins[k])
            jac[3] = row
            # a joint resting on its limit that the step would push further out is left out of the step
            free = np.ones(4, dtype=bool)
            step = np.zeros(4)
            for _ in range(4):
                jf = jac[:, free]
                step[:] = 0.0
                step[free] = np.linalg.solve(jf.T @ jf + 1e-9 * np.eye(int(free.sum())), jf.T @ err)
                stuck = free & (((q[:4] <= lo[:4] + 1e-9) & (step < 0)) | ((q[:4] >= hi[:4] - 1e-9) & (step > 0)))
                if not stuck.any():
                    break
                free &= ~stuck
                if not free.any():
                    break
            big = float(np.abs(step).max())
            if big > 0.2:                                   # small steps: the solution stays on its branch
                step *= 0.2 / big
            q[:4] = np.clip(q[:4] + step, lo[:4], hi[:4])
            if big < 1e-8:
                break
        return q, float(np.linalg.norm(pos - self.tip(q))), abs(float(tilt - self.tilt(q)))

    def reach(self, pos, tilt, roll, seed, pos_tol=3e-4, tilt_tol=2e-3):
        """The position comes first: joint angles that put the fingertip at ``pos``, with the gripper tilt as close
        to ``tilt`` as the arm allows there (far out the arm has to stretch and the gripper comes up; close to the
        base the wrist reaches its limit). ``seed`` is where the arm is now.

        Returns (q, tilt reached, ok). ok is False when the position itself is out of reach; q is then the seed."""
        def good(t):
            q_, e_pos, e_tilt = self.solve(pos, t, roll, seed)
            return q_ if (e_pos < pos_tol and e_tilt < tilt_tol) else None

        q = good(tilt)
        if q is not None:
            return q, tilt, True
        # the wanted tilt does not work here: find the nearest one that does, starting from the tilt the arm has
        now = self.tilt(seed)
        base = good(now)
        if base is None:
            for delta in (0.03, 0.08, 0.15, 0.3):           # look around the present tilt, away from the wanted one first
                for t in (now - math.copysign(delta, tilt - now), now + math.copysign(delta, tilt - now)):
                    base = good(t)
                    if base is not None:
                        now = t
                        break
                if base is not None:
                    break
            if base is None:
                return np.asarray(seed, dtype=float).copy(), self.tilt(seed), False
        ok_t, bad_t, best = now, tilt, base
        for _ in range(7):                                  # bisect between a tilt that works and the wanted one
            mid = 0.5 * (ok_t + bad_t)
            q = good(mid)
            if q is not None:
                ok_t, best = mid, q
            else:
                bad_t = mid
        return best, ok_t, True


class ArmCommand:
    """What the teleop's keys move: the fingertip position, the gripper tilt and the wrist roll, kept as a joint
    command ``q`` [5] that the arm follows.

    ``move`` advances it by one tick of key motion, ``release`` is called when the keys are let go, ``target`` gives
    the joint targets to send. Pure arithmetic on the kinematics above: it can be tried without the simulator.
    """

    def __init__(self, kin, work_min, work_max, dt=1.0 / 30.0, r_min=0.10, tilt_range=(-0.8, 2.1),
                 max_step=math.radians(4.0), max_lead=math.radians(10.0), hold_torque=1.2,
                 tilt_return=math.radians(1.0)):
        """``dt``: the control tick [s]. ``max_step`` / ``max_lead``: what the real arm's runner allows a joint
        target to change per tick, and to be ahead of the joint (layout.MAX_STEP_DEG, layout.MAX_LEAD_DEG).
        ``hold_torque`` [N m]: see release() (holding the arm up against gravity takes up to about 0.7 N m)."""
        self.kin, self.dt = kin, dt
        self.work_min, self.work_max = np.asarray(work_min, dtype=float), np.asarray(work_max, dtype=float)
        self.r_min, self.tilt_range = r_min, tilt_range
        self.max_step, self.max_lead, self.hold_torque, self.tilt_return = max_step, max_lead, hold_torque, tilt_return
        self.pan_axis = kin.pan_axis()

    def anchor(self, q, stiffness, damping):
        """Command = these joint angles (after a reset and after homing). ``stiffness`` and ``damping`` [5] are the
        joint drives' gains."""
        self.q = np.array(q, dtype=float)
        self.pos = self.kin.tip(self.q)
        self.tilt = self.kin.tilt(self.q)          # the tilt you asked for; the arm returns to it when it can
        self.sent = self.q.copy()
        kp, kd = np.maximum(np.asarray(stiffness, dtype=float), 1e-6), np.asarray(damping, dtype=float)
        # Each joint drive follows a moving target with a delay of (damping / stiffness) seconds, a different one
        # per joint, which would bend straight lines. The target sent to a joint is therefore set ahead by that
        # delay times the commanded joint speed; it is the command itself again as soon as the command stands still.
        self.delay = kd / kp
        self.slack = self.hold_torque / kp         # joint error [rad] at which a drive pushes with hold_torque
        # How far the command may turn each joint in one tick: a little under what the real arm's runner lets a
        # target change, and no faster than the joint's drive can follow with its target max_lead ahead of it (the
        # wrist's drive is the slow one).
        self.joint_step = np.minimum(0.875 * self.max_step, 0.8 * self.max_lead * self.dt / np.maximum(self.delay, 1e-6))
        self.vel = np.zeros(5)                     # the command's joint step of the last tick
        self.glide = 0                             # ticks of slowing down left after the keys were let go
        self.moved = False                         # did a key move the command in this tick?

    def target(self):
        """The joint targets for this tick: the command, set ahead by each drive's delay. Call once per tick, after
        move() or release()."""
        if not self.moved:
            if self.glide > 0:
                # The keys were let go: the command comes to rest over two ticks (half a step, a quarter step)
                # instead of stopping dead. The targets run ahead of the command by each drive's delay, and the
                # real arm's runner lets a target change by max_step per tick only: stopping dead would leave the
                # targets ahead for a moment and the arm would overshoot and come back.
                self.glide -= 1
                self.vel = 0.5 * self.vel
                self.q = np.clip(self.q + self.vel, self.kin.limits[:, 0], self.kin.limits[:, 1])
                self.pos = self.kin.tip(self.q)
            else:
                self.vel = np.zeros(5)
        self.moved = False
        cap = self.max_lead - math.radians(2.0)
        ahead = np.clip(self.delay * (self.q - self.sent) / self.dt, -cap, cap)
        self.sent = self.q.copy()
        return self.q + ahead

    def clamp(self, point):
        """Keep a fingertip point inside the box above the table and r_min away from the arm's turning axis."""
        point = np.minimum(np.maximum(point, self.work_min), self.work_max)
        d = point[:2] - self.pan_axis
        r = float(np.linalg.norm(d))
        if r < self.r_min:
            point[:2] = self.pan_axis + (d * (self.r_min / r) if r > 1e-4 else np.array([self.r_min, 0.0]))
        return point

    def move(self, d, dtilt, droll, measured):
        """One tick of key motion: fingertip step ``d`` [3] (scene frame), tilt step, roll step; ``measured`` is
        where the arm's joints are. Returns "edge" when the arm cannot go (all the way) where the keys ask, "tilt"
        when a tilt key changed nothing because the wrist is at its limit, else None."""
        if float(np.abs(self.q - measured).max()) > self.max_lead:
            self.release(measured)                 # something is in the arm's way: go on from where the arm is
        now = self.kin.tilt(self.q)
        if dtilt:                                  # a tilt key starts from the tilt the arm has, not from an old wish
            wanted = min(max(now + dtilt, self.tilt_range[0]), self.tilt_range[1])
            rate = abs(dtilt)
        else:                                      # otherwise the gripper goes back to the tilt you last set, gently
            wanted, rate = self.tilt, self.tilt_return
        best = None
        for part in (1.0, 0.5, 0.25, 0.125):
            pos = self.clamp(self.pos + part * d)
            tilt = now + part * min(max(wanted - now, -rate), rate)
            q, got, ok = self.kin.reach(pos, tilt, self.q[4] + part * droll, self.q)
            if not ok:
                continue
            step = float((np.abs(q - self.q) / self.joint_step).max())      # 1 = as fast as a joint may turn
            if best is None or step < best[0]:
                best = (step, q, pos, got, part)
            if step <= 1.0:
                break
        if best is None or best[0] > 1.7:          # even the shortest move would whip a joint round: full stretch
            return "edge"
        _, q, pos, got, part = best
        short = float(np.linalg.norm(pos - (self.pos + part * d))) > 1e-6
        self.vel, self.moved, self.glide = q - self.q, True, 0
        self.q, self.pos = q, pos                  # the asked-for point is kept, so small errors do not add up
        if dtilt:
            self.tilt = got                        # holding a tilt key against the wrist's limit winds nothing up
            if abs(got - now) < 0.05 * abs(dtilt) and not short:
                return "tilt"                      # the tilt key changed nothing: the wrist is at its limit here
        return "edge" if short else None

    def release(self, measured):
        """Keys let go: the arm stops. What the command is still ahead of the arm, beyond what a drive needs to hold
        the arm up against gravity, is dropped, so an arm leaning on something stops pushing."""
        q = measured + np.clip(self.q - measured, -self.slack, self.slack)
        if float(np.abs(q - self.q).max()) > 1e-6:
            self.q, self.pos = q, self.kin.tip(q)
            self.vel, self.glide = np.zeros(5), 0
        else:
            self.glide = 2                         # free motion: come to rest gently, see target()

