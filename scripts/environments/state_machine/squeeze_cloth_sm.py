# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted squeeze state machine for the AVBD/RVBD cloth scenes.

Drives the interactive-IK target of ``Isaac-Pick-AVBD-Cloth-Direct-v0`` /
``Isaac-Pick-RVBD-Cloth-Direct-v0`` through a sequence that moves the arm into a
fixed initialization pose, closes the gripper, then presses the cloth down
against the ground:

1. REST         -- hold the default (reset) EE pose, gripper open.
2. MOVE_TO_INIT -- move the EE (position + orientation) from the default pose to
                   the fixed initialization pose ``INIT_TF``, gripper open.
3. GRASP        -- hold the init pose and close the gripper.
4. MOVE_DOWN    -- with the gripper closed, lower the EE to ``--squeeze_z``
                   (holding the init x/y and orientation), pressing the cloth
                   against the ground.
5. DONE         -- hold the low (squeezed) pose, gripper closed.

The scene's interactive IK is single-environment, so this runs with num_envs=1.

.. code-block:: bash

    ./isaaclab.sh -p scripts/environments/state_machine/squeeze_cloth_sm.py \
        --task Isaac-Pick-RVBD-Cloth-Direct-v0 --num_envs 1 --visualizer newton presets=newton
"""

import argparse
import sys

import gymnasium as gym
import torch
import warp as wp

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

# Fixed initialization pose the arm moves into before grasping (EE = panda_hand).
# wp.quat / body_link_pose_w order is (x, y, z, w).
INIT_POS = (0.5202, 0.0714, 0.3366)
INIT_QUAT = (0.7171, -0.6926, 0.0599, 0.0495)

parser = argparse.ArgumentParser(description="Scripted squeeze state machine for the cloth scenes.")
parser.add_argument("--task", type=str, default="Isaac-Pick-RVBD-Cloth-Direct-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (IK is single-env).")
parser.add_argument(
    "--squeeze_z",
    type=float,
    default=0.10,
    help="World z [m] the EE descends to (gripper closed) to press the cloth against the ground.",
)
parser.add_argument(
    "--move_speed",
    type=float,
    default=0.3,
    help="EE target slew speed [m/s] for the move into the initialization pose.",
)
parser.add_argument(
    "--squeeze_speed",
    type=float,
    default=0.2,
    help="EE target slew speed [m/s] for the squeeze descent.",
)
parser.add_argument("--settle_wait", type=float, default=0.5, help="Wait [s] after reaching the init pose.")
parser.add_argument("--grasp_wait", type=float, default=0.6, help="Wait [s] for the gripper to close before descending.")
parser.add_argument("--squeeze_wait", type=float, default=1.0, help="Wait [s] to hold the squeeze at full depth.")
parser.add_argument("--max_steps", type=int, default=100000, help="Max env steps when no visualizer is open.")
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args


def _slerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two unit quaternions (order-agnostic)."""
    q0 = q0 / torch.norm(q0)
    q1 = q1 / torch.norm(q1)
    dot = float(torch.dot(q0, q1))
    if dot < 0.0:  # take the shorter arc
        q1 = -q1
        dot = -dot
    if dot > 0.9995:  # nearly parallel -> linear interp + renormalize
        q = q0 + (q1 - q0) * t
        return q / torch.norm(q)
    theta0 = torch.acos(torch.tensor(dot).clamp(-1.0, 1.0))
    theta = theta0 * t
    q2 = q1 - q0 * dot
    q2 = q2 / torch.norm(q2)
    return q0 * torch.cos(theta) + q2 * torch.sin(theta)


class ClothSqueezeSm:
    """Single-environment task-space squeeze state machine.

    Moves the EE from the default (reset) pose into a fixed init pose (position +
    orientation), closes the gripper, then lowers to ``squeeze_z`` with the
    gripper closed, pressing the cloth against the ground. Returns a desired EE
    target pose (position + quaternion) and a gripper-closed flag each tick.
    """

    REST = 0
    MOVE_TO_INIT = 1
    GRASP = 2
    MOVE_DOWN = 3
    DONE = 4

    def __init__(
        self,
        start_pos,
        start_quat,
        init_pos,
        init_quat,
        squeeze_z,
        dt,
        move_speed,
        squeeze_speed,
        settle_wait,
        grasp_wait,
        squeeze_wait,
        rest_wait=0.3,
        min_move_time=1.0,
    ):
        self.start_pos = start_pos.clone()
        self.start_quat = start_quat.clone()
        self.init_pos = init_pos.clone()
        self.init_quat = init_quat.clone()
        self.squeeze_pos = init_pos.clone()
        self.squeeze_pos[2] = squeeze_z
        self.dt = float(dt)
        self.squeeze_speed = float(squeeze_speed)
        self.settle_wait = float(settle_wait)
        self.grasp_wait = float(grasp_wait)
        self.squeeze_wait = float(squeeze_wait)
        self.rest_wait = float(rest_wait)

        # Parameterize the init move by time so both position and orientation
        # interpolate smoothly even when one of them changes little.
        init_dist = float(torch.norm(init_pos - start_pos))
        self._move_duration = max(init_dist / float(move_speed), float(min_move_time))

        self.state = self.REST
        self.wait = 0.0
        self.s = 0.0  # MOVE_TO_INIT progress in [0, 1]
        self.target_pos = start_pos.clone()
        self.target_quat = start_quat.clone()

    def _slew_to(self, goal, speed) -> bool:
        """Move the commanded position toward ``goal`` at ``speed``. Returns True if reached."""
        delta = goal - self.target_pos
        dist = float(torch.norm(delta))
        max_step = speed * self.dt
        if dist <= max_step or dist < 1e-9:
            self.target_pos = goal.clone()
            return True
        self.target_pos = self.target_pos + delta * (max_step / dist)
        return False

    def compute(self):
        """Advance the state machine one tick; return (target_pos, target_quat, gripper_closed)."""
        gripper_closed = False

        if self.state == self.REST:
            self.target_pos = self.start_pos.clone()
            self.target_quat = self.start_quat.clone()
            if self.wait >= self.rest_wait:
                self.state = self.MOVE_TO_INIT
                self.wait = 0.0

        elif self.state == self.MOVE_TO_INIT:
            self.s = min(1.0, self.s + self.dt / self._move_duration)
            self.target_pos = self.start_pos + (self.init_pos - self.start_pos) * self.s
            self.target_quat = _slerp(self.start_quat, self.init_quat, self.s)
            if self.s < 1.0:
                self.wait = 0.0  # only count settle time once the move completes
            elif self.wait >= self.settle_wait:
                self.state = self.GRASP
                self.wait = 0.0

        elif self.state == self.GRASP:
            # Close the gripper first, holding the init pose.
            self.target_pos = self.init_pos.clone()
            self.target_quat = self.init_quat.clone()
            gripper_closed = True
            if self.wait >= self.grasp_wait:
                self.state = self.MOVE_DOWN
                self.wait = 0.0

        elif self.state == self.MOVE_DOWN:
            # Gripper stays closed while we lower the EE and press the cloth into the ground.
            reached = self._slew_to(self.squeeze_pos, self.squeeze_speed)
            self.target_quat = self.init_quat.clone()
            gripper_closed = True
            if not reached:
                self.wait = 0.0
            elif self.wait >= self.squeeze_wait:
                self.state = self.DONE
                self.wait = 0.0

        elif self.state == self.DONE:
            self.target_pos = self.squeeze_pos.clone()
            self.target_quat = self.init_quat.clone()
            gripper_closed = True

        self.wait += self.dt
        return self.target_pos, self.target_quat, gripper_closed


def main():
    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        # The state machine drives the interactive-IK target, so enable it.
        env_cfg.interactive_ik = True
        # Disable the periodic episode time-out. Otherwise the env auto-resets the
        # robot to its default joint config every max_episode_length steps while the
        # state machine keeps commanding the (far) squeeze pose -- the stiff PD then
        # violently snaps the arm back, which looks like the arm "going crazy".
        env_cfg.episode_length_s = 1.0e9

        env = gym.make(args_cli.task, cfg=env_cfg)
        u = env.unwrapped
        env.reset()

        if not getattr(u, "_ik_available", False):
            raise RuntimeError(
                "Interactive IK is not available on this env; the state machine requires it."
                " Ensure the task supports interactive_ik."
            )

        device = u.device
        sim = u.sim
        actions = torch.zeros(env.action_space.shape, device=device)

        # Default (reset) EE pose = where the arm starts. body_link_pose_w is
        # [px, py, pz, qx, qy, qz, qw]. Set the IK target to it so the arm holds
        # the default pose (no snap); the state machine then moves it to INIT.
        ee_pose = wp.to_torch(u.robot.data.body_link_pose_w)[0, u._ee_ik_index]
        u._ee_tf = wp.transform(
            wp.vec3(float(ee_pose[0]), float(ee_pose[1]), float(ee_pose[2])),
            wp.quat(float(ee_pose[3]), float(ee_pose[4]), float(ee_pose[5]), float(ee_pose[6])),
        )
        start_pos = ee_pose[:3].clone()
        start_quat = ee_pose[3:7].clone()
        init_pos = torch.tensor(INIT_POS, device=device)
        init_quat = torch.tensor(INIT_QUAT, device=device)

        step_dt = env_cfg.sim.dt * env_cfg.decimation
        sm = ClothSqueezeSm(
            start_pos=start_pos,
            start_quat=start_quat,
            init_pos=init_pos,
            init_quat=init_quat,
            squeeze_z=args_cli.squeeze_z,
            dt=step_dt,
            move_speed=args_cli.move_speed,
            squeeze_speed=args_cli.squeeze_speed,
            settle_wait=args_cli.settle_wait,
            grasp_wait=args_cli.grasp_wait,
            squeeze_wait=args_cli.squeeze_wait,
        )
        _ang = 2.0 * float(torch.acos(torch.dot(start_quat / start_quat.norm(), init_quat / init_quat.norm()).abs().clamp(max=1.0)))
        print(
            f"[squeeze_cloth_sm] start_pos={start_pos.tolist()} init_pos={init_pos.tolist()}"
            f" squeeze_z={args_cli.squeeze_z} step_dt={step_dt:.4f}s"
        )
        print(
            f"[squeeze_cloth_sm] start_quat={[round(float(x),4) for x in start_quat]} "
            f"init_quat={[round(float(x),4) for x in init_quat]} angle={_ang*57.2958:.1f}deg"
        )

        prev_state = -1
        n_steps = 0
        while True:
            if sim.visualizers:
                if not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                    break
            elif n_steps >= args_cli.max_steps:
                break

            with torch.inference_mode():
                ee_pos = wp.to_torch(u.robot.data.body_pos_w)[0, u._ee_body_idx]
                target_pos, target_quat, gripper_closed = sm.compute()

                # Drive the env's interactive-IK target (pose) and gripper.
                u._ee_tf = wp.transform(
                    wp.vec3(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])),
                    wp.quat(float(target_quat[0]), float(target_quat[1]), float(target_quat[2]), float(target_quat[3])),
                )
                u._gripper_closed = bool(gripper_closed)

                nodes = wp.to_torch(u.cloth.data.nodal_pos_w)[0]  # (N, 3)
                d = torch.norm(nodes - ee_pos.unsqueeze(0), dim=-1)
                d_min = float(d.min())
                near = d < 0.08  # cloth nodes within 8 cm of the gripper ("grasp region")
                near_z = float(nodes[near, 2].mean()) if bool(near.any()) else float("nan")
                cloth_min_z = float(nodes[:, 2].min())
                cloth_max_z = float(nodes[:, 2].max())
                if sm.state != prev_state:
                    names = {0: "REST", 1: "MOVE_TO_INIT", 2: "GRASP", 3: "MOVE_DOWN", 4: "DONE"}
                    print(
                        f"[squeeze_cloth_sm] state -> {names[sm.state]:12s} | "
                        f"ee=({float(ee_pos[0]):.3f},{float(ee_pos[1]):.3f},{float(ee_pos[2]):.3f}) "
                        f"grip={'closed' if gripper_closed else 'open':6s} | "
                        f"d_min={d_min:.3f} near_z={near_z:.3f} "
                        f"cloth_minz={cloth_min_z:.3f} cloth_maxz={cloth_max_z:.3f}"
                    )
                    prev_state = sm.state
                elif n_steps % 20 == 0:
                    print(
                        f"[squeeze_cloth_sm] step {n_steps:4d} | ee_z={float(ee_pos[2]):.3f} "
                        f"d_min={d_min:.3f} near_z={near_z:.3f} "
                        f"cloth_minz={cloth_min_z:.3f} cloth_maxz={cloth_max_z:.3f}"
                    )

                env.step(actions)
                n_steps += 1

        env.close()


if __name__ == "__main__":
    main()
