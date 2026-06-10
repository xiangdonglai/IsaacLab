# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted pick state machine for the AVBD/RVBD cloth scenes.

Drives the interactive-IK target of ``Isaac-Pick-AVBD-Cloth-Direct-v0`` /
``Isaac-Pick-RVBD-Cloth-Direct-v0`` through a simple sequence:

1. REST       -- hold the initial (home) EE pose, gripper open.
2. MOVE_DOWN  -- lower the EE straight down to ``--grasp_z`` (default 0.0981), gripper open.
3. GRASP      -- hold at the low pose and close the gripper.
4. MOVE_UP    -- raise the EE back to the home pose, gripper closed.
5. DONE       -- hold the home pose, gripper closed.

The scene's interactive IK is single-environment, so this runs with num_envs=1.
The state machine only changes the EE z; x/y and orientation are held at the home pose.

.. code-block:: bash

    ./isaaclab.sh -p scripts/environments/state_machine/pick_cloth_sm.py \
        --task Isaac-Pick-RVBD-Cloth-Direct-v0 --num_envs 1 --visualizer newton presets=newton
"""

import argparse
import sys

import gymnasium as gym
import torch
import warp as wp

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

parser = argparse.ArgumentParser(description="Scripted pick state machine for the cloth scenes.")
parser.add_argument("--task", type=str, default="Isaac-Pick-RVBD-Cloth-Direct-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (IK is single-env).")
parser.add_argument("--grasp_z", type=float, default=0.0981, help="World z [m] the EE descends to before grasping.")
parser.add_argument("--approach_speed", type=float, default=0.3, help="EE target slew speed [m/s] for descent/ascent.")
parser.add_argument("--settle_wait", type=float, default=0.5, help="Wait [s] after a move completes before advancing.")
parser.add_argument("--grasp_wait", type=float, default=0.6, help="Wait [s] for the gripper to close.")
parser.add_argument("--max_steps", type=int, default=100000, help="Max env steps when no visualizer is open.")
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args


class ClothPickSm:
    """Single-environment task-space pick state machine.

    Outputs a desired EE target position (z slewed toward the active goal) and a
    gripper-closed flag. x/y and orientation are held at the home pose.
    """

    REST = 0
    MOVE_DOWN = 1
    GRASP = 2
    MOVE_UP = 3
    DONE = 4

    def __init__(self, home_pos, grasp_z, dt, approach_speed, settle_wait, grasp_wait, rest_wait=0.3):
        self.home_pos = home_pos.clone()
        self.grasp_pos = home_pos.clone()
        self.grasp_pos[2] = grasp_z
        self.dt = float(dt)
        self.approach_speed = float(approach_speed)
        self.settle_wait = float(settle_wait)
        self.grasp_wait = float(grasp_wait)
        self.rest_wait = float(rest_wait)

        self.state = self.REST
        self.wait = 0.0
        self.target = home_pos.clone()  # current commanded target position (slewed)

    def _slew_to(self, goal) -> bool:
        """Move the commanded target toward ``goal`` at ``approach_speed``. Returns True if reached."""
        delta = goal - self.target
        dist = float(torch.norm(delta))
        max_step = self.approach_speed * self.dt
        if dist <= max_step or dist < 1e-9:
            self.target = goal.clone()
            return True
        self.target = self.target + delta * (max_step / dist)
        return False

    def compute(self, ee_pos):
        """Advance the state machine one tick; return (target_pos, gripper_closed)."""
        gripper_closed = False

        if self.state == self.REST:
            self.target = self.home_pos.clone()
            if self.wait >= self.rest_wait:
                self.state = self.MOVE_DOWN
                self.wait = 0.0

        elif self.state == self.MOVE_DOWN:
            reached = self._slew_to(self.grasp_pos)
            if not reached:
                self.wait = 0.0  # only count settle time once the slew completes
            elif self.wait >= self.settle_wait:
                self.state = self.GRASP
                self.wait = 0.0

        elif self.state == self.GRASP:
            self.target = self.grasp_pos.clone()
            gripper_closed = True
            if self.wait >= self.grasp_wait:
                self.state = self.MOVE_UP
                self.wait = 0.0

        elif self.state == self.MOVE_UP:
            reached = self._slew_to(self.home_pos)
            gripper_closed = True
            if not reached:
                self.wait = 0.0
            elif self.wait >= self.settle_wait:
                self.state = self.DONE
                self.wait = 0.0

        elif self.state == self.DONE:
            self.target = self.home_pos.clone()
            gripper_closed = True

        self.wait += self.dt
        return self.target, gripper_closed


def main():
    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        # The state machine drives the interactive-IK target, so enable it.
        env_cfg.interactive_ik = True

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

        # Home pose = the env's initial IK target (orientation held throughout).
        home_pos = torch.tensor(
            [float(x) for x in wp.transform_get_translation(u._ee_tf)], device=device
        )
        home_quat = wp.transform_get_rotation(u._ee_tf)

        step_dt = env_cfg.sim.dt * env_cfg.decimation
        sm = ClothPickSm(
            home_pos=home_pos,
            grasp_z=args_cli.grasp_z,
            dt=step_dt,
            approach_speed=args_cli.approach_speed,
            settle_wait=args_cli.settle_wait,
            grasp_wait=args_cli.grasp_wait,
        )
        print(
            f"[pick_cloth_sm] home_pos={home_pos.tolist()} grasp_z={args_cli.grasp_z}"
            f" step_dt={step_dt:.4f}s"
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
                target_pos, gripper_closed = sm.compute(ee_pos)

                # Drive the env's interactive-IK target and gripper.
                u._ee_tf = wp.transform(
                    wp.vec3(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])),
                    home_quat,
                )
                u._gripper_closed = bool(gripper_closed)

                nodes = wp.to_torch(u.cloth.data.nodal_pos_w)[0]  # (N, 3)
                d = torch.norm(nodes - ee_pos.unsqueeze(0), dim=-1)
                d_min = float(d.min())
                near = d < 0.08  # cloth nodes within 8 cm of the gripper ("grasp region")
                near_z = float(nodes[near, 2].mean()) if bool(near.any()) else float("nan")
                cloth_max_z = float(nodes[:, 2].max())
                if sm.state != prev_state:
                    names = {0: "REST", 1: "MOVE_DOWN", 2: "GRASP", 3: "MOVE_UP", 4: "DONE"}
                    print(
                        f"[pick_cloth_sm] state -> {names[sm.state]:9s} | "
                        f"ee=({float(ee_pos[0]):.3f},{float(ee_pos[1]):.3f},{float(ee_pos[2]):.3f}) "
                        f"grip={'closed' if gripper_closed else 'open':6s} | "
                        f"d_min={d_min:.3f} near_z={near_z:.3f} cloth_maxz={cloth_max_z:.3f}"
                    )
                    prev_state = sm.state
                elif n_steps % 20 == 0:
                    print(
                        f"[pick_cloth_sm] step {n_steps:4d} | ee_z={float(ee_pos[2]):.3f} "
                        f"d_min={d_min:.3f} near_z={near_z:.3f} cloth_maxz={cloth_max_z:.3f}"
                    )

                env.step(actions)
                n_steps += 1

        env.close()


if __name__ == "__main__":
    main()
