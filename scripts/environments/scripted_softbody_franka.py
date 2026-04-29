# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run the Softbody-Franka demo with scripted keyframe motion.

Usage::

    ./isaaclab.sh -p scripts/environments/scripted_softbody_franka.py --num_envs 1 --visualizer newton presets=newton

"""

import argparse
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config

TASK = "Isaac-Softbody-Franka-Direct-v0"

parser = argparse.ArgumentParser(description="Scripted Softbody-Franka demo.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default=TASK, help="Task name.")
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args


def main():
    torch.manual_seed(42)

    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        env = gym.make(args_cli.task, cfg=env_cfg)

        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        env.reset()

        sim = env.unwrapped.sim
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

        while True:
            if sim.visualizers:
                if not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                    break
            with torch.inference_mode():
                env.step(actions)

        env.close()


if __name__ == "__main__":
    main()
