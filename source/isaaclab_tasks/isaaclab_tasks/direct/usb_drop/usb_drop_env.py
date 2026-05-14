# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USB drop environment: articulated USB dropped onto the floor using ArticulationCfg + UrdfFileCfg."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .usb_drop_env_cfg import USBDropEnvCfg

logger = logging.getLogger(__name__)


class USBDropEnv(DirectRLEnv):
    """USB drop environment: articulated USB falls to the ground."""

    cfg: USBDropEnvCfg

    def __init__(self, cfg: USBDropEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        logger.info("USBDropEnv initialized")

    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.usb = Articulation(self.cfg.usb)
        self.scene.articulations["usb"] = self.usb

        self.scene.clone_environments(copy_from_source=False)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        pass

    def _apply_action(self) -> None:
        pass

    def _get_observations(self) -> dict:
        obs = torch.zeros(self.num_envs, 1, device=self.device)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == 0:
            return
        super()._reset_idx(env_ids)

        # Reset root pose (position + orientation)
        root_pose = wp.to_torch(self.usb.data.default_root_pose)[env_ids].clone()
        self.usb.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        # Reset root velocity
        root_vel = wp.to_torch(self.usb.data.default_root_vel)[env_ids].clone()
        self.usb.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        # Reset joint positions and velocities
        joint_pos = wp.to_torch(self.usb.data.default_joint_pos)[env_ids].clone()
        joint_vel = wp.to_torch(self.usb.data.default_joint_vel)[env_ids].clone()
        self.usb.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.usb.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
        # Clear internal buffers
        self.usb.reset(env_ids)
        logger.info(f"USB reset: joint_pos={joint_pos}, root_pos={root_pose[:, :3]}")
