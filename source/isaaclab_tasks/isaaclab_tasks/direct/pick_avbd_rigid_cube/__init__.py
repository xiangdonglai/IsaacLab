# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Pick-AVBD-Rigid-Cube-Direct-v0",
    entry_point=f"{__name__}.pick_avbd_rigid_cube_env:PickAVBDRigidCubeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_avbd_rigid_cube_env_cfg:PickAVBDRigidCubeEnvCfg",
    },
)
