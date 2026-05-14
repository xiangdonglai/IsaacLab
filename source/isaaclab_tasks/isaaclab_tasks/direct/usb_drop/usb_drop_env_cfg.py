# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the USB drop environment."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_visualizers.newton import NewtonVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg

_USB_USD = "/mnt/nvme1/Workspace/robotics/MonocularRigidDecomposition/partnet_mobility__USB__101886__joint_0_bg__view_0__video/articulated.usda"


@configclass
class USBNewtonCfg(NewtonCfg):
    """Subclass to ensure Kit is launched."""

    pass


@configclass
class USBDropPhysicsCfg(PresetCfg):

    default: USBNewtonCfg = USBNewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=40,
            nconmax=40,
            ls_iterations=20,
            cone="pyramidal",
            impratio=1,
            ls_parallel=False,
            integrator="implicitfast",
        ),
        num_substeps=4,
        use_cuda_graph=True,
    )

    newton: USBNewtonCfg = default


@configclass
class USBDropEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 10.0
    action_space = 1
    observation_space = 1
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics=USBDropPhysicsCfg(),
        visualizer_cfgs=NewtonVisualizerCfg(),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=4.0,
        replicate_physics=True,
    )

    usb: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/USB",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_USB_USD,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            joint_pos={"joint_0": 1.5708},
        ),
        actuators={},
    )
