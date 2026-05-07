# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Pick-AVBD-Rigid-Cube: Franka + rigid cube, all via AVBD solver."""

from isaaclab_contrib.deformable.newton_manager_cfg import NewtonModelCfg, VBDSolverCfg
from isaaclab_newton.physics import NewtonCfg
from isaaclab_visualizers.newton import NewtonVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg, preset

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

# AVBD-specific Franka config
FRANKA_PANDA_AVBD_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()
FRANKA_PANDA_AVBD_CFG.actuators["panda_shoulder"].stiffness = 1e4
FRANKA_PANDA_AVBD_CFG.actuators["panda_shoulder"].damping = 1.0
FRANKA_PANDA_AVBD_CFG.actuators["panda_forearm"].stiffness = 1e4
FRANKA_PANDA_AVBD_CFG.actuators["panda_forearm"].damping = 1.0
FRANKA_PANDA_AVBD_CFG.actuators["panda_hand"].stiffness = 2e3
FRANKA_PANDA_AVBD_CFG.actuators["panda_hand"].damping = 1.0
FRANKA_PANDA_AVBD_CFG.spawn.rigid_props.disable_gravity = False


@configclass
class RigidAVBDNewtonCfg(NewtonCfg):
    """NewtonCfg subclass — distinct name ensures Kit is launched for USD spawning."""

    model_cfg: NewtonModelCfg | None = None
    """Global Newton model parameters applied after builder finalization."""


MODEL_CFG = NewtonModelCfg(
    shape_material_ke=1e3,
    shape_material_kd=1.0,
    shape_material_mu=0.5,
)


@configclass
class PickAVBDRigidCubePhysicsCfg(PresetCfg):
    """Physics presets: pure rigid-body scene using AVBD solver."""

    default: RigidAVBDNewtonCfg = RigidAVBDNewtonCfg(
        solver_cfg=VBDSolverCfg(
            iterations=10,
            integrate_with_external_rigid_solver=False,
            particle_enable_self_contact=False,
            particle_collision_detection_interval=-1,
            rigid_contact_k_start=1.0e2,
            rigid_avbd_beta=1.0e5,
            rigid_avbd_gamma=0.99,
            rigid_joint_linear_k_start=1.0e4,
            rigid_joint_angular_k_start=1.0e1,
            rigid_joint_linear_ke=1.0e9,
            rigid_joint_angular_ke=1.0e9,
            rigid_joint_linear_kd=1.0e-2,
            rigid_joint_angular_kd=0.0,
        ),
        model_cfg=MODEL_CFG,
        num_substeps=10,
        use_cuda_graph=True,
    )

    newton: RigidAVBDNewtonCfg = default


@configclass
class PickAVBDRigidCubeEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 5.0
    action_space = 7
    observation_space = 17
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics=PickAVBDRigidCubePhysicsCfg(),
        visualizer_cfgs=NewtonVisualizerCfg(),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=4.0,
        replicate_physics=True,
    )

    robot_cfg = preset(
        default=FRANKA_PANDA_AVBD_CFG.replace(prim_path="/World/envs/env_.*/Robot"),
        franka_high_pd=FRANKA_PANDA_AVBD_CFG.replace(prim_path="/World/envs/env_.*/Robot"),
    )

    arm_joint_names = ["panda_joint[1-7]"]
    control_mode: str = "position"
    action_scale = 0.5

    # Rigid cube (100g, ~same mass as finger bodies)
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.25, 0.0, 0.05),
        ),
    )

    disable_robot_ground_collision: bool = True
    interactive_ik: bool = False

    rew_scale_cube_height = 5.0
    rew_scale_ee_cube_dist = -2.0
    rew_scale_joint_vel = -0.01
