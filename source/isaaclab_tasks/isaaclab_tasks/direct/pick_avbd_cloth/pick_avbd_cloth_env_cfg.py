# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Pick-AVBD-Cloth environment: Franka robot + cloth, all via unified AVBD solver."""

import importlib.util
import os.path

from isaaclab_contrib.deformable.newton_manager_cfg import NewtonModelCfg, VBDSolverCfg
from isaaclab_newton.physics import NewtonCfg
from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg
from isaaclab_visualizers.newton import NewtonVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg, preset

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

# Locate shirt USD from Newton package (defer import to avoid pxr before SimulationApp).
_newton_spec = importlib.util.find_spec("newton")
_SHIRT_USD = os.path.join(
    os.path.dirname(_newton_spec.origin),
    "examples",
    "assets",
    "unisex_shirt.usd",
)

# AVBD-specific Franka config: high stiffness for ALM joints
FRANKA_PANDA_AVBD_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()
FRANKA_PANDA_AVBD_CFG.actuators["panda_shoulder"].stiffness = 1e6
FRANKA_PANDA_AVBD_CFG.actuators["panda_shoulder"].damping = 0.01
FRANKA_PANDA_AVBD_CFG.actuators["panda_forearm"].stiffness = 1e6
FRANKA_PANDA_AVBD_CFG.actuators["panda_forearm"].damping = 0.01
FRANKA_PANDA_AVBD_CFG.actuators["panda_hand"].stiffness = 1e6
FRANKA_PANDA_AVBD_CFG.actuators["panda_hand"].damping = 0.1
FRANKA_PANDA_AVBD_CFG.spawn.rigid_props.disable_gravity = False


@configclass
class ClothAVBDNewtonCfg(NewtonCfg):
    """NewtonCfg subclass — distinct name ensures Kit is launched for USD spawning."""

    model_cfg: NewtonModelCfg | None = None
    """Global Newton model parameters applied after builder finalization."""


MODEL_CFG = NewtonModelCfg(
    soft_contact_ke=1e4,
    soft_contact_kd=1e-2,
    soft_contact_mu=1.5,
    shape_material_ke=1e4,
    shape_material_kd=1.0,
    shape_material_mu=1.5,
)


@configclass
class PickAVBDClothPhysicsCfg(PresetCfg):
    """Physics presets for AVBD cloth picking (unified rigid + cloth AVBD solver)."""

    default: ClothAVBDNewtonCfg = ClothAVBDNewtonCfg(
        solver_cfg=VBDSolverCfg(
            iterations=10,
            integrate_with_external_rigid_solver=False,
            particle_enable_self_contact=True,
            particle_self_contact_radius=2e-3,  # good for substeps=10
            particle_self_contact_margin=2e-3,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.0,
            particle_vertex_contact_buffer_size=16,
            particle_edge_contact_buffer_size=20,
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
        collision_cfg=NewtonCollisionPipelineCfg(
            soft_contact_margin=0.01,
        ),
        model_cfg=MODEL_CFG,
        num_substeps=10,
        use_cuda_graph=True,
    )

    newton: ClothAVBDNewtonCfg = default


@configclass
class PickAVBDClothEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # obs = joint_pos(7) + joint_vel(7) + cloth_centroid(3) = 17, act = 7
    action_space = 7
    observation_space = 17
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics=PickAVBDClothPhysicsCfg(),
        visualizer_cfgs=NewtonVisualizerCfg(),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # robot
    robot_cfg = preset(
        default=FRANKA_PANDA_AVBD_CFG.replace(prim_path="/World/envs/env_.*/Robot"),
        franka_high_pd=FRANKA_PANDA_AVBD_CFG.replace(prim_path="/World/envs/env_.*/Robot"),
    )

    # joint names to control (7 arm joints, excluding fingers)
    arm_joint_names = ["panda_joint[1-7]"]

    # control mode: "position" (PD, actions are joint position offsets [rad])
    #               "velocity" (P on velocity, actions are joint velocity targets [rad/s])
    control_mode: str = "position"

    # action scale applied to raw actions before use as targets
    action_scale = 0.5

    # cloth asset -- shirt mesh loaded from Newton assets
    cloth: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/cloth",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_SHIRT_USD,
            scale=(0.01, 0.01, 0.01),  # shirt USD vertices are in cm -> convert to meters
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)),
            physics_material=sim_utils.SurfaceDeformableBodyMaterialCfg(
                density=0.02,
                tri_ke=1e4,
                tri_ka=1e4,
                tri_kd=1.5e-6,
                edge_ke=0.5,
                edge_kd=1e-2,
                particle_radius=0.01,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.5, 1.25, 0.10),  # in front of robot, reachable height
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # disable rigid-body collision between robot and ground plane
    disable_robot_ground_collision: bool = False
    """When True, set the ground plane's collision group to 0 in Newton so the
    robot arm does not collide with the ground. Soft (particle) contacts are
    unaffected. Defaults to True."""

    # interactive IK: when True, spawn a draggable sphere and solve IK each step
    interactive_ik: bool = False

    # reward scales
    rew_scale_cloth_height = 5.0
    """Reward for lifting cloth centroid higher [per m]."""

    rew_scale_ee_cloth_dist = -2.0
    """Penalty for EE-to-cloth-centroid distance [per m]."""

    rew_scale_joint_vel = -0.01
    """Penalty for joint velocities."""
