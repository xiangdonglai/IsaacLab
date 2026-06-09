# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Pick-Proxy-Cloth environment: Franka + cloth via lagged-impulse proxy coupling.

Pairs MJWarp (rigid Franka) with VBD (cloth) through the proxy coupling layer
in :mod:`isaaclab_contrib.coupling`. The Franka hand and fingers are exposed
as proxy bodies in the cloth solver's view so the cloth detects contact with
them and feeds back lagged impulses to the rigid solver.

Compare with :mod:`isaaclab_tasks.direct.pick_avbd_cloth`, which runs the same
scene through monolithic AVBD (no MJWarp, rigid Franka integrated by VBD).
"""

from isaaclab_contrib.coupling import CoupledProxySolverCfg
from isaaclab_contrib.deformable.newton_manager_cfg import CoupledNewtonCfg, NewtonModelCfg, VBDSolverCfg
from isaaclab_newton.physics import MJWarpSolverCfg
from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg
from isaaclab_newton.sim.spawners.materials.physics_materials_cfg import NewtonSurfaceDeformableBodyMaterialCfg
from isaaclab_visualizers.newton import NewtonVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg, preset

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG

# Path to the unisex shirt USD shipped with the Newton package.
# Resolved lazily at runtime in the env's _setup_scene to avoid importing pxr
# or newton before isaacsim's SimulationApp starts.
_SHIRT_USD_PLACEHOLDER = "__SHIRT_USD_PLACEHOLDER__"

FRANKA_PANDA_PROXY_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()
FRANKA_PANDA_PROXY_CFG.actuators["panda_shoulder"].stiffness = 4000.0
FRANKA_PANDA_PROXY_CFG.actuators["panda_shoulder"].damping = 400.0
FRANKA_PANDA_PROXY_CFG.actuators["panda_forearm"].stiffness = 4000.0
FRANKA_PANDA_PROXY_CFG.actuators["panda_forearm"].damping = 400.0
FRANKA_PANDA_PROXY_CFG.actuators["panda_hand"].stiffness = 4000.0
FRANKA_PANDA_PROXY_CFG.actuators["panda_hand"].damping = 400.0
FRANKA_PANDA_PROXY_CFG.spawn.rigid_props.disable_gravity = True


# Match franka_soft_env_cfg's MODEL_CFG values. The AVBD task used kd=1.0,
# mu=1.5 — those work for AVBD's unified contact handling but destabilize
# MJWarp's rigid-rigid contact damping (kd=1.0 is 100,000× franka_soft's 1e-5).
MODEL_CFG = NewtonModelCfg(
    soft_contact_ke=1e4,
    soft_contact_kd=1e-5,
    soft_contact_mu=5.0,
    shape_material_ke=4e4,
    shape_material_kd=1e-5,
    shape_material_mu=5.0,
)


@configclass
class PickProxyClothPhysicsCfg(PresetCfg):
    """Physics presets for proxy-coupled cloth picking (MJWarp rigid + VBD cloth)."""

    default: CoupledNewtonCfg = CoupledNewtonCfg(
        solver_cfg=CoupledProxySolverCfg(
            src_solver_cfg=MJWarpSolverCfg(
                cone="elliptic",
                ls_parallel=True,
                ls_iterations=20,
                integrator="implicitfast",
            ),
            dst_solver_cfg=VBDSolverCfg(
                iterations=10,
                particle_enable_self_contact=True,
                particle_self_contact_radius=2e-3,
                particle_self_contact_margin=2e-3,
                particle_topological_contact_filter_threshold=1,
                particle_rest_shape_contact_exclusion_radius=0.0,
                particle_vertex_contact_buffer_size=16,
                particle_edge_contact_buffer_size=20,
                particle_collision_detection_interval=-1,
            ),
            src_bodies=["/World/envs/env_.*/Robot"],
            proxy_bodies=[
                "/World/envs/env_.*/Robot/panda_hand",
                "/World/envs/env_.*/Robot/panda_(left|right)finger",
            ],
            proxy_collide_interval=5,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            soft_contact_margin=0.01,
        ),
        model_cfg=MODEL_CFG,
        num_substeps=10,
        use_cuda_graph=True,
    )

    newton: CoupledNewtonCfg = default


@configclass
class PickProxyClothEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 1
    episode_length_s = 5.0
    # obs = arm_joint_pos(7) + arm_joint_vel(7) + cloth_centroid(3) = 17
    action_space = 7
    observation_space = 17
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics=PickProxyClothPhysicsCfg(),
        visualizer_cfgs=NewtonVisualizerCfg(),
        # Keep default gravity (0,0,-9.81) so the cloth drapes; the arm
        # actuator overrides in __post_init__ hold pose against gravity.
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # robot — base Franka cfg; actuator overrides applied in __post_init__
    # exactly like franka_soft_env_cfg's _FrankaSoftSceneCfg.
    robot_cfg = preset(
        default=FRANKA_PANDA_PROXY_CFG.replace(prim_path="/World/envs/env_.*/Robot"),
        franka=FRANKA_PANDA_PROXY_CFG.replace(prim_path="/World/envs/env_.*/Robot"),
    )

    # joint names to control (7 arm joints, excluding fingers)
    arm_joint_names = ["panda_joint[1-7]"]

    # control mode: "position" or "velocity"
    control_mode: str = "position"

    # action scale applied to raw actions before use as targets
    action_scale = 0.5

    # cloth asset -- shirt mesh loaded from Newton assets.  usd_path is a
    # placeholder resolved at runtime in the env's _setup_scene (avoids
    # importing newton / pxr before SimulationApp starts).
    cloth: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/cloth",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_SHIRT_USD_PLACEHOLDER,
            scale=(0.01, 0.01, 0.01),  # shirt USD vertices are in cm; convert to meters
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
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
            pos=(0.35, 1.20, 0.10),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # disable rigid-body collision between robot and ground plane
    disable_robot_ground_collision: bool = True

    # interactive IK: when True, spawn a draggable sphere and solve IK each step
    interactive_ik: bool = False

    # reward scales
    rew_scale_cloth_height = 5.0
    """Reward for lifting cloth centroid higher [per m]."""

    rew_scale_ee_cloth_dist = -2.0
    """Penalty for EE-to-cloth-centroid distance [per m]."""

    rew_scale_joint_vel = -0.01
    """Penalty for joint velocities."""

    def __post_init__(self) -> None:
        # Match franka_soft_env_cfg actuator overrides — applied here (not at
        # copy time) to mirror that file's structure as closely as possible.
        # Gravity is enabled on every body (overrides the FRANKA_PANDA_HIGH_PD_CFG
        # default of disable_gravity=True). MJWarp's actuator PD holds the arm
        # against gravity through joint_target_pos.
        robot = self.robot_cfg.default if hasattr(self.robot_cfg, "default") else self.robot_cfg
        robot.spawn.rigid_props.disable_gravity = False
        robot.actuators["panda_hand"].effort_limit_sim = 500.0
        robot.actuators["panda_hand"].stiffness = 1.0e4
        robot.actuators["panda_hand"].damping = 100.0
