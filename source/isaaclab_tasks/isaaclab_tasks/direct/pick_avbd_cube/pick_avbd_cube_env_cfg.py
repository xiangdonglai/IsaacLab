# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Pick-AVBD-Cube environment: Franka robot + deformable cube with unified AVBD solver."""

from isaaclab_contrib.deformable.newton_manager_cfg import NewtonModelCfg, VBDSolverCfg
from isaaclab_newton.physics import NewtonCfg
from isaaclab_visualizers.newton import NewtonVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg, preset

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG, FRANKA_PANDA_HIGH_PD_CFG

# AVBD-specific Franka config: high position stiffness, low damping.
# AVBD damping formula is drive_d = kd * ke, and the implicit Hessian is
# H = ke + kd * ke / dt.  With small substep dt (~1/600), even moderate kd
# produces massive velocity damping that locks joints. Use kd ~ 0.01 so
# that H ≈ ke + kd * ke / dt ≈ 400 + 2400 = 2800, allowing drives to
# converge within ~10 iterations.
FRANKA_PANDA_AVBD_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()
FRANKA_PANDA_AVBD_CFG.actuators["panda_shoulder"].stiffness = 1e4
FRANKA_PANDA_AVBD_CFG.actuators["panda_shoulder"].damping = 1.0
FRANKA_PANDA_AVBD_CFG.actuators["panda_forearm"].stiffness = 1e4
FRANKA_PANDA_AVBD_CFG.actuators["panda_forearm"].damping = 1.0
FRANKA_PANDA_AVBD_CFG.actuators["panda_hand"].stiffness = 2e4
FRANKA_PANDA_AVBD_CFG.actuators["panda_hand"].damping = 1.0


@configclass
class DeformableNewtonCfg(NewtonCfg):
    """NewtonCfg extended with model-level contact parameters for deformable objects.

    Uses a distinct class name so that ``_is_kitless_physics`` does not
    match it, ensuring Kit is launched for USD deformable spawning.
    """

    model_cfg: NewtonModelCfg | None = None
    """Global Newton model parameters applied after builder finalization."""


MODEL_CFG = NewtonModelCfg(
    soft_contact_ke=1e4,
    soft_contact_kd=1e-2,
    soft_contact_mu=1.0,
)


@configclass
class PickAVBDCubePhysicsCfg(PresetCfg):
    """Physics presets for the Pick-AVBD-Cube environment.

    Uses the unified AVBD solver: VBD handles both particles and articulated
    rigid bodies in a single solver (no external rigid solver).

    Presets:
        - ``default`` / ``newton``: Unified AVBD solver (recommended).
    """

    default: DeformableNewtonCfg = DeformableNewtonCfg(
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

    newton: DeformableNewtonCfg = default


@configclass
class PickAVBDCubeEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # obs = joint_pos(7) + joint_vel(7) + cube_centroid(3) = 17, act = 7
    action_space = 7
    observation_space = 17
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics=PickAVBDCubePhysicsCfg(),
        visualizer_cfgs=NewtonVisualizerCfg(),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # robot — AVBD needs low damping (see FRANKA_PANDA_AVBD_CFG comment above)
    robot_cfg = preset(
        default=FRANKA_PANDA_AVBD_CFG.replace(prim_path="/World/envs/env_.*/Robot"),
        franka_high_pd=FRANKA_PANDA_AVBD_CFG.replace(prim_path="/World/envs/env_.*/Robot"),
    )

    # joint names to control (7 arm joints, excluding fingers)
    arm_joint_names = ["panda_joint[1-7]"]

    # control mode: "position" or "velocity"
    control_mode: str = "position"

    # action scale applied to raw actions before use as targets
    action_scale = 0.5

    # deformable cube (VBD)
    cube: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.05, 0.05, 0.05),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                density=500.0,
                youngs_modulus=2.5e5,
                poissons_ratio=0.25,
                particle_radius=0.005,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.25, 0.0, 0.05),
        ),
    )

    # disable rigid-body collision between robot and ground plane
    disable_robot_ground_collision: bool = True
    """When True, set the ground plane's collision group to 0 in Newton so the
    robot arm does not collide with the ground. Soft (particle) contacts are
    unaffected. Defaults to True."""

    # interactive IK: when True, spawn a draggable sphere and solve IK each step
    interactive_ik: bool = False

    # reward scales
    rew_scale_cube_height = 5.0
    """Reward for lifting cube centroid higher [per m]."""

    rew_scale_ee_cube_dist = -2.0
    """Penalty for EE-to-cube-centroid distance [per m]."""

    rew_scale_joint_vel = -0.01
    """Penalty for joint velocities."""
