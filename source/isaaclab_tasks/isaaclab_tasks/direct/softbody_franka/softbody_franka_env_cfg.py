# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Softbody-Franka environment.

Mirrors ``newton/examples/softbody/example_softbody_franka.py`` as closely as
possible: same scene geometry, contact parameters, keyframe sequence, and IK
settings.  The only structural difference is that the IsaacLab coupled solver
uses PD-controlled dynamics (MuJoCo Warp) instead of kinematic integration.
"""

import numpy as np

from isaaclab_contrib.deformable.newton_manager_cfg import CoupledSolverCfg, NewtonModelCfg, VBDSolverCfg
from isaaclab_newton.physics import FeatherstoneSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_visualizers.newton import NewtonVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg, UrdfFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg, preset


@configclass
class DeformableNewtonCfg(NewtonCfg):
    """NewtonCfg extended with model-level contact parameters for deformable objects."""

    model_cfg: NewtonModelCfg | None = None


# ---------------------------------------------------------------------------
# Contact parameters — identical to the Newton example.
# ---------------------------------------------------------------------------
MODEL_CFG = NewtonModelCfg(
    soft_contact_ke=2e6,
    soft_contact_kd=1e-7,
    soft_contact_mu=0.5,
    shape_material_ke=2e6,
    shape_material_kd=1e-7,
    shape_material_mu=1.5,
)


# ---------------------------------------------------------------------------
# Physics presets
# ---------------------------------------------------------------------------
@configclass
class SoftbodyFrankaPhysicsCfg(PresetCfg):
    """Physics presets for the Softbody-Franka environment."""

    default: DeformableNewtonCfg = DeformableNewtonCfg(
        solver_cfg=CoupledSolverCfg(
            rigid_solver_cfg=MJWarpSolverCfg(
                njmax=40,
                nconmax=20,
                ls_iterations=20,
                cone="pyramidal",
                impratio=1,
                ls_parallel=False,
                integrator="implicitfast",
                ccd_iterations=100,
            ),
            vbd_cfg=VBDSolverCfg(
                iterations=5,
                integrate_with_external_rigid_solver=True,
                particle_enable_self_contact=False,
                particle_collision_detection_interval=-1,
            ),
            soft_contact_margin=0.01,
            coupling_mode="kinematic",
        ),
        model_cfg=MODEL_CFG,
        num_substeps=10,
        use_cuda_graph=True,
    )

    newton: DeformableNewtonCfg = default
    newton_mjwarp: DeformableNewtonCfg = default

    newton_featherstone: DeformableNewtonCfg = DeformableNewtonCfg(
        solver_cfg=CoupledSolverCfg(
            rigid_solver_cfg=FeatherstoneSolverCfg(),
            vbd_cfg=VBDSolverCfg(
                iterations=5,
                integrate_with_external_rigid_solver=True,
                particle_enable_self_contact=False,
                particle_collision_detection_interval=-1,
            ),
            soft_contact_margin=0.01,
            coupling_mode="one_way",
        ),
        model_cfg=MODEL_CFG,
        num_substeps=30,
        use_cuda_graph=True,
    )


# ---------------------------------------------------------------------------
# Keyframes — verbatim from the Newton example (world-space coordinates).
# Robot base is at (-0.5, -0.5, -0.1), duck at (0, -0.5, 0.23).
# IK uses link_offset=(0,0,0.22) targeting the fingertip point.
# ---------------------------------------------------------------------------
_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = 0.5

# fmt: off
KEYFRAMES = np.array(
    [
        # [duration_s, px, py, pz, qx, qy, qz, qw, gripper_activation]
        # Descend z=0.22: slightly below duck center (z=0.23) so fingers surround it.
        [2.5, -0.005, -0.5, 0.35, 1.0, 0.0, 0.0, 0.0, _GRIPPER_OPEN],   # approach
        [2.0, -0.005, -0.5, 0.22, 1.0, 0.0, 0.0, 0.0, _GRIPPER_OPEN],   # descend
        [2.5, -0.005, -0.5, 0.22, 1.0, 0.0, 0.0, 0.0, _GRIPPER_CLOSE],  # pinch
        [2.0, -0.005, -0.5, 0.35, 1.0, 0.0, 0.0, 0.0, _GRIPPER_CLOSE],  # lift
        [2.0, -0.005, -0.5, 0.35, 1.0, 0.0, 0.0, 0.0, _GRIPPER_CLOSE],  # hold
        [2.0, -0.005, -0.5, 0.22, 1.0, 0.0, 0.0, 0.0, _GRIPPER_CLOSE],  # place
        [1.0, -0.005, -0.5, 0.22, 1.0, 0.0, 0.0, 0.0, _GRIPPER_OPEN],   # release
        [2.0, -0.005, -0.5, 0.35, 1.0, 0.0, 0.0, 0.0, _GRIPPER_OPEN],   # retract
    ],
    dtype=np.float32,
)
# fmt: on


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------
@configclass
class SoftbodyFrankaEnvCfg(DirectRLEnvCfg):
    decimation = 1
    episode_length_s = 20.0
    action_space = 7
    observation_space = 21
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=decimation,
        physics=SoftbodyFrankaPhysicsCfg(),
        visualizer_cfgs=NewtonVisualizerCfg(),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # FR3 Franka loaded from the same URDF as the Newton example.
    # usd_path placeholder is resolved at runtime via newton.utils.download_asset.
    # Robot at (-0.5, -0.5, -0.1), initial joint config [0, 0, 0, -1.597, 0, 2.531, 0].
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UrdfFileCfg(
            asset_path="__FR3_URDF_PLACEHOLDER__",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            # Set non-zero stiffness in the USD joint drives so Newton infers
            # JointTargetMode.POSITION (not EFFORT). Without this, MuJoCo Warp
            # doesn't create PD actuators and the robot ignores position targets.
            joint_drive=UrdfFileCfg.JointDriveCfg(
                gains=UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=2000.0,
                    damping=80.0,
                ),
            ),
            fix_base=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.5, -0.5, -0.1),
            joint_pos={
                "fr3_joint1": 0.0,
                "fr3_joint2": 0.0,
                "fr3_joint3": 0.0,
                "fr3_joint4": -1.59695,
                "fr3_joint5": 0.0,
                "fr3_joint6": 2.5307,
                "fr3_joint7": 0.0,
                "fr3_finger_joint.*": 0.04,
            },
        ),
        actuators={
            "fr3_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[1-4]"],
                effort_limit_sim=500.0,
                stiffness=2000.0,
                damping=80.0,
            ),
            "fr3_forearm": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[5-7]"],
                effort_limit_sim=100.0,
                stiffness=2000.0,
                damping=80.0,
            ),
            "fr3_hand": ImplicitActuatorCfg(
                joint_names_expr=["fr3_finger_joint.*"],
                effort_limit_sim=200.0,
                stiffness=2000.0,
                damping=100.0,
            ),
        },
    )

    arm_joint_names = ["fr3_joint[1-7]"]

    # When True, directly write IK-solved joint positions to sim (kinematic,
    # like the Newton example). When False, use PD position targets.
    kinematic_control: bool = True

    # Duck at (0.0, -0.5, 0.23) — same as Newton example.
    soft_body: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/soft_body",
        spawn=UsdFileCfg(
            usd_path="__DUCK_PLACEHOLDER__",
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                density=100.0,
                youngs_modulus=2.5e6,
                poissons_ratio=0.25,
                particle_radius=0.005,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.0, -0.5, 0.23),
        ),
    )

    # Table: center (0, -0.5, 0.1), half-extents (0.4, 0.4, 0.1) → top at z=0.2.
    table_size: tuple[float, float, float] = (0.8, 0.8, 0.2)
    table_pos: tuple[float, float, float] = (0.0, -0.5, 0.1)

    # Disable rigid-body collision for ground and table so the PD-controlled
    # robot doesn't explode from contact forces (the Newton example controls
    # the robot kinematically so this isn't an issue there).
    disable_robot_ground_collision: bool = True
