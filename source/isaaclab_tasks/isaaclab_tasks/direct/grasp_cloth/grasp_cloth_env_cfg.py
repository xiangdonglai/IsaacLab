# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Grasp-Cloth environment: Franka + hanging cloth, coupled solver (MuJoCo + VBD)."""

import math

from isaaclab_contrib.deformable.newton_manager_cfg import CoupledSolverCfg, NewtonModelCfg, VBDSolverCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_visualizers.newton import NewtonVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.utils import PresetCfg, preset

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG


@configclass
class ClothNewtonCfg(NewtonCfg):
    """NewtonCfg subclass — distinct name ensures Kit is launched for USD spawning."""

    model_cfg: NewtonModelCfg | None = None


MODEL_CFG = NewtonModelCfg(
    soft_contact_ke=1e4,
    soft_contact_kd=1.0,
    soft_contact_mu=1.5,
    shape_material_ke=1e4,
    shape_material_kd=1.0,
    shape_material_mu=1.5,
)


@configclass
class GraspClothPhysicsCfg(PresetCfg):
    """Physics presets for coupled-solver cloth grasping (MuJoCo rigid + VBD cloth)."""

    default: ClothNewtonCfg = ClothNewtonCfg(
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
                iterations=10,
                particle_enable_self_contact=True,
                particle_self_contact_radius=2e-3,
                particle_self_contact_margin=2e-3,
                particle_topological_contact_filter_threshold=1,
                particle_rest_shape_contact_exclusion_radius=0.0,
                particle_vertex_contact_buffer_size=16,
                particle_edge_contact_buffer_size=20,
                particle_collision_detection_interval=-1,
                integrate_with_external_rigid_solver=True,
            ),
            soft_contact_margin=0.01,
            coupling_mode="one_way",
        ),
        model_cfg=MODEL_CFG,
        num_substeps=10,
        use_cuda_graph=True,
    )

    newton: ClothNewtonCfg = default


@configclass
class GraspClothEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 30.0
    action_space = 7
    observation_space = 17
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics=GraspClothPhysicsCfg(),
        visualizer_cfgs=NewtonVisualizerCfg(),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=4.0,
        replicate_physics=True,
    )

    robot_cfg = preset(
        default=FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot"),
        franka_high_pd=FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot"),
    )

    arm_joint_names = ["panda_joint[1-7]"]
    control_mode: str = "position"
    action_scale = 0.5

    # Hanging cloth — a square mesh with two top corners pinned
    cloth: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/cloth",
        spawn=sim_utils.MeshSquareCfg(
            size=0.3,
            resolution=(15, 15),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.8)),
            physics_material=sim_utils.SurfaceDeformableBodyMaterialCfg(
                density=0.02,
                tri_ke=1e4,
                tri_ka=1e4,
                tri_kd=1.5e-6,
                edge_ke=0.05,
                edge_kd=1e-2,
                particle_radius=0.01,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.75, 0.0, 0.66),
            rot=(-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
        ),
    )

    cloth_resolution: int = 15

    disable_robot_ground_collision: bool = True
    interactive_ik: bool = False

    rew_scale_cloth_height = 5.0
    rew_scale_ee_cloth_dist = -2.0
    rew_scale_joint_vel = -0.01
