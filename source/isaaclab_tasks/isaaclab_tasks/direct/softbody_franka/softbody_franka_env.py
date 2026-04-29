# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Softbody-Franka environment: scripted Franka grasping a deformable object using a coupled solver."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch
import warp as wp

from isaaclab_contrib.deformable import register_hooks as _register_deformable_hooks

_register_deformable_hooks()

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.assets.deformable_object import DeformableObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.shapes import CuboidCfg, spawn_cuboid

from .softbody_franka_env_cfg import KEYFRAMES, SoftbodyFrankaEnvCfg

logger = logging.getLogger(__name__)


class SoftbodyFrankaEnv(DirectRLEnv):
    """Scripted Franka grasping demo with a deformable object and coupled rigid+VBD solver.

    The robot follows a predefined keyframe sequence (approach, descend, pinch, lift, hold,
    place, release, retract) using Newton's GPU IK solver to convert end-effector pose targets
    into joint position commands.
    """

    cfg: SoftbodyFrankaEnvCfg

    def __init__(self, cfg: SoftbodyFrankaEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._arm_joint_idx, _ = self.robot.find_joints(self.cfg.arm_joint_names)
        self._finger_joint_idx, _ = self.robot.find_joints(["fr3_finger_joint1", "fr3_finger_joint2"])

        # Set initial joint position targets to default config so PD doesn't
        # drive to zeros before IK is ready.
        default_pos = wp.to_torch(self.robot.data.default_joint_pos)
        self.robot.set_joint_position_target_index(target=default_pos, joint_ids=list(range(len(self.robot.joint_names))))

        # Parse keyframes
        kf = KEYFRAMES.copy()

        # Detect coupling mode and adjust gripper close value.
        # Kinematic mode uses Newton's original value (0.5); one-way/two-way
        # needs tighter grip (0.2) to hold the duck with PD dynamics.
        coupling_mode = getattr(self.cfg.sim.physics.solver_cfg, "coupling_mode", "kinematic")
        is_kinematic = coupling_mode == "kinematic"
        # Sync kinematic_control flag with coupling mode
        self.cfg.kinematic_control = is_kinematic
        if not is_kinematic:
            gripper_close_override = 0.4
            close_mask = kf[:, -1] < 1.0  # rows where gripper is "close"
            kf[close_mask, -1] = gripper_close_override

        self._kf_targets = kf[:, 1:]  # (N, 8): px,py,pz, qx,qy,qz,qw, gripper
        self._kf_cum_time = np.cumsum(kf[:, 0])  # cumulative time for each keyframe
        self._sim_time = 0.0
        self._step_dt = self.cfg.sim.dt * self.cfg.decimation

        # IK state (set up lazily after Newton model is ready)
        self._ik_available = False
        self._ik_setup_attempted = False

        # R-key reset support
        self._request_reset = False
        self._reset_key_registered = False

        logger.info(
            "SoftbodyFrankaEnv: %d keyframes, total_time=%.1fs, step_dt=%.4fs",
            len(kf),
            self._kf_cum_time[-1],
            self._step_dt,
        )

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        # Resolve Newton asset paths at runtime (newton must be imported after Kit init)
        self._resolve_newton_assets()

        self.robot = Articulation(self.cfg.robot_cfg)
        self.soft_body = DeformableObject(self.cfg.soft_body)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Spawn table as a static cuboid
        tx, ty, tz = self.cfg.table_pos
        sx, sy, sz = self.cfg.table_size
        table_cfg = CuboidCfg(
            size=(sx, sy, sz),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.15)),
        )
        spawn_cuboid("/World/envs/env_0/table", table_cfg, translation=(tx, ty, tz))

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Deactivate the root_joint FixedJoint under the Physics/ scope.
        # Newton's parser tries to merge joints in this scope into D6 and fails on
        # FixedJoints. The fix_base_joint (outside Physics/) anchors the robot and
        # must stay active.
        from isaaclab.sim.utils.stage import get_current_stage

        stage = get_current_stage()
        for prim in list(stage.Traverse()):
            pp = str(prim.GetPath())
            if prim.GetTypeName() == "PhysicsFixedJoint" and "/Physics/" in pp:
                logger.info("Deactivating %s for Newton compatibility.", pp)
                prim.SetActive(False)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

        if self.cfg.disable_robot_ground_collision:
            self._register_ground_collision_disable()

    def _resolve_newton_assets(self):
        """Resolve Newton asset paths for the FR3 robot and duck at runtime."""
        import pathlib

        import newton.utils

        # FR3 URDF
        if hasattr(self.cfg.robot_cfg.spawn, "asset_path") and "PLACEHOLDER" in str(self.cfg.robot_cfg.spawn.asset_path):
            franka_dir = newton.utils.download_asset("franka_emika_panda")
            self.cfg.robot_cfg.spawn.asset_path = str(franka_dir / "urdf" / "fr3_franka_hand.urdf")

        # Duck TetMesh wrapper
        if hasattr(self.cfg.soft_body.spawn, "usd_path") and "PLACEHOLDER" in str(self.cfg.soft_body.spawn.usd_path):
            duck_dir = newton.utils.download_asset("manipulation_objects/rubber_duck")
            mesh_usd = str(duck_dir / "mesh.usd")
            wrapper_path = duck_dir / "tetmesh_only.usda"

            if not wrapper_path.exists():
                wrapper = (
                    '#usda 1.0\n'
                    '(\n'
                    '    defaultPrim = "duck"\n'
                    '    metersPerUnit = 1\n'
                    '    upAxis = "Z"\n'
                    ')\n'
                    '\n'
                    'def Xform "duck"\n'
                    '{\n'
                    f'    def TetMesh "geometry" (\n'
                    f'        prepend references = @{mesh_usd}@</TetModel>\n'
                    '    )\n'
                    '    {\n'
                    '    }\n'
                    '}\n'
                )
                wrapper_path.write_text(wrapper)

            self.cfg.soft_body.spawn.usd_path = str(wrapper_path)

    def _register_ground_collision_disable(self):
        """Disable ground-robot and table-robot rigid collision in Newton.

        Sets collision group to 0 for ground and table shapes so the robot arm
        doesn't collide with them. Soft (particle) contacts are unaffected.
        The Newton example controls the robot kinematically so table collisions
        are harmless; with PD dynamics they cause instability.
        """
        from isaaclab_newton.physics import NewtonManager

        from isaaclab.physics import PhysicsEvent

        def _disable(payload=None):
            model = NewtonManager._model
            if model is None:
                return
            labels = list(model.shape_label)
            groups = model.shape_collision_group.numpy()
            disabled = []
            for i, label in enumerate(labels):
                ll = label.lower()
                if "ground" in ll or "defaultgroundplane" in ll or "table" in ll:
                    groups[i] = 0
                    disabled.append(label)
            model.shape_collision_group.assign(wp.array(groups, dtype=int, device=model.shape_collision_group.device))
            logger.info("Disabled rigid collision for shapes: %s", disabled)

        NewtonManager.register_callback(_disable, PhysicsEvent.PHYSICS_READY)

    # ------------------------------------------------------------------
    # IK setup (lazy, needs Newton model)
    # ------------------------------------------------------------------

    def _try_setup_ik(self):
        """Initialize Newton IK solver (once, after Newton model is available)."""
        if self._ik_setup_attempted:
            return

        try:
            import newton
            import newton.ik as ik
            from isaaclab_newton.physics import NewtonManager

            if NewtonManager._model is None:
                return
            self._ik_setup_attempted = True

            # Build a separate fixed-base Newton model for IK, identical to
            # the Newton example: builder.add_urdf(floating=False).  This
            # avoids the floating-base DOFs that corrupt the IK solution.
            ik_builder = newton.ModelBuilder(gravity=-9.81)
            franka_dir = newton.utils.download_asset("franka_emika_panda")
            ik_builder.add_urdf(
                str(franka_dir / "urdf" / "fr3_franka_hand.urdf"),
                xform=wp.transform((-0.5, -0.5, -0.1), wp.quat_identity()),
                floating=False,
                scale=1.0,
                enable_self_collisions=False,
                collapse_fixed_joints=True,
                force_show_colliders=False,
            )
            ik_builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]
            ik_builder.color()
            ik_model = ik_builder.finalize(requires_grad=False)

            # The fixed-base model has body_count-3 as the EE (same as Newton example)
            self._ee_ik_index = ik_model.body_count - 3

            # IK objectives
            kf0 = self._kf_targets[0]
            target_pos = wp.vec3(float(kf0[0]), float(kf0[1]), float(kf0[2]))
            target_rot = wp.vec4(float(kf0[3]), float(kf0[4]), float(kf0[5]), float(kf0[6]))

            self._pos_obj = ik.IKObjectivePosition(
                link_index=self._ee_ik_index,
                link_offset=wp.vec3(0.0, 0.0, 0.22),
                target_positions=wp.array([target_pos], dtype=wp.vec3),
            )
            self._rot_obj = ik.IKObjectiveRotation(
                link_index=self._ee_ik_index,
                link_offset_rotation=wp.quat_identity(),
                target_rotations=wp.array([target_rot], dtype=wp.vec4),
            )
            self._joint_limit_obj = ik.IKObjectiveJointLimit(
                joint_limit_lower=ik_model.joint_limit_lower,
                joint_limit_upper=ik_model.joint_limit_upper,
                weight=10.0,
            )

            self._n_ik_coords = ik_model.joint_coord_count
            self._ik_joint_q = wp.array(ik_model.joint_q, shape=(1, self._n_ik_coords))
            self._ik_solver = ik.IKSolver(
                model=ik_model,
                n_problems=1,
                objectives=[self._pos_obj, self._rot_obj, self._joint_limit_obj],
                lambda_initial=0.1,
                jacobian_mode=ik.IKJacobianType.ANALYTIC,
            )
            self._ik_model = ik_model

            self._ik_available = True
            logger.info(
                "[SoftbodyFrankaEnv] IK model built (fixed-base, %d coords, EE=%d)",
                self._n_ik_coords,
                self._ee_ik_index,
            )

        except Exception as exc:
            import traceback

            print(f"[SoftbodyFrankaEnv] IK setup failed: {exc}", flush=True)
            traceback.print_exc()
            logger.warning("[SoftbodyFrankaEnv] IK setup failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Keyframe interpolation
    # ------------------------------------------------------------------

    def _interpolate_keyframe(self) -> tuple[np.ndarray, float]:
        """Return the interpolated (pos3, rot4, gripper) target for the current sim time."""
        t = self._sim_time
        cum = self._kf_cum_time
        targets = self._kf_targets

        # Clamp to last keyframe
        if t >= cum[-1]:
            return targets[-1], float(targets[-1, -1])

        idx = int(np.searchsorted(cum, t))
        t_start = cum[idx - 1] if idx > 0 else 0.0
        t_end = cum[idx]
        alpha = float(np.clip((t - t_start) / (t_end - t_start + 1e-8), 0.0, 1.0))

        prev = targets[idx - 1] if idx > 0 else targets[idx]
        cur = targets[idx]
        interp = (1.0 - alpha) * prev + alpha * cur
        return interp, float(interp[-1])

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        pass  # actions are ignored (scripted motion)

    def _try_register_reset_key(self):
        """Lazily find the Newton viewer and register R key for reset."""
        if self._reset_key_registered:
            return
        try:
            from isaaclab_visualizers.newton import NewtonVisualizer

            for v in self.sim.visualizers:
                if isinstance(v, NewtonVisualizer) and v._viewer is not None:
                    import pyglet.window.key as key

                    def _on_key(symbol, modifiers, _self=self):
                        if symbol == key.R:
                            _self._request_reset = True
                            print("[SoftbodyFrankaEnv] Reset requested via R key", flush=True)

                    v._viewer.renderer.register_key_press(_on_key)
                    self._reset_key_registered = True
                    logger.info("R key (reset) registered on Newton viewer.")
                    return
        except Exception:
            pass

    def _apply_action(self) -> None:
        if not self._reset_key_registered:
            self._try_register_reset_key()

        if not self._ik_available:
            self._try_setup_ik()

        target, gripper_activation = self._interpolate_keyframe()

        if self._ik_available:
            # Update IK targets
            self._pos_obj.set_target_position(
                0, wp.vec3(float(target[0]), float(target[1]), float(target[2]))
            )
            self._rot_obj.set_target_rotation(
                0, wp.vec4(float(target[3]), float(target[4]), float(target[5]), float(target[6]))
            )

            # Solve IK — accumulate from previous solution (like Newton example).
            # Do NOT re-seed from current robot state; different seeding leads to
            # different IK local minima.
            self._ik_solver.step(self._ik_joint_q, self._ik_joint_q, iterations=24)

            # Set finger positions in the IK buffer (like Newton's set_gripper_q kernel)
            ik_q_torch = wp.to_torch(self._ik_joint_q)
            ik_q_torch[0, self._n_ik_coords - 2] = gripper_activation * 0.04
            ik_q_torch[0, self._n_ik_coords - 1] = gripper_activation * 0.04

            # Apply IK solution to the robot
            solved_arm_q = wp.to_torch(self._ik_joint_q)[0, self._arm_joint_idx]

            if self.cfg.kinematic_control:
                # Kinematic mode: compute joint velocity = (target - current) / frame_dt
                # and set as velocity target. The coupled solver's kinematic step assigns
                # this to joint_qd and uses Featherstone as a kinematic integrator.
                current_q = wp.to_torch(self.robot.data.joint_pos)[0]
                arm_vel = (solved_arm_q - current_q[self._arm_joint_idx]) / self._step_dt
                self.robot.set_joint_velocity_target_index(
                    target=arm_vel.unsqueeze(0), joint_ids=self._arm_joint_idx
                )
            else:
                # PD mode: set position targets for the PD controller.
                self.robot.set_joint_position_target_index(
                    target=solved_arm_q.unsqueeze(0), joint_ids=self._arm_joint_idx
                )

        # Set finger targets
        finger_pos = gripper_activation * 0.04
        finger_target = torch.full(
            (self.num_envs, len(self._finger_joint_idx)),
            finger_pos,
            dtype=torch.float32,
            device=self.device,
        )
        if self.cfg.kinematic_control:
            current_finger = wp.to_torch(self.robot.data.joint_pos)[0, self._finger_joint_idx]
            finger_vel = (finger_target[0] - current_finger) / self._step_dt
            self.robot.set_joint_velocity_target_index(
                target=finger_vel.unsqueeze(0), joint_ids=self._finger_joint_idx
            )
        else:
            self.robot.set_joint_position_target_index(target=finger_target, joint_ids=self._finger_joint_idx)

        # Advance scripted time
        self._sim_time += self._step_dt

    def _get_observations(self) -> dict:
        self.soft_body.update(self.step_dt)
        nodal_pos = wp.to_torch(self.soft_body.data.nodal_pos_w)
        self._object_centroid = nodal_pos.mean(dim=1)

        joint_pos = wp.to_torch(self.robot.data.joint_pos)
        joint_vel = wp.to_torch(self.robot.data.joint_vel)

        obs = torch.cat(
            (
                joint_pos,
                joint_vel,
                self._object_centroid,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        if self._request_reset:
            time_out = torch.ones_like(time_out)
            self._request_reset = False

        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == 0:
            return
        super()._reset_idx(env_ids)

        # Reset robot
        joint_pos = wp.to_torch(self.robot.data.default_joint_pos)[env_ids].clone()
        joint_vel = wp.to_torch(self.robot.data.default_joint_vel)[env_ids].clone()

        default_root_pose = wp.to_torch(self.robot.data.default_root_pose)[env_ids].clone()
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]
        default_root_vel = wp.to_torch(self.robot.data.default_root_vel)[env_ids].clone()

        self.robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        # Reset deformable object
        env_ids_list = env_ids.cpu().tolist() if hasattr(env_ids, "cpu") else list(env_ids)
        default_state = wp.to_torch(self.soft_body.data.default_nodal_state_w)
        self.soft_body.write_nodal_state_to_sim_index(default_state, env_ids=env_ids_list)
        self.soft_body.reset(env_ids=env_ids_list)

        # Reset scripted time and joint targets
        self._sim_time = 0.0
        default_pos = wp.to_torch(self.robot.data.default_joint_pos)
        self.robot.set_joint_position_target_index(
            target=default_pos, joint_ids=list(range(len(self.robot.joint_names)))
        )
