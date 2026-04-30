# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Solver factory functions for VBD and coupled solvers.

These factories are registered with :class:`NewtonManager` via
:meth:`NewtonManager.register_solver_factory` so the manager can create
VBD and coupled solvers without hard-coding their logic.
"""

from __future__ import annotations

import inspect
import logging

from newton.solvers import SolverVBD

logger = logging.getLogger(__name__)


def create_vbd_solver(manager_cls, cfg_dict: dict, solver_cfg) -> None:
    """Create and assign a VBD solver on the Newton manager.

    Args:
        manager_cls: The :class:`NewtonManager` class (not an instance).
        cfg_dict: Solver configuration dictionary with ``solver_type`` already popped.
        solver_cfg: The original solver configuration object.
    """
    manager_cls._use_single_state = False
    solver_sig = inspect.signature(SolverVBD.__init__)
    valid_solver_args = set(solver_sig.parameters.keys()) - {"self", "model"}
    filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_solver_args}
    logger.info("VBD: Creating SolverVBD with args: %s", filtered_cfg)
    logger.info("VBD: model particle_count=%s", getattr(manager_cls._model, "particle_count", "N/A"))
    num_groups = len(getattr(manager_cls._model, "particle_color_groups", []))
    logger.info("VBD: model particle_color_groups has %d groups", num_groups)
    manager_cls._solver = SolverVBD(manager_cls._model, **filtered_cfg)
    logger.info("VBD: SolverVBD created successfully")

    # VBD needs a collision pipeline for:
    # - Particle self-contacts (cloth self-collision)
    # - Body-particle contacts when AVBD integrates rigid bodies
    needs_pipeline = False
    if hasattr(solver_cfg, "particle_enable_self_contact") and solver_cfg.particle_enable_self_contact:
        needs_pipeline = True
    if not getattr(solver_cfg, "integrate_with_external_rigid_solver", True):
        # AVBD mode: VBD handles rigid bodies; needs collision pipeline for
        # body-particle and body-body contacts.
        needs_pipeline = True
        # AVBD updates body_q/body_qd but not joint_q/joint_qd.  Signal the
        # newton manager to run eval_ik after each simulation step so that
        # articulation data (joint positions/velocities) stays in sync.
        manager_cls._needs_ik_sync = True
    if needs_pipeline:
        manager_cls._needs_collision_pipeline = True
        manager_cls._initialize_contacts()


def create_coupled_solver(manager_cls, cfg_dict: dict, solver_cfg) -> None:
    """Create and assign a coupled rigid-body + VBD solver on the Newton manager.

    Args:
        manager_cls: The :class:`NewtonManager` class (not an instance).
        cfg_dict: Solver configuration dictionary with ``solver_type`` already popped.
        solver_cfg: The original solver configuration object.
    """
    from .coupled_solver import CoupledSolver

    manager_cls._use_single_state = False
    manager_cls._soft_contact_margin = solver_cfg.soft_contact_margin

    # Initialize collision pipeline to pass into the coupled solver
    manager_cls._needs_collision_pipeline = True
    manager_cls._initialize_contacts()

    manager_cls._solver = CoupledSolver(
        model=manager_cls._model,
        cfg=solver_cfg,
        collision_pipeline=manager_cls._collision_pipeline,
        contacts=manager_cls._contacts,
    )
