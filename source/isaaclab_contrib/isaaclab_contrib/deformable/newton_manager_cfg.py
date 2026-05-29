# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for VBD, coupled solver, and global Newton model parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_newton.physics import FeatherstoneSolverCfg, MJWarpSolverCfg, NewtonCfg, NewtonSolverCfg

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager

    from isaaclab.scene import InteractiveSceneCfg


@configclass
class VBDSolverCfg(NewtonSolverCfg):
    """Configuration for the Vertex Block Descent (VBD) solver.

    Supports cloth, soft bodies, and coupled rigid-body systems. Requires
    ``ModelBuilder.color()`` before ``finalize()`` to build the vertex coloring.
    """

    class_type: type[NewtonManager] | str = "{DIR}.vbd_manager:NewtonVBDManager"
    """Manager class for the VBD solver."""

    requires_graph_coloring: bool = True

    iterations: int = 10
    """Number of VBD iterations per substep."""

    integrate_with_external_rigid_solver: bool = False
    """Whether rigid bodies are integrated by an external solver (one-way coupling).

    Set to ``True`` when coupling cloth with a separate rigid-body solver so VBD
    only integrates the cloth particles.
    """

    particle_enable_self_contact: bool = False
    """Whether to enable VBD deformable's self-contact."""

    particle_self_contact_radius: float = 0.005
    """Particle radius used for self-contact detection [m]."""

    particle_self_contact_margin: float = 0.005
    """Self-contact detection margin [m]. Should be >= particle_self_contact_radius."""

    particle_collision_detection_interval: int = -1
    """How often particle self-contact detection is applied.

    ``< 0``: once before initialization. ``0``: once before and once after
    initialization. ``k >= 1``: before every ``k`` VBD iterations.
    """

    particle_vertex_contact_buffer_size: int = 32
    """Preallocation size for each vertex's vertex-triangle collision buffer."""

    particle_edge_contact_buffer_size: int = 64
    """Preallocation size for each edge's edge-edge collision buffer."""

    particle_topological_contact_filter_threshold: int = 2
    """Maximum topological distance (in rings) below which self-contacts are discarded.

    Only used when ``particle_enable_self_contact`` is ``True``. Values > 3
    significantly increase computation time.
    """

    particle_rest_shape_contact_exclusion_radius: float = 0.0
    """Rest-configuration separation threshold for filtering close primitives [m].

    Only used when ``particle_enable_self_contact`` is ``True``.
    """

    rigid_contact_k_start: float = 1.0e2
    """Initial stiffness seed for all rigid body contacts [N/m]."""

    # ----------------------------------------------------------------------
    # AVBD rigid integrator parameters.  When ``integrate_with_external_rigid_solver``
    # is ``False`` VBD integrates rigid bodies via the AVBD algorithm; these
    # knobs control its stabilization, penalty ramping, and joint constraint
    # stiffness.  Defaults match upstream ``newton.solvers.SolverVBD``.
    # ----------------------------------------------------------------------

    rigid_avbd_alpha: float = 0.95
    """C0 stabilization strength (``C_stab = C - alpha * C0``)."""

    rigid_avbd_joint_alpha: float | None = None
    """Joint alpha override [dimensionless]. ``None`` uses :attr:`rigid_avbd_alpha`."""

    rigid_avbd_contact_alpha: float | None = None
    """Body-body contact alpha override [dimensionless]. ``None`` selects default."""

    rigid_avbd_beta: float = 0.0
    """Penalty ramp rate per iteration. ``0`` disables ramping (fixed-k)."""

    rigid_avbd_linear_beta: float | None = None
    """Linear beta override. ``None`` uses :attr:`rigid_avbd_beta`."""

    rigid_avbd_angular_beta: float | None = None
    """Angular beta override. ``None`` uses :attr:`rigid_avbd_beta`."""

    rigid_avbd_gamma: float = 0.999
    """Per-step decay for penalty ``k`` and persisted hard-mode ``lambda``."""

    rigid_joint_linear_k_start: float = 1.0e4
    """Linear joint constraint penalty seed when ramping is enabled [N/m]."""

    rigid_joint_angular_k_start: float = 1.0e1
    """Angular joint constraint penalty seed when ramping is enabled [N*m/rad]."""

    rigid_joint_linear_ke: float = 1.0e5
    """Penalty stiffness ceiling for structural linear joint constraints [N/m]."""

    rigid_joint_angular_ke: float = 1.0e5
    """Penalty stiffness ceiling for structural angular joint constraints [N*m/rad]."""

    rigid_joint_linear_kd: float = 0.0
    """Penalty damping for structural linear joint constraints [N*s/m]."""

    rigid_joint_angular_kd: float = 0.0
    """Penalty damping for structural angular joint constraints [N*m*s/rad]."""


@configclass
class CoupledMJWarpVBDSolverCfg(NewtonSolverCfg):
    """Configuration for the coupled MJWarp + VBD solver.

    Alternates a rigid-body solver (:class:`MJWarpSolverCfg`) and VBD per substep.
    The coupling direction is controlled by :attr:`coupling_mode`.
    """

    class_type: type[NewtonManager] | str = "{DIR}.coupled_mjwarp_vbd_manager:NewtonCoupledMJWarpVBDManager"
    """Manager class for the coupled MJWarp + VBD solver."""

    rigid_solver_cfg: MJWarpSolverCfg = MJWarpSolverCfg()
    """Rigid-body sub-solver configuration."""

    soft_solver_cfg: VBDSolverCfg = VBDSolverCfg(integrate_with_external_rigid_solver=True)
    """VBD sub-solver configuration for cloth/particle dynamics."""

    coupling_mode: str = "two_way"
    """Coupling direction between the rigid and VBD solvers.

    - ``"one_way"``: Rigid -> soft only.
    - ``"two_way"``: Same-substep two-way coupling with normal + Coulomb friction.
    """


@configclass
class CoupledFeatherstoneVBDSolverCfg(NewtonSolverCfg):
    """Configuration for the coupled Featherstone + VBD solver.

    Alternates a rigid-body solver (:class:`FeatherstoneSolverCfg`) and VBD per
    substep. The coupling direction is controlled by :attr:`coupling_mode`.
    """

    class_type: type[NewtonManager] | str = "{DIR}.coupled_featherstone_vbd_manager:NewtonCoupledFeatherstoneVBDManager"
    """Manager class for the coupled Featherstone + VBD solver."""

    rigid_solver_cfg: FeatherstoneSolverCfg = FeatherstoneSolverCfg()
    """Rigid-body sub-solver configuration."""

    soft_solver_cfg: VBDSolverCfg = VBDSolverCfg(integrate_with_external_rigid_solver=True)
    """VBD sub-solver configuration for cloth/particle dynamics."""

    coupling_mode: str = "kinematic"
    """Coupling direction between the rigid and VBD solvers.

    Accepts the same values as :attr:`CoupledMJWarpVBDSolverCfg.coupling_mode`,
    plus ``"kinematic"`` (rigid -> soft only, rigid bodies kinematically updated).
    """


@configclass
class NewtonModelCfg:
    """Global Newton model parameters applied after builder finalization.

    These control model-level contact behavior shared across all objects.
    """

    soft_contact_ke: float = 1.0e3
    """Body-particle and particle self-contact stiffness [N/m].

    Effective per-contact stiffness is the average of this value and the rigid
    shape's material stiffness.
    """

    soft_contact_kd: float = 1.0e-2
    """Body-particle contact damping [N*s/m]."""

    soft_contact_mu: float = 0.5
    """Body-particle contact friction coefficient [dimensionless].

    Effective per-contact friction is ``sqrt(soft_contact_mu * shape_material_mu)``.
    """

    shape_material_ke: float | None = None
    """Per-shape contact stiffness override [N/m]. ``None`` keeps USD/MJCF values."""

    shape_material_kd: float | None = None
    """Per-shape contact damping override [N*s/m]. ``None`` keeps USD/MJCF values."""

    shape_material_mu: float | None = None
    """Per-shape friction coefficient override [dimensionless]. ``None`` keeps USD/MJCF values."""


@configclass
class CoupledNewtonCfg(NewtonCfg):
    """:class:`NewtonCfg` extended for coupled-solver setups.

    Adds :attr:`model_cfg` for global model parameters and :attr:`scene_cfg` so
    the manager can resolve :class:`~isaaclab.managers.SceneEntityCfg` selectors
    against the scene at solver-build time.

    Uses a distinct class name so :func:`_is_kitless_physics` does not match it,
    ensuring Kit is launched for USD deformable/coupled spawning.
    """

    model_cfg: NewtonModelCfg | None = None
    """Global Newton model parameters applied after builder finalization."""

    scene_cfg: InteractiveSceneCfg | None = None
    """Scene cfg used by coupled solvers to resolve scene-entity selectors.

    Set to ``self.scene`` from the env's ``__post_init__``.
    """
