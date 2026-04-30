# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick-AVBD-Cube environment: Franka robot interacts with a deformable cube using the unified AVBD solver.

This environment is identical to Pick-VBD-Cube except that it uses Newton's
unified AVBD solver (SolverVBD with ``integrate_with_external_rigid_solver=False``)
instead of the CoupledSolver approach. AVBD handles both articulated rigid bodies
and deformable particles in a single solver.
"""

from __future__ import annotations

from isaaclab_tasks.direct.pick_vbd_cube.pick_vbd_cube_env import PickVBDCubeEnv

from .pick_avbd_cube_env_cfg import PickAVBDCubeEnvCfg


class PickAVBDCubeEnv(PickVBDCubeEnv):
    """Pick-AVBD-Cube environment using the unified AVBD solver.

    Inherits all behavior from :class:`PickVBDCubeEnv`. The only difference
    is the physics configuration: AVBD handles both rigid and deformable
    bodies in a single solver instead of using a coupled solver.
    """

    cfg: PickAVBDCubeEnvCfg
