# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum functions for the racing quadcopter environment."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv


def check_collision_curriculum_enabled(
    env: "DirectRLEnv",
    gates_passed_per_env: torch.Tensor,  # [N] tensor
    passage_velocity_per_env: torch.Tensor,  # [N] tensor
    passage_centering_per_env: torch.Tensor,  # [N] tensor
    passage_count_per_env: torch.Tensor,  # [N] tensor
    min_gates_passed: int = 3,
    min_mean_velocity: float = 2.5,
    max_center_offset: float = 0.15,
) -> torch.Tensor:
    """Check which environments should have collision detection enabled.
    
    Returns a boolean tensor [N] indicating which envs should use contact sensor collision.
    
    Args:
        env: The environment.
        gates_passed_per_env: Total gates passed in current episode for each env [N].
        passage_velocity_per_env: Sum of velocities at gate passages [N].
        passage_centering_per_env: Sum of center offsets at gate passages [N].
        passage_count_per_env: Number of gate passages in current episode [N].
        min_gates_passed: Minimum gates passed to enable collision detection.
        min_mean_velocity: Minimum mean velocity (m/s) at passages.
        max_center_offset: Maximum acceptable mean center offset at passages (m).
    
    Returns:
        Boolean tensor [N] indicating which environments have collision detection enabled.
    """
    N = gates_passed_per_env.shape[0]
    
    # Check gates passed threshold
    enough_gates = gates_passed_per_env >= min_gates_passed
    
    # Check mean velocity (only for envs that passed gates)
    has_passages = passage_count_per_env > 0
    mean_velocity = torch.where(
        has_passages,
        passage_velocity_per_env / passage_count_per_env,
        torch.zeros_like(passage_velocity_per_env)
    )
    good_velocity = (mean_velocity >= min_mean_velocity) | (~has_passages)
    
    # Check mean centering (only for envs that passed gates)
    mean_centering = torch.where(
        has_passages,
        passage_centering_per_env / passage_count_per_env,
        torch.zeros_like(passage_centering_per_env)
    )
    good_centering = (mean_centering <= max_center_offset) | (~has_passages)
    
    # Enable collision detection for envs meeting all criteria
    enabled = enough_gates & good_velocity & good_centering
    
    return enabled

