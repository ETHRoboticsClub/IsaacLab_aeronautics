# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward manager for racing quadcopter environment.

Optimized for vectorized computation with torch tensors.
All rewards are designed to encourage:
1. Flying through gates in order
2. Staying centered when passing through
3. Approaching from the correct side
4. Smooth, controlled flight
"""

from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class RewardWeights:
    """Reward weight configuration.
    
    Primary rewards encourage gate passage and proper trajectory.
    Penalties discourage unsafe or inefficient behavior.
    """
    
    # Primary rewards (positive)
    gate_passage: float = 25.0  # Large reward for passing through gate
    gate_progress: float = 2.0   # Reward for moving toward next gate
    gate_centering: float = 0.5  # Reward for staying aligned with gate center (only on approach)
    velocity_toward_gate: float = 0.1  # Reward for velocity directed toward gate
    
    # Penalties (negative weights)
    velocity_limit: float = -0.05   # Penalty for exceeding velocity limit
    orientation: float = -0.005     # Penalty for non-upright orientation
    angular_velocity: float = -0.001  # Penalty for excessive rotation
    action_smoothness: float = -0.0001  # Penalty for jerky actions
    action_magnitude: float = -0.001  # Penalty for large action values (encourages staying near 0)
    crash: float = -50.0            # Large penalty for crashing
    wrong_side: float = -0.1        # Penalty for being on wrong side of gate

    reward_scale: float = 1.0  # Overall scaling factor for rewards

    def __post_init__(self):
        """Validate reward weights."""
        assert self.gate_passage > 0, "Gate passage reward should be positive"
        assert self.gate_progress >= 0, "Gate progress reward should be non-negative"
        assert self.gate_centering >= 0, "Gate centering reward should be non-negative"
        assert self.velocity_toward_gate >= 0, "Velocity toward gate reward should be non-negative"
        
        assert self.velocity_limit <= 0, "Velocity limit penalty should be non-positive"
        assert self.orientation <= 0, "Orientation penalty should be non-positive"
        assert self.angular_velocity <= 0, "Angular velocity penalty should be non-positive"
        assert self.action_smoothness <= 0, "Action smoothness penalty should be non-positive"
        assert self.action_magnitude <= 0, "Action magnitude penalty should be non-positive"
        assert self.crash <= 0, "Crash penalty should be non-positive"
        assert self.wrong_side <= 0, "Wrong side penalty should be non-positive"
        
        assert self.reward_scale > 0, "Reward scale should be positive"

        # Scale all rewards by the overall reward scale factor, looping through all fields
        for field in self.__dataclass_fields__:
            if field != "reward_scale":  # Don't scale the reward_scale itself
                setattr(self, field, getattr(self, field) * self.reward_scale)


class GateRewardManager:
    """Manages reward computation for racing quadcopter gate navigation.
    
    Optimized for fully vectorized operations across all environments.
    Tracks episode statistics for tensorboard logging.
    """
    
    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        weights: RewardWeights | None = None,
    ):
        """Initialize reward manager.
        
        Args:
            num_envs: Number of parallel environments
            device: Device to run computations on
            weights: Reward weights configuration. If None, uses defaults.
        """
        self.num_envs = num_envs
        self.device = torch.device(device) if isinstance(device, str) else device
        self.weights = weights if weights is not None else RewardWeights()
        
        # Pre-allocate tensors for efficiency
        self._zeros = torch.zeros(num_envs, device=self.device)
        
        # Episode tracking for rewards
        self._episode_sums = {
            "gate_passage": torch.zeros(num_envs, device=self.device),
            "gate_progress": torch.zeros(num_envs, device=self.device),
            "gate_centering": torch.zeros(num_envs, device=self.device),
            "velocity_toward_gate": torch.zeros(num_envs, device=self.device),
            "velocity_penalty": torch.zeros(num_envs, device=self.device),
            "orientation_penalty": torch.zeros(num_envs, device=self.device),
            "angvel_penalty": torch.zeros(num_envs, device=self.device),
            "action_smoothness_penalty": torch.zeros(num_envs, device=self.device),
            "action_magnitude_penalty": torch.zeros(num_envs, device=self.device),
            "crash_penalty": torch.zeros(num_envs, device=self.device),
            "wrong_side_penalty": torch.zeros(num_envs, device=self.device),
            "total": torch.zeros(num_envs, device=self.device),
        }
        
        # Episode tracking for metrics (logged at episode end)
        self._passage_velocity_sum = torch.zeros(num_envs, device=self.device)
        self._passage_centering_sum = torch.zeros(num_envs, device=self.device)
        self._passage_count = torch.zeros(num_envs, device=self.device)
        
    def compute_rewards(
        self,
        # Robot state
        drone_pos: torch.Tensor,        # [N, 3]
        drone_vel: torch.Tensor,        # [N, 3]
        drone_angvel: torch.Tensor,     # [N, 3]
        projected_gravity: torch.Tensor, # [N, 3]
        actions: torch.Tensor,          # [N, 4]
        prev_actions: torch.Tensor,     # [N, 4]
        # Gate state
        gate_pos: torch.Tensor,         # [N, 3]
        gate_ori: torch.Tensor,         # [N, 4]
        gate_size: float,
        # Helper functions for gate directions
        gate_forward_fn,
        gate_right_fn,
        gate_up_fn,
        # Previous state
        prev_drone_pos: torch.Tensor,   # [N, 3]
        # Gate passage detection
        gate_passed: torch.Tensor,      # [N] float (0 or 1)
        # Limits
        velocity_limit: torch.Tensor,   # [N]
        # Termination
        crashed: torch.Tensor,          # [N] bool
        # Additional info for metrics (optional)
        crossing_offset: torch.Tensor | None = None,  # [N, 3] offset from gate center at crossing
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute all reward components using vectorized operations.
        
        Returns:
            total_reward: [N] tensor of total rewards
            components: Dictionary of individual reward components for debugging
        """
        N = drone_pos.shape[0]
        
        # Pre-compute commonly used values (vectorized)
        to_gate = gate_pos - drone_pos  # [N, 3]
        dist_to_gate = torch.norm(to_gate, dim=1)  # [N]
        to_gate_normalized = to_gate / (dist_to_gate.unsqueeze(-1) + 1e-8)  # [N, 3]
        
        # Gate directions (computed once)
        gate_fwd = gate_forward_fn(gate_ori)  # [N, 3]
        gate_right = gate_right_fn(gate_ori)  # [N, 3]
        gate_up = gate_up_fn(gate_ori)        # [N, 3]
        
        # Signed distance to gate plane (negative = approach side, positive = exit side)
        dist_to_plane = -(to_gate * gate_fwd).sum(dim=1)  # [N]
        on_approach_side = dist_to_plane <= 0.0
        
        # Lateral offset from gate center (in gate's local right/up plane)
        lateral_right = ((-to_gate) * gate_right).sum(dim=1)  # [N]
        lateral_up = ((-to_gate) * gate_up).sum(dim=1)        # [N]
        lateral_dist = torch.sqrt(lateral_right ** 2 + lateral_up ** 2)  # [N]
        
        # Drone speed and velocity direction
        speed = torch.norm(drone_vel, dim=1)  # [N]
        vel_normalized = drone_vel / (speed.unsqueeze(-1) + 1e-8)  # [N, 3]
        
        components = {}
        
        # =====================================================================
        # PRIMARY REWARDS
        # =====================================================================
        
        # 1. Gate passage: large reward for successfully passing through
        r_passage = self.weights.gate_passage * gate_passed
        components["gate_passage"] = r_passage
        
        # 2. Progress toward gate: reward for reducing distance
        prev_dist = torch.norm(prev_drone_pos - gate_pos, dim=1)
        progress = (prev_dist - dist_to_gate).clamp(-1.0, 1.0)
        r_progress = self.weights.gate_progress * progress
        components["gate_progress"] = r_progress
        
        # 3. Centering reward: encourage alignment with gate center
        # Only when on approach side, stronger when closer to gate
        proximity_factor = torch.exp(-0.5 * dist_to_plane.abs())  # More reward when close to gate
        centering_value = torch.exp(-3.0 * lateral_dist / gate_size)  # Exponential decay from center
        r_centering = self.weights.gate_centering * centering_value * proximity_factor * on_approach_side.float()
        components["gate_centering"] = r_centering
        
        # 4. Velocity toward gate: reward velocity pointing toward gate
        vel_toward_gate = (drone_vel * to_gate_normalized).sum(dim=1).clamp(min=0.0)  # [N]
        # Only reward when on approach side
        r_vel_toward = self.weights.velocity_toward_gate * vel_toward_gate * on_approach_side.float()
        components["velocity_toward_gate"] = r_vel_toward
        
        # =====================================================================
        # PENALTIES
        # =====================================================================
        
        # 5. Velocity limit penalty: quadratic penalty for exceeding limit
        speed_excess = (speed - velocity_limit).clamp(min=0.0)
        p_velocity = self.weights.velocity_limit * speed_excess ** 2
        components["velocity_penalty"] = p_velocity
        
        # 6. Orientation penalty: prefer upright (gravity should point down in body frame)
        # projected_gravity[:, 2] should be close to -1 for upright
        # Use squared deviation of horizontal gravity components
        upright_projected_gravity = torch.tensor([0.0, 0.0, -1.0], device=projected_gravity.device)  # Desired gravity direction
        # acos will break with input close to -1.0 and 1.0
        orientation_error_angle = torch.acos(torch.clamp(torch.norm((projected_gravity * upright_projected_gravity), dim=1), -0.98, 0.98)) 
        orientation_error = torch.square(orientation_error_angle)
        p_orientation = self.weights.orientation * orientation_error
        components["orientation_penalty"] = p_orientation
        
        # 7. Angular velocity penalty: prefer smooth rotation
        p_angvel = self.weights.angular_velocity * drone_angvel.square().sum(dim=1)
        components["angvel_penalty"] = p_angvel
        
        # 8. Action smoothness penalty: prefer consistent actions
        action_diff = actions - prev_actions
        p_smoothness = self.weights.action_smoothness * action_diff.square().sum(dim=1)
        components["action_smoothness_penalty"] = p_smoothness
        
        # 9. Action magnitude penalty: encourage actions close to 0
        p_action_mag = self.weights.action_magnitude * actions.square().sum(dim=1)
        components["action_magnitude_penalty"] = p_action_mag
        
        # 10. Crash penaltyy
        p_crash = self.weights.crash * crashed.float()
        components["crash_penalty"] = p_crash
        
        # 11. Wrong side penalty: discourage being on exit side without passing
        # Small continuous penalty when on wrong side of gate
        p_wrong_side = self.weights.wrong_side * (~on_approach_side).float() * (1.0 - gate_passed)
        components["wrong_side_penalty"] = p_wrong_side
        
        # =====================================================================
        # TOTAL REWARD
        # =====================================================================
        
        total_reward = (
            r_passage + r_progress + r_centering + r_vel_toward +
            p_velocity + p_orientation + p_angvel + p_smoothness + p_action_mag + p_crash + p_wrong_side
        )
        components["total"] = total_reward
        
        # =====================================================================
        # UPDATE EPISODE SUMS
        # =====================================================================
        
        self._episode_sums["gate_passage"] += r_passage
        self._episode_sums["gate_progress"] += r_progress
        self._episode_sums["gate_centering"] += r_centering
        self._episode_sums["velocity_toward_gate"] += r_vel_toward
        self._episode_sums["velocity_penalty"] += p_velocity
        self._episode_sums["orientation_penalty"] += p_orientation
        self._episode_sums["angvel_penalty"] += p_angvel
        self._episode_sums["action_smoothness_penalty"] += p_smoothness
        self._episode_sums["action_magnitude_penalty"] += p_action_mag
        self._episode_sums["crash_penalty"] += p_crash
        self._episode_sums["wrong_side_penalty"] += p_wrong_side
        self._episode_sums["total"] += total_reward
        
        # =====================================================================
        # UPDATE PASSAGE METRICS (for tensorboard logging)
        # =====================================================================
        
        # Track metrics at gate passage time
        passed_mask = gate_passed > 0.5
        if passed_mask.any():
            self._passage_velocity_sum[passed_mask] += speed[passed_mask]
            self._passage_centering_sum[passed_mask] += lateral_dist[passed_mask]
            self._passage_count[passed_mask] += 1.0
        
        return total_reward, components
    
    def reset_episode_sums(self, env_ids: torch.Tensor) -> dict[str, float]:
        """Reset episode sums for specified environments and return averages.
        
        Args:
            env_ids: Indices of environments to reset
            
        Returns:
            Dictionary of metrics for tensorboard logging
        """
        if len(env_ids) == 0:
            return {}
        
        metrics = {}
        
        # Reward component averages
        for key, sums in self._episode_sums.items():
            metrics[f"Episode_Reward/{key}"] = sums[env_ids].mean().item()
            sums[env_ids] = 0.0
        
        # Passage metrics (only if any gates were passed)
        passage_counts = self._passage_count[env_ids]
        valid_mask = passage_counts > 0
        
        if valid_mask.any():
            valid_ids = env_ids[valid_mask]
            valid_counts = passage_counts[valid_mask]
            
            # Mean velocity at passage
            mean_vel = (self._passage_velocity_sum[valid_ids] / valid_counts).mean().item()
            metrics["Episode_Metrics/passage_velocity_mean"] = mean_vel
            
            # Mean distance from center at passage
            mean_centering = (self._passage_centering_sum[valid_ids] / valid_counts).mean().item()
            metrics["Episode_Metrics/passage_center_offset_mean"] = mean_centering
        else:
            metrics["Episode_Metrics/passage_velocity_mean"] = 0.0
            metrics["Episode_Metrics/passage_center_offset_mean"] = 0.0
        
        # Reset passage metrics
        self._passage_velocity_sum[env_ids] = 0.0
        self._passage_centering_sum[env_ids] = 0.0
        self._passage_count[env_ids] = 0.0
        
        return metrics
    
    def get_episode_sums(self) -> dict[str, torch.Tensor]:
        """Get current episode sum tensors."""
        return self._episode_sums
