# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager classes for observations, rewards, and state tracking in quadcopter racing environment."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from .trajectory import RacingTrajectory
    from isaaclab.assets import Articulation


@dataclass
class RewardConfig:
    """Configuration for reward function weights and parameters."""

    progress_scale: float = 10.0
    contour_error_scale: float = -2.0
    velocity_alignment_scale: float = 1.0
    speed_tracking_scale: float = 0.5
    velocity_limit_penalty_scale: float = -2.0
    orientation_penalty_scale: float = -0.5
    ang_vel_penalty_scale: float = -0.01
    action_smoothness_scale: float = -0.001


class RewardManager:
    """Manages reward computation for quadcopter racing."""

    def __init__(
        self,
        cfg: RewardConfig,
        num_envs: int,
        device: str,
        trajectory: RacingTrajectory,
    ):
        """Initialize reward manager.

        Args:
            cfg: Reward configuration with scales.
            num_envs: Number of parallel environments.
            device: Device for tensor operations.
            trajectory: Racing trajectory instance for progress tracking.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self._trajectory = trajectory

        # State tracking
        self._previous_progress = torch.zeros(num_envs, device=device)
        self._closest_traj_point = torch.zeros(num_envs, 3, device=device)
        self._contour_error = torch.zeros(num_envs, device=device)
        self._traj_tangent = torch.zeros(num_envs, 3, device=device)

        # Episode statistics
        self._episode_sums = {
            key: torch.zeros(num_envs, dtype=torch.float, device=device)
            for key in [
                "progress",
                "contour_error",
                "velocity_alignment",
                "speed_tracking",
                "velocity_limit_penalty",
                "orientation_penalty",
                "ang_vel_penalty",
                "action_smoothness",
            ]
        }

    def compute_rewards(
        self,
        robot: Articulation,
        actions: torch.Tensor,
        previous_actions: torch.Tensor,
        step_dt: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute total reward and individual components.

        Args:
            robot: Robot articulation with state data.
            actions: Current actions [num_envs, action_dim].
            previous_actions: Previous actions [num_envs, action_dim].
            step_dt: Physics timestep duration.

        Returns:
            total_reward: Combined reward [num_envs].
            reward_components: Dictionary of individual reward terms.
        """
        # Update trajectory tracking
        self._closest_traj_point, self._contour_error, self._traj_tangent = self._trajectory.update_progress(
            robot.data.root_pos_w
        )

        # Compute individual reward components
        progress_reward = self._compute_progress_reward()
        contour_reward = self._compute_contour_reward()
        velocity_alignment_reward = self._compute_velocity_alignment_reward(robot)
        speed_reward = self._compute_speed_tracking_reward(robot)
        velocity_limit_penalty = self._compute_velocity_limit_penalty(robot)
        orientation_penalty = self._compute_orientation_penalty(robot)
        ang_vel_penalty = self._compute_angular_velocity_penalty(robot)
        action_smoothness_penalty = self._compute_action_smoothness_penalty(actions, previous_actions)

        # Scale and combine rewards
        rewards = {
            "progress": progress_reward * self.cfg.progress_scale * step_dt,
            "contour_error": contour_reward * self.cfg.contour_error_scale * step_dt,
            "velocity_alignment": velocity_alignment_reward * self.cfg.velocity_alignment_scale * step_dt,
            "speed_tracking": speed_reward * self.cfg.speed_tracking_scale * step_dt,
            "velocity_limit_penalty": velocity_limit_penalty * self.cfg.velocity_limit_penalty_scale * step_dt,
            "orientation_penalty": orientation_penalty * self.cfg.orientation_penalty_scale * step_dt,
            "ang_vel_penalty": ang_vel_penalty * self.cfg.ang_vel_penalty_scale * step_dt,
            "action_smoothness": action_smoothness_penalty * self.cfg.action_smoothness_scale * step_dt,
        }

        # Update episode statistics
        for key, value in rewards.items():
            self._episode_sums[key] += value

        # Total reward
        total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        return total_reward, rewards

    def _compute_progress_reward(self) -> torch.Tensor:
        """Compute reward for forward progress along trajectory."""
        current_progress = self._trajectory.progress
        progress_delta = current_progress - self._previous_progress

        # Handle wraparound
        half_length = self._trajectory.total_arc_length / 2
        progress_delta = torch.where(
            progress_delta < -half_length,
            progress_delta + self._trajectory.total_arc_length,
            progress_delta,
        )
        progress_delta = torch.where(
            progress_delta > half_length,
            progress_delta - self._trajectory.total_arc_length,
            progress_delta,
        )

        self._previous_progress = current_progress
        return progress_delta  # Returns velocity (m/s when divided by dt)

    def _compute_contour_reward(self) -> torch.Tensor:
        """Compute reward for staying close to trajectory (exponential decay with distance).
        
        Penalty is amplified near gates to encourage precise passage.
        """
        # Get gate penalty multiplier
        gate_multiplier = self._trajectory.get_gate_penalty_multiplier(self._closest_traj_point)
        
        # Apply multiplier to contour error penalty
        return torch.exp(-2.0 * self._contour_error * gate_multiplier)

    def _compute_velocity_alignment_reward(self, robot: Articulation) -> torch.Tensor:
        """Compute reward for velocity aligned with trajectory direction."""
        velocity_world = robot.data.root_lin_vel_w
        speed = torch.norm(velocity_world, dim=1)
        velocity_normalized = velocity_world / (speed.unsqueeze(1) + 1e-6)

        # Dot product with trajectory tangent
        alignment = torch.sum(velocity_normalized * self._traj_tangent, dim=1)

        # Only reward forward alignment
        return torch.clamp(alignment, 0.0, 1.0)

    def _compute_speed_tracking_reward(self, robot: Articulation) -> torch.Tensor:
        """Compute reward for maintaining desired speed."""
        velocity_world = robot.data.root_lin_vel_w
        speed = torch.norm(velocity_world, dim=1)
        desired_speed = self._trajectory.cfg.desired_speed

        speed_error = torch.abs(speed - desired_speed)
        return torch.exp(-speed_error)

    def _compute_velocity_limit_penalty(self, robot: Articulation) -> torch.Tensor:
        """Compute penalty for exceeding velocity limit."""
        velocity_world = robot.data.root_lin_vel_w
        speed = torch.norm(velocity_world, dim=1)
        velocity_limit = self._trajectory.velocity_limit

        # Only penalize when over limit
        velocity_excess = torch.clamp(speed - velocity_limit, min=0.0)
        return torch.square(velocity_excess)

    def _compute_orientation_penalty(self, robot: Articulation) -> torch.Tensor:
        """Compute penalty for non-upright orientation."""
        # Gravity in body frame should point down (0, 0, -1)
        gravity_z = robot.data.projected_gravity_b[:, 2]
        return torch.square(gravity_z + 1.0)

    def _compute_angular_velocity_penalty(self, robot: Articulation) -> torch.Tensor:
        """Compute penalty for excessive angular velocity."""
        return torch.sum(torch.square(robot.data.root_ang_vel_b), dim=1)

    def _compute_action_smoothness_penalty(
        self, actions: torch.Tensor, previous_actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute penalty for large control changes."""
        action_diff = actions - previous_actions
        return torch.sum(torch.square(action_diff), dim=1)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset reward tracking for specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        self._previous_progress[env_ids] = self._trajectory.progress[env_ids]
        for key in self._episode_sums.keys():
            self._episode_sums[key][env_ids] = 0.0

    def get_episode_stats(self, env_ids: torch.Tensor, max_episode_length_s: float) -> dict[str, float]:
        """Get episode statistics for logging.

        Args:
            env_ids: Environment indices that are resetting.
            max_episode_length_s: Maximum episode length in seconds.

        Returns:
            Dictionary of averaged reward components.
        """
        stats = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            stats[f"Episode_Reward/{key}"] = (episodic_sum_avg / max_episode_length_s).item()
        return stats

    @property
    def contour_error(self) -> torch.Tensor:
        """Current contour error for all environments."""
        return self._contour_error


@dataclass
class ObservationConfig:
    """Configuration for observation generation."""

    history_length: int = 3
    lookahead_distances: tuple[float, ...] = (0.3, 0.6, 1.0, 1.3, 1.6, 2.0)


class ObservationManager:
    """Manages observation generation for quadcopter racing."""

    def __init__(
        self,
        cfg: ObservationConfig,
        num_envs: int,
        action_dim: int,
        device: str,
        trajectory: RacingTrajectory,
    ):
        """Initialize observation manager.

        Args:
            cfg: Observation configuration.
            num_envs: Number of parallel environments.
            action_dim: Dimension of action space.
            device: Device for tensor operations.
            trajectory: Racing trajectory instance.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self._trajectory = trajectory

        # History buffers [num_envs, history_length, feature_dim]
        self._lin_vel_history = torch.zeros(num_envs, cfg.history_length, 3, device=device)
        self._ang_vel_history = torch.zeros(num_envs, cfg.history_length, 3, device=device)
        self._action_history = torch.zeros(num_envs, cfg.history_length, action_dim, device=device)

    def compute_observations(self, robot: Articulation, actions: torch.Tensor) -> torch.Tensor:
        """Compute observation vector for the policy network.

        Args:
            robot: Robot articulation with state data.
            actions: Current actions [num_envs, action_dim].

        Returns:
            observations: Combined observation vector [num_envs, obs_dim].
        """
        # Get look-ahead points along trajectory
        lookahead_points_w = self._trajectory.get_lookahead_points(robot.data.root_pos_w)
        lookahead_points_b = self._transform_lookahead_to_body_frame(robot, lookahead_points_w)

        # Get velocity limit
        velocity_limit = self._trajectory.velocity_limit.unsqueeze(-1)

        # Flatten history buffers
        lin_vel_hist_flat = self._lin_vel_history.reshape(self.num_envs, -1)
        ang_vel_hist_flat = self._ang_vel_history.reshape(self.num_envs, -1)
        action_hist_flat = self._action_history.reshape(self.num_envs, -1)

        # Combine all observations
        obs = torch.cat(
            [
                robot.data.root_lin_vel_b,  # [3] current linear velocity
                robot.data.root_ang_vel_b,  # [3] current angular velocity
                robot.data.projected_gravity_b,  # [3] gravity direction
                velocity_limit,  # [1] current velocity limit
                lookahead_points_b,  # [num_lookahead * 3] trajectory lookahead points
                lin_vel_hist_flat,  # [history_length * 3] past linear velocities
                ang_vel_hist_flat,  # [history_length * 3] past angular velocities
                action_hist_flat,  # [history_length * action_dim] past actions
            ],
            dim=-1,
        )

        return obs

    def _transform_lookahead_to_body_frame(
        self, robot: Articulation, lookahead_points_w: torch.Tensor
    ) -> torch.Tensor:
        """Transform look-ahead points from world to body frame.

        Args:
            robot: Robot articulation with pose data.
            lookahead_points_w: Lookahead points in world frame [num_envs, num_lookahead, 3].

        Returns:
            Flattened lookahead points in body frame [num_envs, num_lookahead * 3].
        """
        num_lookahead = lookahead_points_w.shape[1]
        lookahead_points_b = torch.zeros_like(lookahead_points_w)

        for i in range(num_lookahead):
            lookahead_points_b[:, i, :], _ = subtract_frame_transforms(
                robot.data.root_pos_w,
                robot.data.root_quat_w,
                lookahead_points_w[:, i, :],
            )

        # Flatten: [num_envs, num_lookahead * 3]
        return lookahead_points_b.reshape(self.num_envs, -1)

    def update_history(self, robot: Articulation, actions: torch.Tensor) -> None:
        """Update observation history buffers.

        Args:
            robot: Robot articulation with current state.
            actions: Current actions [num_envs, action_dim].
        """
        # Roll history: move timestep t to t-1, t-1 to t-2, etc.
        self._lin_vel_history = torch.roll(self._lin_vel_history, shifts=1, dims=1)
        self._ang_vel_history = torch.roll(self._ang_vel_history, shifts=1, dims=1)
        self._action_history = torch.roll(self._action_history, shifts=1, dims=1)

        # Store current values at index 0 (most recent)
        self._lin_vel_history[:, 0, :] = robot.data.root_lin_vel_b
        self._ang_vel_history[:, 0, :] = robot.data.root_ang_vel_b
        self._action_history[:, 0, :] = actions

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset observation history for specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        self._lin_vel_history[env_ids] = 0.0
        self._ang_vel_history[env_ids] = 0.0
        self._action_history[env_ids] = 0.0

    @staticmethod
    def get_observation_dim(cfg: ObservationConfig, action_dim: int) -> int:
        """Calculate total observation dimension.

        Args:
            cfg: Observation configuration.
            action_dim: Dimension of action space.

        Returns:
            Total observation dimension.
        """
        base_dim = 10  # lin_vel(3) + ang_vel(3) + gravity(3) + velocity_limit(1)
        lookahead_dim = len(cfg.lookahead_distances) * 3
        history_dim = (3 + 3 + action_dim) * cfg.history_length
        return base_dim + lookahead_dim + history_dim


class DisturbanceManager:
    """Manages dynamic disturbances for sim2real robustness with difficulty-based scaling."""

    def __init__(
        self,
        num_envs: int,
        device: str,
        robot_weight: float,
        force_scale: float = 0.05,
        torque_scale: float = 0.02,
        enabled: bool = True,
        difficulty_dependent: bool = True,
        difficulty_threshold: float = 0.33,
        max_force_scale_at_max_difficulty: float = 0.15,
        max_torque_scale_at_max_difficulty: float = 0.06,
    ):
        """Initialize disturbance manager.

        Args:
            num_envs: Number of parallel environments.
            device: Device for tensor operations.
            robot_weight: Weight of the robot in Newtons.
            force_scale: Base scale of force disturbances (used below threshold).
            torque_scale: Base scale of torque disturbances (used below threshold).
            enabled: Whether disturbances are enabled.
            difficulty_dependent: Whether to scale disturbances based on difficulty.
            difficulty_threshold: Difficulty below which minimal disturbance is applied.
            max_force_scale_at_max_difficulty: Maximum force scale at difficulty=1.0.
            max_torque_scale_at_max_difficulty: Maximum torque scale at difficulty=1.0.
        """
        self.num_envs = num_envs
        self.device = device
        self.robot_weight = robot_weight
        self.base_force_scale = force_scale
        self.base_torque_scale = torque_scale
        self.enabled = enabled
        self.difficulty_dependent = difficulty_dependent
        self.difficulty_threshold = difficulty_threshold
        self.max_force_scale = max_force_scale_at_max_difficulty
        self.max_torque_scale = max_torque_scale_at_max_difficulty

        # Per-environment episode disturbance std dev (sampled at episode start)
        self._episode_force_std = torch.ones(num_envs, device=device) * force_scale
        self._episode_torque_std = torch.ones(num_envs, device=device) * torque_scale

        # Disturbance tensors
        self._force_disturbance = torch.zeros(num_envs, 1, 3, device=device)
        self._torque_disturbance = torch.zeros(num_envs, 1, 3, device=device)

    def sample_disturbances(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample random disturbances for current timestep using per-episode std dev.

        Returns:
            force_disturbance: Random force [num_envs, 1, 3].
            torque_disturbance: Random torque [num_envs, 1, 3].
        """
        if self.enabled:
            # Sample using per-environment episode-specific std dev
            self._force_disturbance[:, 0, :] = (
                torch.randn(self.num_envs, 3, device=self.device) * 
                self._episode_force_std.unsqueeze(1) * self.robot_weight
            )
            self._torque_disturbance[:, 0, :] = (
                torch.randn(self.num_envs, 3, device=self.device) * 
                self._episode_torque_std.unsqueeze(1)
            )
            )
        else:
            self._force_disturbance.zero_()
            self._torque_disturbance.zero_()

        return self._force_disturbance, self._torque_disturbance

    def reset_episode_disturbances(self, env_ids: torch.Tensor, difficulties: torch.Tensor | None = None) -> None:
        """Reset episode-specific disturbance std dev based on difficulty.
        
        For difficulty < threshold: minimal disturbance (base scales)
        For difficulty >= threshold: linear scaling from base to max
        At episode start, sample random std dev from uniform range [0, max_for_difficulty]
        
        Args:
            env_ids: Environment indices to reset.
            difficulties: Difficulty levels [0, 1] for each environment. If None, uses base scales.
        """
        if not self.difficulty_dependent or difficulties is None:
            # Use base scales without difficulty dependence
            self._episode_force_std[env_ids] = self.base_force_scale
            self._episode_torque_std[env_ids] = self.base_torque_scale
            return
        
        # Compute max allowed std dev based on difficulty
        difficulties_batch = difficulties[env_ids]
        
        # Below threshold: use minimal disturbance (base scale)
        below_threshold = difficulties_batch < self.difficulty_threshold
        
        # Above threshold: linear interpolation from base to max
        # Normalize difficulty from [threshold, 1.0] to [0, 1]
        normalized_difficulty = torch.clamp(
            (difficulties_batch - self.difficulty_threshold) / (1.0 - self.difficulty_threshold),
            0.0, 1.0
        )
        
        max_force_std = self.base_force_scale + normalized_difficulty * (self.max_force_scale - self.base_force_scale)
        max_torque_std = self.base_torque_scale + normalized_difficulty * (self.max_torque_scale - self.base_torque_scale)
        
        # Apply minimal disturbance below threshold
        max_force_std = torch.where(below_threshold, 
                                    torch.tensor(self.base_force_scale, device=self.device),
                                    max_force_std)
        max_torque_std = torch.where(below_threshold,
                                     torch.tensor(self.base_torque_scale, device=self.device),
                                     max_torque_std)
        
        # Sample random std dev from uniform [0, max_for_difficulty]
        self._episode_force_std[env_ids] = torch.rand(len(env_ids), device=self.device) * max_force_std
        self._episode_torque_std[env_ids] = torch.rand(len(env_ids), device=self.device) * max_torque_std
