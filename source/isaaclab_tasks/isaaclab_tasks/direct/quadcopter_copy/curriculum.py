"""Curriculum learning manager for progressive trajectory difficulty."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@dataclass
class CurriculumCfg:
    """Configuration for curriculum learning."""
    
    # Thresholds for advancing difficulty
    contour_error_threshold: float = 0.15  # Max contour error to advance (meters)
    progress_velocity_threshold: float = 0.8  # Min progress velocity to advance (m/s)
    success_streak_required: int = 50  # Number of consecutive successes to advance
    
    # Difficulty progression
    initial_difficulty: float = 0.0  # Start with easiest trajectories
    max_difficulty: float = 1.0  # Maximum difficulty level
    
    # Continuous progression parameters
    use_continuous_progression: bool = True  # Use performance-based continuous adjustment
    base_increment: float = 0.05  # Base increment for continuous progression
    max_increment: float = 0.15  # Maximum single step increment
    base_decrement: float = 0.05  # Base decrement for continuous progression
    max_decrement: float = 0.15  # Maximum single step decrement
    
    # Demotion on failure
    enable_demotion: bool = True  # Whether to decrease difficulty on failure
    failure_streak_threshold: int = 100  # Consecutive failures before demotion


class TrajectoryCurriculumManager:
    """Manages curriculum learning for trajectory difficulty progression.
    
    Tracks per-environment performance and automatically adjusts trajectory
    difficulty based on agent success. Inspired by Isaac Lab's DifficultyScheduler
    pattern used in manipulation tasks.
    """

    def __init__(
        self,
        num_envs: int,
        device: str = "cuda",
        cfg: CurriculumCfg | None = None,
    ):
        """Initialize curriculum manager.

        Args:
            num_envs: Number of parallel environments.
            device: Device for tensor operations.
            cfg: Configuration for curriculum behavior.
        """
        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg if cfg is not None else CurriculumCfg()

        # Per-environment difficulty levels [0.0, 1.0]
        self.difficulty_levels = torch.full(
            (num_envs,), self.cfg.initial_difficulty, device=device, dtype=torch.float32
        )

        # Performance tracking
        self.success_streaks = torch.zeros(num_envs, device=device, dtype=torch.int32)
        self.failure_streaks = torch.zeros(num_envs, device=device, dtype=torch.int32)

        # Episode statistics for monitoring
        self.total_advancements = torch.zeros(num_envs, device=device, dtype=torch.int32)
        self.total_demotions = torch.zeros(num_envs, device=device, dtype=torch.int32)

    def update(
        self,
        contour_errors: torch.Tensor,
        progress_velocities: torch.Tensor,
        reset_ids: torch.Tensor,
    ) -> None:
        """Update curriculum based on episode performance.

        Should be called at episode termination for environments that reset.

        Args:
            contour_errors: [num_envs] Mean contour error over the episode (meters).
            progress_velocities: [num_envs] Mean progress velocity over the episode (m/s).
            reset_ids: Indices of environments that just terminated.
        """
        if len(reset_ids) == 0:
            return

        # Check success criteria for environments that reset
        success_mask = (
            (contour_errors[reset_ids] < self.cfg.contour_error_threshold) &
            (progress_velocities[reset_ids] > self.cfg.progress_velocity_threshold)
        )

        # Update success streaks
        self.success_streaks[reset_ids] = torch.where(
            success_mask,
            self.success_streaks[reset_ids] + 1,
            torch.zeros_like(self.success_streaks[reset_ids])
        )

        # Update failure streaks
        self.failure_streaks[reset_ids] = torch.where(
            ~success_mask,
            self.failure_streaks[reset_ids] + 1,
            torch.zeros_like(self.failure_streaks[reset_ids])
        )

        # Advance difficulty for environments with sufficient success streak
        advance_mask = self.success_streaks[reset_ids] >= self.cfg.success_streak_required
        if advance_mask.any():
            advance_ids = reset_ids[advance_mask]
            
            if self.cfg.use_continuous_progression:
                # Continuous progression based on performance
                # Better performance = larger increment
                error_quality = 1.0 - torch.clamp(
                    contour_errors[advance_ids] / self.cfg.contour_error_threshold, 0.0, 1.0
                )
                velocity_quality = torch.clamp(
                    progress_velocities[advance_ids] / (self.cfg.progress_velocity_threshold * 2.0), 0.0, 1.0
                )
                performance_quality = (error_quality + velocity_quality) / 2.0
                
                # Scale increment based on performance (better = faster progression)
                increments = self.cfg.base_increment + performance_quality * (self.cfg.max_increment - self.cfg.base_increment)
                
                self.difficulty_levels[advance_ids] = torch.clamp(
                    self.difficulty_levels[advance_ids] + increments,
                    0.0,
                    self.cfg.max_difficulty
                )
            else:
                # Fixed increment
                self.difficulty_levels[advance_ids] = torch.clamp(
                    self.difficulty_levels[advance_ids] + self.cfg.base_increment,
                    0.0,
                    self.cfg.max_difficulty
                )
            
            self.success_streaks[advance_ids] = 0  # Reset streak after advancement
            self.total_advancements[advance_ids] += 1

        # Demote difficulty for environments with excessive failures
        if self.cfg.enable_demotion:
            demote_mask = self.failure_streaks[reset_ids] >= self.cfg.failure_streak_threshold
            if demote_mask.any():
                demote_ids = reset_ids[demote_mask]
                
                if self.cfg.use_continuous_progression:
                    # Continuous demotion based on how badly they failed
                    # Worse performance = larger decrement
                    error_penalty = torch.clamp(
                        (contour_errors[demote_ids] - self.cfg.contour_error_threshold) / self.cfg.contour_error_threshold,
                        0.0, 1.0
                    )
                    velocity_penalty = torch.clamp(
                        (self.cfg.progress_velocity_threshold - progress_velocities[demote_ids]) / self.cfg.progress_velocity_threshold,
                        0.0, 1.0
                    )
                    failure_severity = (error_penalty + velocity_penalty) / 2.0
                    
                    # Scale decrement based on failure severity
                    decrements = self.cfg.base_decrement + failure_severity * (self.cfg.max_decrement - self.cfg.base_decrement)
                    
                    self.difficulty_levels[demote_ids] = torch.clamp(
                        self.difficulty_levels[demote_ids] - decrements,
                        0.0,
                        self.cfg.max_difficulty
                    )
                else:
                    # Fixed decrement
                    self.difficulty_levels[demote_ids] = torch.clamp(
                        self.difficulty_levels[demote_ids] - self.cfg.base_decrement,
                        0.0,
                        self.cfg.max_difficulty
                    )
                
                self.failure_streaks[demote_ids] = 0  # Reset streak after demotion
                self.total_demotions[demote_ids] += 1

    def get_difficulty(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Get current difficulty level for specified environments.

        Args:
            env_ids: Environment indices. If None, returns all difficulties.

        Returns:
            difficulty: [len(env_ids)] or [num_envs] Difficulty levels in [0, 1].
        """
        if env_ids is None:
            return self.difficulty_levels
        return self.difficulty_levels[env_ids]

    def get_statistics(self) -> dict[str, float]:
        """Get curriculum statistics for logging.

        Returns:
            Dictionary with mean difficulty, advancements, demotions, etc.
        """
        return {
            "curriculum/mean_difficulty": self.difficulty_levels.mean().item(),
            "curriculum/min_difficulty": self.difficulty_levels.min().item(),
            "curriculum/max_difficulty": self.difficulty_levels.max().item(),
            "curriculum/mean_success_streak": self.success_streaks.float().mean().item(),
            "curriculum/mean_failure_streak": self.failure_streaks.float().mean().item(),
            "curriculum/total_advancements": self.total_advancements.sum().item(),
            "curriculum/total_demotions": self.total_demotions.sum().item(),
        }

    def reset_env_stats(self, env_ids: torch.Tensor) -> None:
        """Reset statistics for specified environments (useful for debugging).

        Args:
            env_ids: Environment indices to reset.
        """
        self.success_streaks[env_ids] = 0
        self.failure_streaks[env_ids] = 0

    def set_difficulty(self, env_ids: torch.Tensor, difficulty: float) -> None:
        """Manually set difficulty for specified environments (for testing).

        Args:
            env_ids: Environment indices.
            difficulty: Difficulty level [0, 1] to set.
        """
        self.difficulty_levels[env_ids] = torch.clamp(
            torch.tensor(difficulty, device=self.device),
            0.0,
            self.cfg.max_difficulty
        )
