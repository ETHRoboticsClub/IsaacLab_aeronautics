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
    
    # Performance thresholds for immediate adjustment
    good_contour_error_threshold: float = 0.10  # Max error for "good" performance (advance)
    bad_contour_error_threshold: float = 0.25  # Min error for "bad" performance (demote)
    good_progress_velocity_threshold: float = 1.0  # Min velocity for "good" performance (advance)
    bad_progress_velocity_threshold: float = 0.5  # Max velocity for "bad" performance (demote)
    
    # Difficulty progression
    initial_difficulty: float = 0.0  # Start with easiest trajectories
    max_difficulty: float = 1.0  # Maximum difficulty level
    
    # Adjustment amounts (randomly sampled from [0, max_increment/decrement])
    max_advancement_increment: float = 0.10  # Max increase when performance is good
    max_demotion_decrement: float = 0.10  # Max decrease when performance is bad


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

        # Episode statistics for monitoring
        self.total_promotions = torch.zeros(num_envs, device=device, dtype=torch.int32)
        self.total_demotions = torch.zeros(num_envs, device=device, dtype=torch.int32)
        self.total_no_change = torch.zeros(num_envs, device=device, dtype=torch.int32)

    def update(
        self,
        contour_errors: torch.Tensor,
        progress_velocities: torch.Tensor,
        reset_ids: torch.Tensor,
    ) -> None:
        """Update curriculum based on immediate episode performance.

        Advances if performance is good, demotes if bad, stays same if ok-ish.
        No streak tracking - every episode is evaluated independently.

        Args:
            contour_errors: [len(reset_ids)] Mean contour error over the episode for resetting envs only (meters).
            progress_velocities: [len(reset_ids)] Mean progress velocity over the episode for resetting envs only (m/s).
            reset_ids: Indices of environments that just terminated.
        """
        if len(reset_ids) == 0:
            return

        # Classify performance as good, bad, or ok-ish (using full reset data, no additional indexing)
        good_performance = (
            (contour_errors < self.cfg.good_contour_error_threshold) &
            (progress_velocities > self.cfg.good_progress_velocity_threshold)
        )
        
        bad_performance = (
            (contour_errors > self.cfg.bad_contour_error_threshold) |
            (progress_velocities < self.cfg.bad_progress_velocity_threshold)
        )

        # Advance difficulty immediately for good performance
        if good_performance.any():
            advance_ids = reset_ids[good_performance]
            # Random increment in [0, max_advancement_increment]
            random_increments = torch.rand(len(advance_ids), device=self.device) * self.cfg.max_advancement_increment
            self.difficulty_levels[advance_ids] = torch.clamp(
                self.difficulty_levels[advance_ids] + random_increments,
                0.0,
                self.cfg.max_difficulty
            )
            self.total_promotions[advance_ids] += 1

        # Demote difficulty immediately for bad performance
        if bad_performance.any():
            demote_ids = reset_ids[bad_performance]
            # Random decrement in [0, max_demotion_decrement]
            random_decrements = torch.rand(len(demote_ids), device=self.device) * self.cfg.max_demotion_decrement
            self.difficulty_levels[demote_ids] = torch.clamp(
                self.difficulty_levels[demote_ids] - random_decrements,
                0.0,
                self.cfg.max_difficulty
            )
            self.total_demotions[demote_ids] += 1
        
        # Track ok-ish performance (no change)
        ok_ish_performance = ~(good_performance | bad_performance)
        if ok_ish_performance.any():
            ok_ish_ids = reset_ids[ok_ish_performance]
            self.total_no_change[ok_ish_ids] += 1

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
            "curriculum/std_difficulty": self.difficulty_levels.std().item(),
            "curriculum/total_promotions": self.total_promotions.sum().item(),
            "curriculum/total_demotions": self.total_demotions.sum().item(),
            "curriculum/total_no_change": self.total_no_change.sum().item(),
        }

    def reset_env_stats(self, env_ids: torch.Tensor) -> None:
        """Reset statistics for specified environments (useful for debugging).

        Args:
            env_ids: Environment indices to reset.
        """
        self.total_promotions[env_ids] = 0
        self.total_demotions[env_ids] = 0
        self.total_no_change[env_ids] = 0

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
