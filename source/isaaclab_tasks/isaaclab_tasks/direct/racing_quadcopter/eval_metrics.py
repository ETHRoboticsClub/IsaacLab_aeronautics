from __future__ import annotations

import torch


class EvalMetricsTracker:
    """Vectorised per-gate, per-environment evaluation metrics.

    Args:
        num_envs:  Number of parallel environments.
        num_gates: Number of gates per lap.
        device:    Torch device string (e.g. ``"cuda:0"``).
    """

    def __init__(self, num_envs: int, num_gates: int, device: str) -> None:
        self.num_envs  = num_envs
        self.num_gates = num_gates
        self.device    = device

        # ------------------------------------------------------------------ #
        # Per-episode state  (reset for each env when its episode ends)       #
        # ------------------------------------------------------------------ #

        # Cumulative speed and step count for computing mean velocity
        self._ep_speed_sum = torch.zeros(num_envs, device=device)
        self._ep_steps     = torch.zeros(num_envs, device=device)

        # Speed recorded at each gate passage within the current episode
        # shape: [num_envs, num_gates]
        self._ep_gate_speed = torch.zeros(num_envs, num_gates, device=device)

        # ------------------------------------------------------------------ #
        # Cross-episode accumulators                                          #
        # ------------------------------------------------------------------ #

        # Number of completed episodes per env
        self._n_episodes = torch.zeros(num_envs, device=device)

        # For each gate: how many episodes (per env) was that gate passed?
        # shape: [num_envs, num_gates]
        self._sum_gate_passages = torch.zeros(num_envs, num_gates, device=device)

        # For each gate: sum of speed-at-passage (only where gate was passed)
        # shape: [num_envs, num_gates]
        self._sum_gate_speed = torch.zeros(num_envs, num_gates, device=device)

        # Episode-level sums (per env, averaged later across envs too)
        self._sum_laps     = torch.zeros(num_envs, device=device)
        self._sum_duration = torch.zeros(num_envs, device=device)
        self._sum_speed    = torch.zeros(num_envs, device=device)

        # Pre-allocated gate index tensor to avoid re-creating it every step
        self._gate_idx = torch.arange(num_gates, device=device)

    # ---------------------------------------------------------------------- #
    # Per-step update                                                         #
    # ---------------------------------------------------------------------- #

    def on_step(self, speed: torch.Tensor) -> None:
        """Accumulate speed for all envs.  Call once per simulation step.

        Args:
            speed: Current drone speed, shape ``[num_envs]``.
        """
        self._ep_speed_sum.add_(speed)
        self._ep_steps.add_(1.0)

    # ---------------------------------------------------------------------- #
    # Gate-passage event                                                      #
    # ---------------------------------------------------------------------- #

    def on_gate_passage(
        self,
        env_ids:     torch.Tensor,  # [K]
        gate_idx:    torch.Tensor,  # [K]
        speed:       torch.Tensor,  # [K]
    ) -> None:
        """Record a gate passage event for a subset of environments.

        Args:
            env_ids:  Indices of envs that just passed a gate.
            gate_idx: Gate index (0 … num_gates-1) passed by each env.
            speed:    Drone speed (m/s) at passage for each env.
        """
        self._ep_gate_speed[env_ids, gate_idx] = speed

    # ---------------------------------------------------------------------- #
    # Episode-end event                                                       #
    # ---------------------------------------------------------------------- #

    def on_episode_end(
        self,
        env_ids:      torch.Tensor,  # [K]
        gates_passed: torch.Tensor,  # [K]
        step_dt:      float,
    ) -> None:
        """Finalise metrics for a batch of environments that just finished.

        Args:
            env_ids:      Indices of the environments that are being reset.
            gates_passed: How many gates each env completed this episode.
            step_dt:      Simulation step duration in seconds.
        """
        # Ensure 1-D inputs regardless of how the caller shapes them
        env_ids      = env_ids.view(-1)
        gates_passed = gates_passed.view(-1)

        if env_ids.numel() == 0:
            return

        # Increment episode counter
        self._n_episodes[env_ids] += 1

        # --- Per-gate passage ---
        # Gate g was passed if gates_passed > g  (gates are sequential)
        # passed_mask: [K, num_gates]  True where the gate was passed
        passed_mask = self._gate_idx.unsqueeze(0) < gates_passed.unsqueeze(1)
        self._sum_gate_passages[env_ids] += passed_mask.float()
        self._sum_gate_speed[env_ids]    += self._ep_gate_speed[env_ids] * passed_mask.float()

        # --- Episode-level metrics ---
        laps = (gates_passed.float() / self.num_gates)
        self._sum_laps[env_ids] += laps

        duration = self._ep_steps[env_ids] * step_dt
        self._sum_duration[env_ids] += duration

        valid_steps = self._ep_steps[env_ids] > 0
        mean_speed = torch.where(
            valid_steps,
            self._ep_speed_sum[env_ids] / self._ep_steps[env_ids].clamp(min=1.0),
            torch.zeros(env_ids.numel(), device=self.device),
        )
        self._sum_speed[env_ids] += mean_speed

        # Reset per-episode buffers for the done envs
        self._ep_speed_sum[env_ids]  = 0.0
        self._ep_steps[env_ids]      = 0.0
        self._ep_gate_speed[env_ids] = 0.0

    # ---------------------------------------------------------------------- #
    # Properties and summary                                                  #
    # ---------------------------------------------------------------------- #

    @property
    def total_episodes(self) -> int:
        """Total number of completed episodes across all environments."""
        return int(self._n_episodes.sum().item())

    @property
    def gate_passage_rates(self) -> torch.Tensor:
        """Fraction of all episodes where each gate was passed.

        Returns:
            Tensor of shape ``[num_gates]``, values in ``[0, 1]``.
        """
        n = self._n_episodes.sum().clamp(min=1.0)
        return self._sum_gate_passages.sum(dim=0) / n

    @property
    def gate_mean_speed(self) -> torch.Tensor:
        """Mean drone speed (m/s) at each gate passage.

        Returns:
            Tensor of shape ``[num_gates]``.  Zero for gates never passed.
        """
        passages = self._sum_gate_passages.sum(dim=0).clamp(min=1.0)
        return self._sum_gate_speed.sum(dim=0) / passages

    def get_summary(self) -> dict:
        """Compute and return all aggregated metrics.

        Returns:
            Dict with scalar Python floats for episode-level metrics and
            ``torch.Tensor`` for per-gate metrics::

                {
                    "total_episodes":       int,
                    "mean_laps_completed":  float,   # 0–1
                    "mean_velocity_ms":     float,
                    "mean_duration_s":      float,
                    "gate_passage_rates":   Tensor[num_gates],   # 0–1
                    "gate_mean_speed":      Tensor[num_gates],   # m/s
                }
        """
        n = self.total_episodes
        if n == 0:
            return {"total_episodes": 0}

        n_t = float(n)
        return {
            "total_episodes":      n,
            "mean_laps_completed": self._sum_laps.sum().item()     / n_t,
            "mean_velocity_ms":    self._sum_speed.sum().item()    / n_t,
            "mean_duration_s":     self._sum_duration.sum().item() / n_t,
            "gate_passage_rates":  self.gate_passage_rates,
            "gate_mean_speed":     self.gate_mean_speed,
        }

    def print_summary(self, batch_id: int | None = None) -> None:
        """Print a human-readable batch summary to stdout.

        Args:
            batch_id: Optional batch counter shown in the prefix.
        """
        s = self.get_summary()
        if s["total_episodes"] == 0:
            print("[EvalMetrics] No episodes recorded.")
            return

        prefix  = f"[eval] Batch {batch_id}" if batch_id is not None else "[eval]"
        vel_ms  = s["mean_velocity_ms"]
        print(
            f"{prefix} ({s['total_episodes']} episodes): "
            f"lap_completion={s['mean_laps_completed']*100:.1f}%, "
            f"mean_velocity={vel_ms:.2f} m/s ({vel_ms * 3.6:.1f} km/h), "
            f"mean_duration={s['mean_duration_s']:.1f}s"
        )

        rates = s["gate_passage_rates"].tolist()
        speeds = s["gate_mean_speed"].tolist()
        rates_str  = " ".join(f"{r:.0%}" for r in rates)
        speeds_str = " ".join(f"{v:.1f}" for v in speeds)
        print(f"  Gate passage rates : [{rates_str}]")
        print(f"  Gate mean speed m/s: [{speeds_str}]")

    # ---------------------------------------------------------------------- #
    # Reset                                                                   #
    # ---------------------------------------------------------------------- #

    def reset_accumulators(self) -> None:
        """Reset all cross-episode accumulators (call after printing a batch)."""
        self._n_episodes.zero_()
        self._sum_gate_passages.zero_()
        self._sum_gate_speed.zero_()
        self._sum_laps.zero_()
        self._sum_duration.zero_()
        self._sum_speed.zero_()
