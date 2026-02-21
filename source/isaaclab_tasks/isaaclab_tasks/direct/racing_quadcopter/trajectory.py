# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory generation and tracking utilities for quadcopter racing."""

from __future__ import annotations

import torch
import math
from dataclasses import dataclass
from typing import Optional

from .trajectory_generator import TrajectoryLibrary, TrajectoryGeneratorCfg


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory generation."""

    trajectory_type: str = "library"  # Options: "circle", "oval", "figure8", "library"
    radius: float = 2.0  # Base radius for circular trajectories (when not using library)
    height: float = 1.5  # Flight height (when not using library)
    waypoints_density: float = 0.5  # Density of waypoints per meter
    desired_speed: float = 2.0  # Target speed along trajectory (m/s)
    
    # Look-ahead distances for observations (in meters along arc-length)
    lookahead_distances: tuple[float, ...] = (0.0, 0.2, 0.5, 0.9, 1.4, 2.0)
    
    # Virtual gate configuration
    enable_gates: bool = True  # Enable virtual gates along trajectory
    gate_spacing: float = 10.0  # Distance between gates (meters along arc-length)
    gate_size: float = 2.0  # Size of gate opening (meters)
    gate_penalty_multiplier: float = 5.0  # Contour error penalty multiplier near gates
    gate_collision_threshold: float = 1.0  # Distance threshold for gate collision (meters)
    
    # Trajectory library configuration
    use_trajectory_library: bool = True  # Use precomputed racing tracks
    track_seed: Optional[int] = 42  # Seed for trajectory generation
    trajectories_per_env: int = 3  # Number of different trajectories per environment (1 = single trajectory per env)
    trajectory_generator_cfg: Optional[TrajectoryGeneratorCfg] = None  # Config for trajectory generation


class RacingTrajectory:
    """Manages racing trajectory with waypoints and progress tracking.
    
    Each environment has its own library of 100 trajectories at different difficulties,
    all pre-offset to the environment's world position.
    """

    def __init__(self, cfg: TrajectoryConfig, device: str, num_envs: int):
        """Initialize trajectory.

        Args:
            cfg: Trajectory configuration.
            device: Device for tensor operations.
            num_envs: Number of parallel environments.
        """
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs
        
        # Store environment origins (set later via set_environment_origins)
        self.env_origins = torch.zeros(num_envs, 3, device=device)
        
        # Per-environment trajectory libraries will be created after env_origins are set
        # Structure: self.env_trajectories[env_id] = list of (waypoints, tangents) tuples
        # Each waypoints tensor is [num_waypoints, 3], already offset to env origin
        self.env_trajectories: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        self.num_trajectories_per_env = cfg.trajectories_per_env  # Each env has N trajectories of varying difficulty
        
        # Initialize trajectory generator config
        self.traj_gen_cfg = cfg.trajectory_generator_cfg if cfg.trajectory_generator_cfg is not None else TrajectoryGeneratorCfg()
        
        # Trajectory library for generation (shared, used to generate trajectories)
        self.trajectory_library: Optional[TrajectoryLibrary] = None
        if cfg.use_trajectory_library:
            print(f"Will generate {num_envs} trajectory libraries with {self.num_trajectories_per_env} trajectory(ies) each...")
            print(f"Using adaptive waypoints density: {cfg.waypoints_density} waypoints/meter")
            
            self.trajectory_library = TrajectoryLibrary(
                waypoints_density=cfg.waypoints_density,
                device=device,
                seed=cfg.track_seed,
                cfg=self.traj_gen_cfg,
                num_environments=num_envs,  # Create one library per environment
            )

        # Per-environment current trajectory index and state
        self.env_trajectory_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        
        # Current active waypoints/tangents per environment (set during reset)
        # These are views/copies from env_trajectories for the currently active trajectory
        self.current_waypoints: Optional[torch.Tensor] = None  # [num_envs, num_waypoints, 3]
        self.current_tangents: Optional[torch.Tensor] = None   # [num_envs, num_waypoints, 3]
        
        # Will be set after trajectories are generated
        self.num_waypoints = 0
        self.segment_lengths: Optional[torch.Tensor] = None
        self.cumulative_arc_length: Optional[torch.Tensor] = None
        self.total_arc_length: Optional[torch.Tensor] = None

        # Per-environment state: current progress (arc-length) and closest waypoint index
        self.progress = torch.zeros(num_envs, device=device)
        self.closest_idx = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Virtual gates (per-environment since each env has different trajectory)
        self.gates_enabled = cfg.enable_gates
        self.gate_positions: Optional[torch.Tensor] = None  # [num_envs, max_gates_per_env, 3]
        self.gate_orientations: Optional[torch.Tensor] = None  # [num_envs, max_gates_per_env, 4]
        self.gate_progress: Optional[torch.Tensor] = None  # [num_envs, max_gates_per_env]
        self.num_gates_per_env: Optional[torch.Tensor] = None  # [num_envs] - actual number of gates per env

    def set_environment_origins(self, env_origins: torch.Tensor) -> None:
        """Generate per-environment trajectory libraries, each offset to its world position.
        
        This is called once after terrain setup. Each environment gets its own
        library of trajectories at different difficulties, all pre-offset.
        
        Args:
            env_origins: [num_envs, 3] environment origin offsets in world coordinates.
        """
        self.env_origins = env_origins
        
        # Generate trajectory library for each environment
        self.env_trajectories = []
        
        for env_id in range(self.num_envs):
            env_origin = env_origins[env_id]  # [3]
            env_library = []
            
            # Generate trajectories for this environment
            for traj_idx in range(self.num_trajectories_per_env):
                if self.trajectory_library is not None:
                    # Get trajectory from shared library (difficulty-ordered)
                    waypoints, tangents = self.trajectory_library.get_trajectory(env_idx=env_id, traj_index=traj_idx)
                else:
                    # Fallback: generate simple trajectory
                    waypoints, tangents = self._generate_simple_trajectory()
                
                # Offset waypoints to this environment's world position
                waypoints_offset = waypoints + env_origin
                
                env_library.append((waypoints_offset, tangents))
            
            self.env_trajectories.append(env_library)
        
        # Initialize current waypoints with first trajectory from each env
        self._set_current_trajectories(torch.zeros(self.num_envs, dtype=torch.long, device=self.device))
        
        # Generate gates for visualization
        if self.gates_enabled:
            self._regenerate_gates()

    def _set_current_trajectories(self, trajectory_ids: torch.Tensor) -> None:
        """Set current active trajectories for all environments.
        
        Args:
            trajectory_ids: [num_envs] trajectory index for each environment
        """
        self.env_trajectory_ids = trajectory_ids
        
        # Stack current waypoints and tangents: [num_envs, num_waypoints, 3]
        waypoints_list = []
        tangents_list = []
        
        for env_id in range(self.num_envs):
            traj_idx = int(trajectory_ids[env_id].item())
            wp, tg = self.env_trajectories[env_id][traj_idx]
            waypoints_list.append(wp)
            tangents_list.append(tg)
        
        # Pad to same size if needed (trajectories may have different num_waypoints)
        max_waypoints = max(wp.shape[0] for wp in waypoints_list)
        self.num_waypoints = max_waypoints
        
        self.current_waypoints = torch.zeros(self.num_envs, max_waypoints, 3, device=self.device)
        self.current_tangents = torch.zeros(self.num_envs, max_waypoints, 3, device=self.device)
        
        for env_id, (wp, tg) in enumerate(zip(waypoints_list, tangents_list)):
            n = wp.shape[0]
            self.current_waypoints[env_id, :n] = wp
            self.current_tangents[env_id, :n] = tg
        
        # Recompute arc lengths for current trajectories
        self._recompute_arc_lengths()

    def _generate_simple_trajectory(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a simple circle trajectory as fallback."""
        n = 50
        theta = torch.linspace(0, 2 * math.pi, n, device=self.device)
        x = self.cfg.radius * torch.cos(theta)
        y = self.cfg.radius * torch.sin(theta)
        z = torch.ones_like(x) * self.cfg.height
        
        waypoints = torch.stack([x, y, z], dim=1)
        tangents = torch.zeros_like(waypoints)
        tangents[:-1] = waypoints[1:] - waypoints[:-1]
        tangents[-1] = waypoints[0] - waypoints[-1]
        tangents = tangents / (torch.norm(tangents, dim=1, keepdim=True) + 1e-8)
        
        return waypoints, tangents

    def _generate_trajectory(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate trajectory waypoints and tangent vectors.

        Returns:
            waypoints: [num_waypoints, 3] trajectory positions
            tangents: [num_waypoints, 3] normalized tangent vectors
        """
        # If using library, return the first trajectory from env 0's library as default
        if self.cfg.use_trajectory_library and self.trajectory_library is not None:
            waypoints, tangents = self.trajectory_library.get_trajectory(env_idx=0, traj_index=0)
            return waypoints, tangents

        # Otherwise generate simple parametric trajectory
        # First generate with coarse sampling to estimate length
        coarse_n = 100  # Use fixed coarse sampling for length estimation
        coarse_theta = torch.linspace(0, 2 * math.pi, coarse_n, device=self.device)
        
        if self.cfg.trajectory_type == "circle":
            coarse_x = self.cfg.radius * torch.cos(coarse_theta)
            coarse_y = self.cfg.radius * torch.sin(coarse_theta)
            coarse_z = torch.ones_like(coarse_x) * self.cfg.height
        elif self.cfg.trajectory_type == "oval":
            a = self.cfg.radius * 1.5
            b = self.cfg.radius * 0.8
            coarse_x = a * torch.cos(coarse_theta)
            coarse_y = b * torch.sin(coarse_theta)
            coarse_z = torch.ones_like(coarse_x) * self.cfg.height
        elif self.cfg.trajectory_type == "figure8":
            scale = self.cfg.radius
            coarse_x = scale * torch.sin(coarse_theta)
            coarse_y = scale * torch.sin(coarse_theta) * torch.cos(coarse_theta)
            coarse_z = self.cfg.height + 0.3 * torch.sin(2 * coarse_theta)
        else:
            raise ValueError(f"Unknown trajectory type: {self.cfg.trajectory_type}")
            
        # Compute actual trajectory length from coarse sampling
        coarse_points = torch.stack([coarse_x, coarse_y, coarse_z], dim=1)
        segment_lengths = torch.norm(coarse_points[1:] - coarse_points[:-1], dim=1)
        trajectory_length = segment_lengths.sum().item()
        
        # Now generate with proper waypoint density
        n = int(trajectory_length * self.cfg.waypoints_density)
        print(f"Generated {self.cfg.trajectory_type} trajectory: length={trajectory_length:.1f}m, density={self.cfg.waypoints_density}/m → {n} waypoints")
        
        theta = torch.linspace(0, 2 * math.pi, n, device=self.device)

        if self.cfg.trajectory_type == "circle":
            x = self.cfg.radius * torch.cos(theta)
            y = self.cfg.radius * torch.sin(theta)
            z = torch.ones_like(x) * self.cfg.height

        elif self.cfg.trajectory_type == "oval":
            # Elliptical track
            a = self.cfg.radius * 1.5  # semi-major axis
            b = self.cfg.radius * 0.8  # semi-minor axis
            x = a * torch.cos(theta)
            y = b * torch.sin(theta)
            z = torch.ones_like(x) * self.cfg.height

        elif self.cfg.trajectory_type == "figure8":
            # Lemniscate (figure-8) in 3D
            scale = self.cfg.radius
            x = scale * torch.sin(theta)
            y = scale * torch.sin(theta) * torch.cos(theta)
            z = self.cfg.height + 0.3 * torch.sin(2 * theta)  # Vertical variation

        else:
            raise ValueError(f"Unknown trajectory type: {self.cfg.trajectory_type}")

        waypoints = torch.stack([x, y, z], dim=1)

        # Compute tangent vectors (forward direction at each waypoint)
        tangents = torch.zeros_like(waypoints)
        tangents[:-1] = waypoints[1:] - waypoints[:-1]
        tangents[-1] = waypoints[0] - waypoints[-1]  # Close the loop
        # Normalize tangents
        tangents = tangents / (torch.norm(tangents, dim=1, keepdim=True) + 1e-8)

        return waypoints, tangents

    def update_progress(self, current_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update progress tracking based on current drone positions.

        Args:
            current_pos: [num_envs, 3] current drone positions

        Returns:
            closest_point: [num_envs, 3] closest point on trajectory
            contour_error: [num_envs] perpendicular distance from trajectory
            tangent: [num_envs, 3] tangent vector at closest point
        """
        # print(f"[TRACE] update_progress START: current_pos shape={current_pos.shape}")
        # Ensure trajectory is initialized
        assert self.current_waypoints is not None, "Trajectory not initialized. Call set_environment_origins first."
        assert self.cumulative_arc_length is not None, "Arc lengths not computed."
        assert self.total_arc_length is not None, "Total arc length not computed."
        
        # Use per-environment current waypoints [num_envs, num_waypoints, 3]
        # Find closest waypoint for each environment
        diffs = current_pos.unsqueeze(1) - self.current_waypoints
        distances = torch.norm(diffs, dim=2)  # [num_envs, num_waypoints]
        
        closest_idx = torch.argmin(distances, dim=1)  # [num_envs]
        self.closest_idx = closest_idx

        env_idx = torch.arange(self.num_envs, device=self.device)

        # Check TWO segments: (prev → closest) and (closest → next)
        # This avoids inflated contour error when drone is between prev and closest waypoint
        prev_idx = (closest_idx - 1) % self.num_waypoints
        next_idx = (closest_idx + 1) % self.num_waypoints
        
        # --- Segment A: closest → next ---
        pA0 = self.current_waypoints[env_idx, closest_idx]  # [num_envs, 3]
        pA1 = self.current_waypoints[env_idx, next_idx]
        segA = pA1 - pA0
        segA_len = torch.norm(segA, dim=1, keepdim=True)
        segA_dir = segA / (segA_len + 1e-8)
        to_droneA = current_pos - pA0
        projA = torch.sum(to_droneA * segA_dir, dim=1, keepdim=True)
        projA = torch.clamp(projA, min=0.0, max=None)
        projA = torch.minimum(projA, segA_len)
        closestA = pA0 + projA * segA_dir
        errorA = torch.norm(current_pos - closestA, dim=1)  # [num_envs]
        
        # --- Segment B: prev → closest ---
        pB0 = self.current_waypoints[env_idx, prev_idx]
        pB1 = pA0  # closest waypoint
        segB = pB1 - pB0
        segB_len = torch.norm(segB, dim=1, keepdim=True)
        segB_dir = segB / (segB_len + 1e-8)
        to_droneB = current_pos - pB0
        projB = torch.sum(to_droneB * segB_dir, dim=1, keepdim=True)
        projB = torch.clamp(projB, min=0.0, max=None)
        projB = torch.minimum(projB, segB_len)
        closestB = pB0 + projB * segB_dir
        errorB = torch.norm(current_pos - closestB, dim=1)  # [num_envs]
        
        # Pick the segment with smaller contour error
        use_A = errorA <= errorB  # [num_envs]
        use_A_3d = use_A.unsqueeze(1)  # [num_envs, 1]
        
        closest_point = torch.where(use_A_3d, closestA, closestB)
        contour_error = torch.where(use_A, errorA, errorB)
        tangent = torch.where(use_A_3d, segA_dir, segB_dir)
        
        # Compute progress: arc-length at segment start + projection along segment
        arc_at_closest = self.cumulative_arc_length[env_idx, closest_idx]
        arc_at_prev = self.cumulative_arc_length[env_idx, prev_idx]
        
        # Segment A: progress = arc_at_closest + projA
        progressA = arc_at_closest + projA.squeeze(-1)
        # Segment B: progress = arc_at_prev + projB
        progressB = arc_at_prev + projB.squeeze(-1)
        
        progress = torch.where(use_A, progressA, progressB)
        # Handle wraparound using per-environment total arc lengths
        progress = progress % self.total_arc_length
        self.progress = progress

        # print(f"[TRACE] update_progress END: contour_error mean={contour_error.mean().item():.3f}")
        return closest_point, contour_error, tangent

    def get_lookahead_points(self, current_pos: torch.Tensor, distances: tuple[float, ...]) -> torch.Tensor:
        """Get look-ahead points at specified distances along trajectory.

        Uses batched interpolation for efficiency - all lookahead distances
        are computed in a single vectorized operation.

        Args:
            current_pos: [num_envs, 3] current drone positions

        Returns:
            lookahead_points: [num_envs, num_lookahead, 3] future trajectory points
        """
        assert self.total_arc_length is not None, "Trajectory not initialized."
        
        # First update progress
        self.update_progress(current_pos)

        # Convert lookahead distances to tensor and compute all target arc-lengths at once
        # [num_lookahead]
        distances_t = torch.tensor(distances, device=self.device, dtype=torch.float32)
        
        # [num_envs, num_lookahead] - broadcast progress [num_envs, 1] + distances [num_lookahead]
        # Use per-environment total arc lengths for wraparound
        target_arcs = self.progress.unsqueeze(1) + distances_t.unsqueeze(0)  # [num_envs, num_lookahead]
        target_arcs = torch.remainder(target_arcs, self.total_arc_length.unsqueeze(1))  # Wrap using per-env lengths
        
        # Batch interpolate all positions at once
        lookahead_points = self._interpolate_at_arc_length_batch(target_arcs)

        return lookahead_points

    def _interpolate_at_arc_length_batch(self, target_arcs: torch.Tensor) -> torch.Tensor:
        """Interpolate positions at multiple arc-lengths efficiently using vectorized operations.
        
        Uses per-environment waypoints for correct positioning.
        
        Args:
            target_arcs: [num_envs, num_distances] target arc-length positions
            
        Returns:
            positions: [num_envs, num_distances, 3] interpolated positions
        """
        assert self.current_waypoints is not None, "Trajectory not initialized."
        assert self.cumulative_arc_length is not None, "Arc lengths not computed."
        
        num_envs, num_distances = target_arcs.shape
        
        # For each target arc, find waypoint segment containing it (per-environment)
        # idx_after: [num_envs, num_distances]
        # Vectorized searchsorted: cumulative_arc_length is [num_envs, num_waypoints], target_arcs is [num_envs, num_distances]
        idx_after = torch.searchsorted(self.cumulative_arc_length.contiguous(), target_arcs.contiguous())
        idx_after = torch.clamp(idx_after, 1, self.num_waypoints - 1)
        idx_before = idx_after - 1
        
        # Get arc-lengths for interpolation (per-environment now)
        # [num_envs, num_distances]
        batch_env_indices = torch.arange(num_envs, device=self.device).unsqueeze(1).expand(-1, num_distances)
        arc0 = self.cumulative_arc_length[batch_env_indices, idx_before]
        arc1 = self.cumulative_arc_length[batch_env_indices, idx_after]
        
        # Linear interpolation parameter
        segment_length = arc1 - arc0
        t = (target_arcs - arc0) / (segment_length + 1e-8)
        t = torch.clamp(t, 0, 1)
        
        # Get waypoints from per-env current_waypoints using advanced indexing
        # current_waypoints: [num_envs, num_waypoints, 3]
        # idx_before/idx_after: [num_envs, num_distances]
        p0 = self.current_waypoints[batch_env_indices, idx_before]  # [num_envs, num_distances, 3]
        p1 = self.current_waypoints[batch_env_indices, idx_after]   # [num_envs, num_distances, 3]
        
        # Interpolate positions
        positions = p0 + t.unsqueeze(-1) * (p1 - p0)  # [num_envs, num_distances, 3]
        
        return positions

    def _interpolate_at_arc_length(self, target_arc: torch.Tensor) -> torch.Tensor:
        """Interpolate position at given arc-length using linear interpolation between waypoints.
        
        Uses per-environment waypoints for correct positioning.
        
        Args:
            target_arc: [num_envs] target arc-length positions
            
        Returns:
            positions: [num_envs, 3] interpolated positions
        """
        assert self.current_waypoints is not None, "Trajectory not initialized."
        assert self.cumulative_arc_length is not None, "Arc lengths not computed."
        
        num_envs = target_arc.shape[0]
        
        # Find waypoint segment containing each target arc-length (per-environment)
        # cumulative_arc_length: [num_envs, num_waypoints], target_arc: [num_envs]
        idx_after = torch.searchsorted(self.cumulative_arc_length.contiguous(), target_arc.unsqueeze(1).contiguous())
        idx_after = idx_after.squeeze(1)  # [num_envs]
        idx_after = torch.clamp(idx_after, 1, self.num_waypoints - 1)
        idx_before = idx_after - 1
        
        # Get arc-lengths for interpolation (per-environment)
        env_indices = torch.arange(num_envs, device=self.device)
        arc0 = self.cumulative_arc_length[env_indices, idx_before]  # [num_envs]
        arc1 = self.cumulative_arc_length[env_indices, idx_after]  # [num_envs]
        
        # Linear interpolation parameter
        segment_length = arc1 - arc0
        t = (target_arc - arc0) / (segment_length + 1e-8)  # [num_envs]
        t = torch.clamp(t, 0, 1)  # Safety clamp
        
        # Get waypoints from per-env current_waypoints
        # current_waypoints: [num_envs, num_waypoints, 3]
        p0 = self.current_waypoints[env_indices, idx_before]  # [num_envs, 3]
        p1 = self.current_waypoints[env_indices, idx_after]   # [num_envs, 3]
        
        # Interpolate position
        position = p0 + t.unsqueeze(-1) * (p1 - p0)  # [num_envs, 3]
        
        return position

    def get_progress_velocity(self, current_vel: torch.Tensor) -> torch.Tensor:
        """Compute velocity component along trajectory direction.

        Args:
            current_vel: [num_envs, 3] current velocity in world frame

        Returns:
            progress_vel: [num_envs] velocity along trajectory (can be negative)
        """
        assert self.current_tangents is not None, "Trajectory not initialized."
        
        # Use per-env tangents at closest waypoint index
        num_envs = current_vel.shape[0]
        env_indices = torch.arange(num_envs, device=self.device)
        tangent = self.current_tangents[env_indices, self.closest_idx]  # [num_envs, 3]
        progress_vel = torch.sum(current_vel * tangent, dim=1)  # [num_envs]
        return progress_vel

    def reset(self, env_ids: torch.Tensor):
        """Reset progress tracking for specified environments.

        Args:
            env_ids: Environment indices to reset
        """
        assert self.total_arc_length is not None, "Trajectory not initialized."
        assert self.cumulative_arc_length is not None, "Arc lengths not computed."
        
        # 1) Select random trajectory from environment's pre-computed library FIRST
        if len(self.env_trajectories) > 0:
            for env_id in env_ids:
                env_idx = int(env_id.item())
                traj_idx = int(torch.randint(0, len(self.env_trajectories[env_idx]), (1,), device=self.device).item())
                self.env_trajectory_ids[env_id] = traj_idx
                
                waypoints, tangents = self.env_trajectories[env_idx][traj_idx]
                assert self.current_waypoints is not None and self.current_tangents is not None
                self.current_waypoints[env_idx, :waypoints.shape[0]] = waypoints
                self.current_tangents[env_idx, :tangents.shape[0]] = tangents
            
            # 2) Recompute arc lengths for the NEW trajectories
            self._recompute_arc_lengths()
        
        # 3) Now set progress using the CORRECT arc lengths
        rand_frac = torch.rand(len(env_ids), device=self.device)
        self.progress[env_ids] = rand_frac * self.total_arc_length[env_ids]
        
        # 4) Find corresponding closest indices (vectorized)
        # cumulative_arc_length: [num_envs, num_waypoints], progress: [num_envs]
        arc_diffs = (self.cumulative_arc_length[env_ids] - self.progress[env_ids].unsqueeze(1)).abs()
        self.closest_idx[env_ids] = torch.argmin(arc_diffs, dim=1)

    def reset_with_difficulty(self, env_ids: torch.Tensor, difficulties: torch.Tensor) -> None:
        """Reset trajectory state for specified environments with curriculum difficulty.

        Args:
            env_ids: Indices of environments to reset.
            difficulties: [len(env_ids)] Difficulty level [0, 1] for each environment.
        """
        assert self.total_arc_length is not None, "Trajectory not initialized."
        assert self.cumulative_arc_length is not None, "Arc lengths not computed."
        
        # 1) Select trajectory based on difficulty from environment's pre-computed library FIRST
        if len(self.env_trajectories) > 0:
            for i, env_id in enumerate(env_ids):
                env_idx = int(env_id.item())
                difficulty = float(difficulties[i].item())
                
                num_trajs = len(self.env_trajectories[env_idx])
                traj_idx = int(difficulty * (num_trajs - 1))
                traj_idx = max(0, min(traj_idx, num_trajs - 1))
                
                self.env_trajectory_ids[env_id] = traj_idx
                
                waypoints, tangents = self.env_trajectories[env_idx][traj_idx]
                assert self.current_waypoints is not None and self.current_tangents is not None
                self.current_waypoints[env_idx, :waypoints.shape[0]] = waypoints
                self.current_tangents[env_idx, :tangents.shape[0]] = tangents
            
            # 2) Recompute arc lengths for the NEW trajectories
            self._recompute_arc_lengths()
        
        # 3) Now set progress using the CORRECT arc lengths
        rand_frac = torch.rand(len(env_ids), device=self.device)
        self.progress[env_ids] = rand_frac * self.total_arc_length[env_ids]
        
        # 4) Find corresponding closest indices (vectorized)
        arc_diffs = (self.cumulative_arc_length[env_ids] - self.progress[env_ids].unsqueeze(1)).abs()
        self.closest_idx[env_ids] = torch.argmin(arc_diffs, dim=1)

    def _recompute_arc_lengths(self) -> None:
        """Recompute arc length parameters for current trajectories.
        
        Computes per-environment arc lengths since each env can have different trajectory shapes.
        """
        if self.current_waypoints is None or self.num_waypoints == 0:
            return
        
        # Compute per-environment arc lengths [num_envs, num_waypoints]
        # Segment lengths: distance between consecutive waypoints [num_envs, num_waypoints-1]
        segment_lengths = torch.norm(self.current_waypoints[:, 1:] - self.current_waypoints[:, :-1], dim=2)
        # Last segment: back to start [num_envs, 1]
        last_segment = torch.norm(self.current_waypoints[:, 0] - self.current_waypoints[:, -1], dim=1, keepdim=True)
        # Concatenate [num_envs, num_waypoints]
        self.segment_lengths = torch.cat([segment_lengths, last_segment], dim=1)
        
        # Cumulative arc length [num_envs, num_waypoints]
        self.cumulative_arc_length = torch.zeros(self.num_envs, self.num_waypoints, device=self.device)
        self.cumulative_arc_length[:, 1:] = torch.cumsum(self.segment_lengths[:, :-1], dim=1)
        # Total arc length per environment [num_envs]
        self.total_arc_length = self.cumulative_arc_length[:, -1] + self.segment_lengths[:, -1]

    def _regenerate_gates(self) -> None:
        """Regenerate gates for all environments based on their trajectories."""
        if not self.gates_enabled or self.current_waypoints is None:
            return
        
        # Generate gates per environment
        all_gate_positions = []
        all_gate_orientations = []
        all_gate_progress = []
        num_gates_list = []
        
        for env_id in range(self.num_envs):
            gate_pos, gate_orient, gate_prog, n_gates = self._generate_gates_for_env(env_id)
            all_gate_positions.append(gate_pos)
            all_gate_orientations.append(gate_orient)
            all_gate_progress.append(gate_prog)
            num_gates_list.append(n_gates)
        
        # Find max gates for padding
        max_gates = max(num_gates_list)
        
        # Pad and stack [num_envs, max_gates, 3/4]
        padded_positions = []
        padded_orientations = []
        padded_progress = []
        
        for gate_pos, gate_orient, gate_prog, n_gates in zip(all_gate_positions, all_gate_orientations, all_gate_progress, num_gates_list):
            if n_gates < max_gates:
                # Pad with zeros
                pad_size = max_gates - n_gates
                gate_pos = torch.cat([gate_pos, torch.zeros(pad_size, 3, device=self.device)], dim=0)
                gate_orient = torch.cat([gate_orient, torch.zeros(pad_size, 4, device=self.device)], dim=0)
                gate_prog = torch.cat([gate_prog, torch.zeros(pad_size, device=self.device)], dim=0)
            padded_positions.append(gate_pos)
            padded_orientations.append(gate_orient)
            padded_progress.append(gate_prog)
        
        self.gate_positions = torch.stack(padded_positions, dim=0)  # [num_envs, max_gates, 3]
        self.gate_orientations = torch.stack(padded_orientations, dim=0)  # [num_envs, max_gates, 4]
        self.gate_progress = torch.stack(padded_progress, dim=0)  # [num_envs, max_gates]
        self.num_gates_per_env = torch.tensor(num_gates_list, device=self.device, dtype=torch.long)  # [num_envs]

    def get_env_waypoints(self, env_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get waypoints for a specific environment.
        
        Args:
            env_id: Environment index.
            
        Returns:
            waypoints: [num_waypoints, 3] trajectory for this environment
            tangents: [num_waypoints, 3] tangent vectors
        """
        if self.current_waypoints is not None and self.current_tangents is not None:
            return self.current_waypoints[env_id], self.current_tangents[env_id]
        else:
            # Fallback - should not happen after proper initialization
            return torch.zeros(50, 3, device=self.device), torch.zeros(50, 3, device=self.device)

    def get_starting_pose(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get starting position and orientation for reset environments.

        Args:
            env_ids: Environment indices to get starting poses for

        Returns:
            positions: [len(env_ids), 3] starting positions in world coordinates
            orientations: [len(env_ids), 4] starting orientations (quaternions) facing along trajectory
        """
        assert self.current_waypoints is not None, "Trajectory not initialized."
        assert self.current_tangents is not None, "Trajectory not initialized."
        
        # Get waypoints at current progress from per-env waypoints (already offset to env origins)
        closest_idx = self.closest_idx[env_ids]
        positions = self.current_waypoints[env_ids, closest_idx].clone()
        
        # Add small random offset for variability (XY only, keep Z close to trajectory)
        position_max_perturbation = 0.1  # meters
        perturbation = torch.randn(len(env_ids), 3, device=self.device) * position_max_perturbation
        perturbation[:, 2] *= 0.3  # Less vertical perturbation
        positions += perturbation

        # Compute orientation to face along trajectory (tangents are direction vectors, same regardless of offset)
        tangent = self.current_tangents[env_ids, closest_idx]  # [len(env_ids), 3]

        # Add small random orientation offset for variability
        tangent_max_perturbation = 0.1 # normalized vector perturbation
        tangent += torch.rand_like(tangent) * tangent_max_perturbation - (tangent_max_perturbation / 2)
        tangent = tangent / (torch.norm(tangent, dim=1, keepdim=True) + 1e-8)
        
        # Create quaternion from forward direction (tangent)
        # For simplicity, assume drone's x-axis should align with tangent
        # and z-axis points up (standard racing drone orientation)
        orientations = self._compute_orientation_from_direction(tangent)

        return positions, orientations

    def _compute_orientation_from_direction(self, forward: torch.Tensor) -> torch.Tensor:
        """Compute quaternion orientation from forward direction vector.

        The body frame convention is: x=forward (along trajectory), y=left, z=up.
        The rotation matrix R maps body axes to world: R @ [1,0,0] = forward_world, etc.
        So R columns are [forward, left, up] in world coordinates.

        Args:
            forward: [N, 3] forward direction vectors (will be normalized)

        Returns:
            quaternions: [N, 4] orientation quaternions (w, x, y, z)
        """
        from isaaclab.utils.math import quat_from_matrix
        
        N = forward.shape[0]
        
        # Normalize forward vector
        forward = forward / (torch.norm(forward, dim=1, keepdim=True) + 1e-8)
        
        # World up vector
        world_up = torch.zeros_like(forward)
        world_up[:, 2] = 1.0
        
        # Handle edge case: forward nearly parallel to world_up
        # Use world_x as fallback up direction
        dot_with_up = torch.abs(torch.sum(forward * world_up, dim=1))
        nearly_vertical = dot_with_up > 0.99
        if nearly_vertical.any():
            world_up[nearly_vertical] = torch.tensor([1.0, 0.0, 0.0], device=forward.device)
        
        # Left vector (body Y): cross(world_up, forward) for right-hand rule
        left = torch.cross(world_up, forward, dim=1)
        left = left / (torch.norm(left, dim=1, keepdim=True) + 1e-8)
        
        # Recompute up to ensure orthogonality: up = cross(forward, left)
        up = torch.cross(forward, left, dim=1)
        up = up / (torch.norm(up, dim=1, keepdim=True) + 1e-8)
        
        # Build rotation matrix [N, 3, 3]
        # Columns are body axes expressed in world frame: [forward(x), left(y), up(z)]
        rot_mat = torch.stack([forward, left, up], dim=2)
        
        # Convert rotation matrix to quaternion using Isaac Lab's robust implementation
        quat = quat_from_matrix(rot_mat)
        
        return quat

    # Removed: use isaaclab.utils.math.quat_from_matrix instead

    def _generate_gates_for_env(self, env_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Generate virtual gate positions and orientations for a specific environment.
        
        Gates are placed at regular intervals along the arc-length of the trajectory.
        Each gate is oriented perpendicular to the trajectory tangent.
        
        Args:
            env_id: Environment index
            
        Returns:
            gate_positions: [num_gates, 3] positions of gate centers
            gate_orientations: [num_gates, 4] orientations as quaternions (w, x, y, z)
            gate_progress: [num_gates] arc-length progress for each gate
            num_gates: number of gates generated
        """
        assert self.total_arc_length is not None, "Trajectory not initialized."
        assert self.current_waypoints is not None, "Trajectory not initialized."
        assert self.current_tangents is not None, "Trajectory not initialized."
        assert self.cumulative_arc_length is not None, "Arc lengths not computed."
        
        # Get this environment's arc length
        env_arc_length = self.total_arc_length[env_id].item()
        num_gates = max(1, int(env_arc_length / self.cfg.gate_spacing))
        
        gate_positions = torch.zeros(num_gates, 3, device=self.device)
        gate_orientations = torch.zeros(num_gates, 4, device=self.device)
        gate_progress = torch.zeros(num_gates, device=self.device)
        
        # Get this environment's waypoints
        env_waypoints = self.current_waypoints[env_id]  # [num_waypoints, 3]
        env_tangents = self.current_tangents[env_id]    # [num_waypoints, 3]
        env_cumulative = self.cumulative_arc_length[env_id]  # [num_waypoints]
        
        for i in range(num_gates):
            # Target arc-length for this gate
            target_arc_length = i * self.cfg.gate_spacing
            gate_progress[i] = target_arc_length
            
            # Find waypoint closest to this arc-length
            idx = torch.searchsorted(env_cumulative, torch.tensor(target_arc_length, device=self.device))
            idx = torch.clamp(idx, 0, self.num_waypoints - 1).item()
            
            # Interpolate between waypoints if needed
            if idx > 0 and idx < self.num_waypoints:
                # Linear interpolation factor
                prev_arc = env_cumulative[idx - 1]
                next_arc = env_cumulative[idx]
                t = (target_arc_length - prev_arc) / (next_arc - prev_arc + 1e-8)
                
                # Interpolate position
                gate_positions[i] = (1 - t) * env_waypoints[idx - 1] + t * env_waypoints[idx]
                
                # Use tangent at the closer waypoint
                tangent = env_tangents[idx] if t > 0.5 else env_tangents[idx - 1]
            else:
                gate_positions[i] = env_waypoints[idx]
                tangent = env_tangents[idx]
            
            # Create perpendicular frame for gate
            # Gate frame: body-x = right (horizontal), body-y = tangent (through), body-z = up
            z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            # Right = tangent cross up (for right-hand rule, gate opening horizontal)
            right = torch.cross(tangent, z_axis)
            right = right / (torch.norm(right) + 1e-8)
            # Recompute up for orthogonality
            up = torch.cross(right, tangent)
            up = up / (torch.norm(up) + 1e-8)
            
            # Rotation matrix columns: [body-x=right, body-y=tangent, body-z=up]
            from isaaclab.utils.math import quat_from_matrix
            rot_mat = torch.stack([right, tangent, up], dim=1)  # [3, 3]
            quat = quat_from_matrix(rot_mat.unsqueeze(0))[0]
            gate_orientations[i] = quat
        
        return gate_positions, gate_orientations, gate_progress, num_gates

    def _generate_gates(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate virtual gate positions and orientations along the trajectory.
        
        Gates are placed at regular intervals along the arc-length of the trajectory.
        Each gate is oriented perpendicular to the trajectory tangent.
        Uses first environment's trajectory for gate visualization.
        
        Returns:
            gate_positions: [num_gates, 3] positions of gate centers
            gate_orientations: [num_gates, 4] orientations as quaternions (w, x, y, z)
        """
        
        assert self.total_arc_length is not None, "Trajectory not initialized."
        assert self.current_waypoints is not None, "Trajectory not initialized."
        assert self.current_tangents is not None, "Trajectory not initialized."
        assert self.cumulative_arc_length is not None, "Arc lengths not computed."
        
        # Use first environment's trajectory for gate visualization (gates are just visual aids)
        env0_arc_length = self.total_arc_length[0].item()
        num_gates = int(env0_arc_length / self.cfg.gate_spacing)
        
        gate_positions = torch.zeros(num_gates, 3, device=self.device)
        gate_orientations = torch.zeros(num_gates, 4, device=self.device)
        gate_progress = torch.zeros(num_gates, device=self.device)
        
        # Use first environment's waypoints for gate visualization
        env_waypoints = self.current_waypoints[0]  # [num_waypoints, 3]
        env_tangents = self.current_tangents[0]    # [num_waypoints, 3]
        env_cumulative = self.cumulative_arc_length[0]  # [num_waypoints]
        
        for i in range(num_gates):
            # Target arc-length for this gate
            target_arc_length = i * self.cfg.gate_spacing
            gate_progress[i] = target_arc_length
            
            # Find waypoint closest to this arc-length
            idx = torch.searchsorted(env_cumulative, torch.tensor(target_arc_length, device=self.device))
            idx = torch.clamp(idx, 0, self.num_waypoints - 1).item()
            
            # Interpolate between waypoints if needed
            if idx > 0 and idx < self.num_waypoints:
                # Linear interpolation factor
                prev_arc = env_cumulative[idx - 1]
                next_arc = env_cumulative[idx]
                t = (target_arc_length - prev_arc) / (next_arc - prev_arc + 1e-8)
                
                # Interpolate position
                gate_positions[i] = (1 - t) * env_waypoints[idx - 1] + t * env_waypoints[idx]
                
                # Use tangent at the closer waypoint
                tangent = env_tangents[idx] if t > 0.5 else env_tangents[idx - 1]
            else:
                gate_positions[i] = env_waypoints[idx]
                tangent = env_tangents[idx]
            
            # Compute gate orientation (perpendicular to trajectory)
            # Gate normal should point along the trajectory (for passing through)
            gate_orientations[i] = self._compute_orientation_from_direction(tangent.unsqueeze(0)).squeeze(0)
        
        return gate_positions, gate_orientations, gate_progress, num_gates

    def get_gate_info(self, env_id: int = 0) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Get gate positions and orientations for a specific environment.
        
        Args:
            env_id: Environment index (default 0 for visualization)
        
        Returns:
            gate_positions: [num_gates, 3] positions of gate centers for this env
            gate_orientations: [num_gates, 4] orientations as quaternions (w, x, y, z) for this env
            num_gates: number of gates for this env
        """
        if self.gate_positions is None or self.num_gates_per_env is None:
            return torch.empty(0, 3, device=self.device), torch.empty(0, 4, device=self.device), 0
        
        n_gates = self.num_gates_per_env[env_id].item()
        return self.gate_positions[env_id, :n_gates], self.gate_orientations[env_id, :n_gates], n_gates

    def get_all_gates_flattened(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Get all gate positions and orientations from all environments, flattened.
        
        Returns:
            gate_positions: [total_gates, 3] positions of all gates from all envs
            gate_orientations: [total_gates, 4] orientations as quaternions (w, x, y, z)
            total_gates: total number of gates across all environments
        """
        if self.gate_positions is None or self.num_gates_per_env is None:
            return torch.empty(0, 3, device=self.device), torch.empty(0, 4, device=self.device), 0
        
        # Collect all valid gates from all environments
        all_positions = []
        all_orientations = []
        
        for env_id in range(self.num_envs):
            n_gates = self.num_gates_per_env[env_id].item()
            if n_gates > 0:
                all_positions.append(self.gate_positions[env_id, :n_gates])
                all_orientations.append(self.gate_orientations[env_id, :n_gates])
        
        if len(all_positions) == 0:
            return torch.empty(0, 3, device=self.device), torch.empty(0, 4, device=self.device), 0
        
        positions = torch.cat(all_positions, dim=0)
        orientations = torch.cat(all_orientations, dim=0)
        
        return positions, orientations, positions.shape[0]

    def check_gate_collision(self, current_pos: torch.Tensor, contour_error: torch.Tensor) -> torch.Tensor:
        """Check if drones have collided with gates.
        
        A collision occurs when the drone is near a gate (based on progress along trajectory)
        and its contour error exceeds the gate opening size (drone passed outside the gate).
        
        Args:
            current_pos: [num_envs, 3] current drone positions
            contour_error: [num_envs] perpendicular distance from trajectory
            
        Returns:
            collision_mask: [num_envs] boolean mask indicating gate collisions
        """
        # print(f"[TRACE] check_gate_collision START")
        if not self.gates_enabled or self.gate_progress is None or self.total_arc_length is None or self.num_gates_per_env is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        max_gates = self.gate_progress.shape[1]
        
        # Create mask for valid gates per env: [num_envs, max_gates]
        gate_range = torch.arange(max_gates, device=self.device).unsqueeze(0)  # [1, max_gates]
        valid_mask = gate_range < self.num_gates_per_env.unsqueeze(1)  # [num_envs, max_gates]
        
        # Progress diffs: [num_envs, max_gates]
        progress_diffs = (self.progress.unsqueeze(1) - self.gate_progress).abs()
        
        # Handle wraparound
        progress_diffs_wrap = torch.minimum(
            progress_diffs,
            self.total_arc_length.unsqueeze(1) - progress_diffs
        )
        
        # Set invalid gates to large distance so they're never "nearest"
        progress_diffs_wrap = torch.where(valid_mask, progress_diffs_wrap, torch.tensor(1e10, device=self.device))
        
        # Get distance to nearest gate along trajectory: [num_envs]
        nearest_gate_progress_dist, _ = progress_diffs_wrap.min(dim=1)
        
        # Check collision
        buffer = 0.15
        near_gate = nearest_gate_progress_dist < self.cfg.gate_collision_threshold
        outside_gate = contour_error > (self.cfg.gate_size / 2.0 - buffer)
        
        # Only flag envs that actually have gates
        has_gates = self.num_gates_per_env > 0
        collision_mask = near_gate & outside_gate & has_gates
        
        # print(f"[TRACE] check_gate_collision END: num_collisions={collision_mask.sum().item()}")
        return collision_mask

    def get_gate_penalty_multiplier(self, current_pos: torch.Tensor) -> torch.Tensor:
        """Get penalty multiplier for contour error based on proximity to gates.
        
        Contour error penalties are amplified when near gates (based on progress along trajectory)
        to encourage precise passage through gate openings.
        
        Args:
            current_pos: [num_envs, 3] current drone positions
            
        Returns:
            multiplier: [num_envs] penalty multiplier (1.0 baseline, higher near gates)
        """
        if not self.gates_enabled or self.gate_progress is None or self.total_arc_length is None or self.num_gates_per_env is None:
            return torch.ones(self.num_envs, device=self.device)
        
        max_gates = self.gate_progress.shape[1]
        
        # Create mask for valid gates per env: [num_envs, max_gates]
        gate_range = torch.arange(max_gates, device=self.device).unsqueeze(0)
        valid_mask = gate_range < self.num_gates_per_env.unsqueeze(1)
        
        # Progress diffs: [num_envs, max_gates]
        progress_diffs = (self.progress.unsqueeze(1) - self.gate_progress).abs()
        
        # Handle wraparound
        progress_diffs_wrap = torch.minimum(
            progress_diffs,
            self.total_arc_length.unsqueeze(1) - progress_diffs
        )
        
        # Set invalid gates to large distance
        progress_diffs_wrap = torch.where(valid_mask, progress_diffs_wrap, torch.tensor(1e10, device=self.device))
        
        # Get distance to nearest gate: [num_envs]
        nearest_gate_progress_dist, _ = progress_diffs_wrap.min(dim=1)
        
        # Smooth falloff from gate center
        decay_distance = self.cfg.gate_collision_threshold
        multiplier = 1.0 + (self.cfg.gate_penalty_multiplier - 1.0) * torch.exp(
            -nearest_gate_progress_dist / decay_distance
        )
        
        # Envs with no gates get multiplier 1.0
        has_gates = self.num_gates_per_env > 0
        multiplier = torch.where(has_gates, multiplier, torch.ones_like(multiplier))
        
        return multiplier
