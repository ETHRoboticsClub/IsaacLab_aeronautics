# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory generation and tracking utilities for quadcopter racing."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Optional

from .trajectory_generator import TrajectoryLibrary, TrajectoryGeneratorCfg


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory generation."""

    trajectory_type: str = "library"  # Options: "circle", "oval", "figure8", "library"
    radius: float = 2.0  # Base radius for circular trajectories (when not using library)
    height: float = 1.5  # Flight height (when not using library)
    num_waypoints: int = 100  # Number of discrete waypoints
    desired_speed: float = 2.0  # Target speed along trajectory (m/s)
    
    # Look-ahead distances for observations (in meters along arc-length)
    lookahead_distances: tuple[float, ...] = (0.3, 0.6, 1.0, 1.3, 1.6, 2.0)
    
    # Velocity limit configuration
    velocity_limit: float = 3.0  # Maximum allowed speed (m/s)
    randomize_velocity_limit: bool = True  # Randomize limit per episode
    velocity_limit_range: tuple[float, float] = (1.5, 4.0)  # Random range for velocity limit
    
    # Virtual gate configuration
    enable_gates: bool = True  # Enable virtual gates along trajectory
    gate_spacing: float = 10.0  # Distance between gates (meters along arc-length)
    gate_size: float = 2.0  # Size of gate opening (meters)
    gate_penalty_multiplier: float = 5.0  # Contour error penalty multiplier near gates
    gate_collision_threshold: float = 1.0  # Distance threshold for gate collision (meters)
    
    # Trajectory library configuration
    use_trajectory_library: bool = True  # Use precomputed racing tracks
    track_seed: Optional[int] = 42  # Seed for trajectory generation
    trajectory_generator_cfg: Optional[TrajectoryGeneratorCfg] = None  # Config for trajectory generation


class RacingTrajectory:
    """Manages racing trajectory with waypoints and progress tracking."""

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

        # Initialize trajectory library if enabled
        self.trajectory_library: Optional[TrajectoryLibrary] = None
        if cfg.use_trajectory_library:
            traj_gen_cfg = cfg.trajectory_generator_cfg if cfg.trajectory_generator_cfg is not None else TrajectoryGeneratorCfg()
            print(f"Generating trajectory library system with {traj_gen_cfg.num_libraries} libraries...")
            self.trajectory_library = TrajectoryLibrary(
                num_waypoints=cfg.num_waypoints,
                device=device,
                seed=cfg.track_seed,
                cfg=traj_gen_cfg,
            )
            print("Trajectory library system generation complete!")

        # Per-environment trajectory assignment (library_idx, traj_idx within library)
        self.env_library_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.env_trajectory_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        
        # Assign libraries to environments
        if cfg.use_trajectory_library and self.trajectory_library is not None:
            for env_id in range(num_envs):
                self.env_library_ids[env_id] = self.trajectory_library.get_library_for_env(env_id)
        
        # Generate initial trajectories - use first trajectory from first library as default
        # (individual env trajectories are handled separately)
        self.waypoints, self.tangents = self._generate_trajectory()
        self.num_waypoints = self.waypoints.shape[0]

        # Compute arc-length distances between consecutive waypoints
        self.segment_lengths = torch.norm(self.waypoints[1:] - self.waypoints[:-1], dim=1)
        # Add last segment to close the loop
        last_segment = torch.norm(self.waypoints[0] - self.waypoints[-1])
        self.segment_lengths = torch.cat([self.segment_lengths, last_segment.unsqueeze(0)])
        
        # Cumulative arc-length at each waypoint
        self.cumulative_arc_length = torch.zeros(self.num_waypoints, device=self.device)
        self.cumulative_arc_length[1:] = torch.cumsum(self.segment_lengths[:-1], dim=0)
        self.total_arc_length = self.cumulative_arc_length[-1] + self.segment_lengths[-1]

        # Initialize virtual gates if enabled
        self.gates_enabled = cfg.enable_gates
        if self.gates_enabled:
            self.gate_positions, self.gate_orientations = self._generate_gates()
            self.num_gates = len(self.gate_positions)
            print(f"Generated {self.num_gates} virtual gates along trajectory (spacing: {cfg.gate_spacing}m)")
        else:
            self.gate_positions = torch.empty(0, 3, device=device)
            self.gate_orientations = torch.empty(0, 4, device=device)
            self.num_gates = 0

        # Per-environment state: current progress (arc-length) and closest waypoint index
        self.progress = torch.zeros(num_envs, device=device)  # Current arc-length progress
        self.closest_idx = torch.zeros(num_envs, dtype=torch.long, device=device)  # Closest waypoint index
        
        # Per-environment velocity limit
        self.velocity_limit = torch.ones(num_envs, device=device) * cfg.velocity_limit

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
        n = self.cfg.num_waypoints
        theta = torch.linspace(0, 2 * torch.pi, n, device=self.device)

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
        # For library mode with per-env trajectories, we need to handle each env separately
        # For efficiency, we'll batch process using the primary trajectory but store per-env data
        
        # Find closest waypoint for each environment
        # [num_envs, num_waypoints, 3]
        diffs = current_pos.unsqueeze(1) - self.waypoints.unsqueeze(0)
        distances = torch.norm(diffs, dim=2)  # [num_envs, num_waypoints]
        
        closest_idx = torch.argmin(distances, dim=1)  # [num_envs]
        self.closest_idx = closest_idx

        # Get closest waypoint positions and tangents
        closest_point = self.waypoints[closest_idx]  # [num_envs, 3]
        tangent = self.tangents[closest_idx]  # [num_envs, 3]

        # Compute contour error (perpendicular distance)
        to_drone = current_pos - closest_point  # [num_envs, 3]
        
        # Project onto tangent to get along-track error
        along_track = torch.sum(to_drone * tangent, dim=1, keepdim=True)  # [num_envs, 1]
        
        # Contour error is the perpendicular component
        perpendicular = to_drone - along_track * tangent
        contour_error = torch.norm(perpendicular, dim=1)  # [num_envs]

        # Update progress: arc-length at closest waypoint + along-track offset
        progress = self.cumulative_arc_length[closest_idx] + along_track.squeeze(-1)
        # Handle wraparound
        progress = progress % self.total_arc_length
        self.progress = progress

        return closest_point, contour_error, tangent

    def get_lookahead_points(self, current_pos: torch.Tensor) -> torch.Tensor:
        """Get look-ahead points at specified distances along trajectory.

        Args:
            current_pos: [num_envs, 3] current drone positions

        Returns:
            lookahead_points: [num_envs, num_lookahead, 3] future trajectory points
        """
        # First update progress
        self.update_progress(current_pos)

        lookahead_points = []
        
        for dist in self.cfg.lookahead_distances:
            # Target arc-length for this look-ahead
            target_arc = (self.progress + dist) % self.total_arc_length  # [num_envs]
            
            # Find waypoint indices for these arc-lengths
            # Broadcasting: [num_envs, 1] vs [num_waypoints]
            arc_diffs = (self.cumulative_arc_length.unsqueeze(0) - target_arc.unsqueeze(1)).abs()
            lookahead_idx = torch.argmin(arc_diffs, dim=1)  # [num_envs]
            
            # Get corresponding waypoints
            points = self.waypoints[lookahead_idx]  # [num_envs, 3]
            lookahead_points.append(points)

        # Stack: [num_envs, num_lookahead, 3]
        return torch.stack(lookahead_points, dim=1)

    def get_progress_velocity(self, current_vel: torch.Tensor) -> torch.Tensor:
        """Compute velocity component along trajectory direction.

        Args:
            current_vel: [num_envs, 3] current velocity in world frame

        Returns:
            progress_vel: [num_envs] velocity along trajectory (can be negative)
        """
        tangent = self.tangents[self.closest_idx]  # [num_envs, 3]
        progress_vel = torch.sum(current_vel * tangent, dim=1)  # [num_envs]
        return progress_vel

    def reset(self, env_ids: torch.Tensor):
        """Reset progress tracking for specified environments.

        Args:
            env_ids: Environment indices to reset
        """
        # Start at random positions along the trajectory
        self.progress[env_ids] = torch.rand(len(env_ids), device=self.device) * self.total_arc_length
        
        # Find corresponding closest indices
        arc_diffs = (self.cumulative_arc_length.unsqueeze(0) - self.progress[env_ids].unsqueeze(1)).abs()
        self.closest_idx[env_ids] = torch.argmin(arc_diffs, dim=1)
        
        # Randomize velocity limit if enabled
        if self.cfg.randomize_velocity_limit:
            low, high = self.cfg.velocity_limit_range
            self.velocity_limit[env_ids] = torch.rand(len(env_ids), device=self.device) * (high - low) + low
        else:
            self.velocity_limit[env_ids] = self.cfg.velocity_limit
        
        # Select random trajectory from environment's assigned library
        if self.cfg.use_trajectory_library and self.trajectory_library is not None:
            # Each environment gets a random trajectory from its assigned library
            for env_id in env_ids:
                env_idx = env_id.item()
                _, _, traj_idx = self.trajectory_library.get_random_trajectory(env_idx)
                self.env_trajectory_ids[env_id] = traj_idx
            
            # Update shared trajectory for visualization (use first reset env)
            if len(env_ids) > 0:
                first_env = env_ids[0].item()
                waypoints, tangents = self.trajectory_library.get_trajectory(
                    first_env, self.env_trajectory_ids[env_ids[0]].item()
                )
                self.waypoints = waypoints
                self.tangents = tangents
                self._recompute_arc_lengths()

    def reset_with_difficulty(self, env_ids: torch.Tensor, difficulties: torch.Tensor) -> None:
        """Reset trajectory state for specified environments with curriculum difficulty.

        Args:
            env_ids: Indices of environments to reset.
            difficulties: [len(env_ids)] Difficulty level [0, 1] for each environment.
        """
        # Start at random positions along the trajectory
        self.progress[env_ids] = torch.rand(len(env_ids), device=self.device) * self.total_arc_length
        
        # Find corresponding closest indices
        arc_diffs = (self.cumulative_arc_length.unsqueeze(0) - self.progress[env_ids].unsqueeze(1)).abs()
        self.closest_idx[env_ids] = torch.argmin(arc_diffs, dim=1)
        
        # Randomize velocity limit if enabled
        if self.cfg.randomize_velocity_limit:
            low, high = self.cfg.velocity_limit_range
            self.velocity_limit[env_ids] = torch.rand(len(env_ids), device=self.device) * (high - low) + low
        else:
            self.velocity_limit[env_ids] = self.cfg.velocity_limit
        
        # Select trajectory based on difficulty from environment's assigned library
        if self.cfg.use_trajectory_library and self.trajectory_library is not None:
            # Each environment gets trajectory matching its difficulty from its library
            for i, env_id in enumerate(env_ids):
                env_idx = env_id.item()
                difficulty = difficulties[i].item()
                _, _, traj_idx = self.trajectory_library.get_trajectory_by_difficulty(env_idx, difficulty)
                self.env_trajectory_ids[env_id] = traj_idx
            
            # Update shared trajectory for visualization (use first reset env)
            if len(env_ids) > 0:
                first_env = env_ids[0].item()
                first_difficulty = difficulties[0].item()
                waypoints, tangents, _ = self.trajectory_library.get_trajectory_by_difficulty(
                    first_env, first_difficulty
                )
                self.waypoints = waypoints
                self.tangents = tangents
                self._recompute_arc_lengths()

    def _recompute_arc_lengths(self) -> None:
        """Recompute arc length parameters after trajectory change."""
        self.segment_lengths = torch.norm(self.waypoints[1:] - self.waypoints[:-1], dim=1)
        last_segment = torch.norm(self.waypoints[0] - self.waypoints[-1])
        self.segment_lengths = torch.cat([self.segment_lengths, last_segment.unsqueeze(0)])
        self.cumulative_arc_length = torch.zeros(self.num_waypoints, device=self.device)
        self.cumulative_arc_length[1:] = torch.cumsum(self.segment_lengths[:-1], dim=0)
        self.total_arc_length = self.cumulative_arc_length[-1] + self.segment_lengths[-1]

    def get_env_waypoints(self, env_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get waypoints for a specific environment.
        
        Args:
            env_id: Environment index.
            
        Returns:
            waypoints: [num_waypoints, 3] trajectory for this environment
            tangents: [num_waypoints, 3] tangent vectors
        """
        if self.cfg.use_trajectory_library and self.trajectory_library is not None:
            traj_id = self.env_trajectory_ids[env_id].item()
            return self.trajectory_library.get_trajectory(env_id, traj_id)
        else:
            return self.waypoints, self.tangents

    def get_starting_pose(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get starting position and orientation for reset environments.

        Args:
            env_ids: Environment indices to get starting poses for

        Returns:
            positions: [len(env_ids), 3] starting positions
            orientations: [len(env_ids), 4] starting orientations (quaternions) facing along trajectory
        """
        # Get waypoints at current progress
        closest_idx = self.closest_idx[env_ids]
        positions = self.waypoints[closest_idx].clone()
        
        # Add small random offset to avoid all drones starting exactly on the line
        offset = torch.randn(len(env_ids), 3, device=self.device) * 0.1
        offset[:, 2] = 0  # No vertical offset
        positions = positions + offset

        # Compute orientation to face along trajectory
        tangent = self.tangents[closest_idx]  # [len(env_ids), 3]
        
        # Create quaternion from forward direction (tangent)
        # For simplicity, assume drone's x-axis should align with tangent
        # and z-axis points up (standard racing drone orientation)
        orientations = self._compute_orientation_from_direction(tangent)

        return positions, orientations

    def _compute_orientation_from_direction(self, forward: torch.Tensor) -> torch.Tensor:
        """Compute quaternion orientation from forward direction vector.

        Args:
            forward: [N, 3] forward direction vectors (will be normalized)

        Returns:
            quaternions: [N, 4] orientation quaternions (w, x, y, z)
        """
        N = forward.shape[0]
        
        # Normalize forward vector
        forward = forward / (torch.norm(forward, dim=1, keepdim=True) + 1e-8)
        
        # World up vector
        world_up = torch.zeros_like(forward)
        world_up[:, 2] = 1.0
        
        # Right vector: cross product of forward and up
        right = torch.cross(forward, world_up, dim=1)
        right = right / (torch.norm(right, dim=1, keepdim=True) + 1e-8)
        
        # Recompute up to ensure orthogonality
        up = torch.cross(right, forward, dim=1)
        
        # Build rotation matrix [N, 3, 3]
        # Convention: x=forward, y=right, z=up
        rot_mat = torch.stack([forward, right, up], dim=2)
        
        # Convert rotation matrix to quaternion
        # Using Shepperd's method for numerical stability
        quat = self._rotation_matrix_to_quaternion(rot_mat)
        
        return quat

    def _rotation_matrix_to_quaternion(self, rot_mat: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrices to quaternions.

        Args:
            rot_mat: [N, 3, 3] rotation matrices

        Returns:
            quaternions: [N, 4] quaternions in (w, x, y, z) format
        """
        # Simple conversion (can be improved for numerical stability)
        batch_size = rot_mat.shape[0]
        quat = torch.zeros(batch_size, 4, device=rot_mat.device)
        
        trace = rot_mat[:, 0, 0] + rot_mat[:, 1, 1] + rot_mat[:, 2, 2]
        
        # When trace is positive
        mask = trace > 0
        s = torch.sqrt(trace[mask] + 1.0) * 2  # s = 4 * qw
        quat[mask, 0] = 0.25 * s  # qw
        quat[mask, 1] = (rot_mat[mask, 2, 1] - rot_mat[mask, 1, 2]) / s
        quat[mask, 2] = (rot_mat[mask, 0, 2] - rot_mat[mask, 2, 0]) / s
        quat[mask, 3] = (rot_mat[mask, 1, 0] - rot_mat[mask, 0, 1]) / s
        
        # When trace is not positive (simplified, can be expanded)
        mask = ~mask
        if mask.any():
            # Use identity quaternion as fallback
            quat[mask, 0] = 1.0
        
        # Normalize
        quat = quat / (torch.norm(quat, dim=1, keepdim=True) + 1e-8)
        
        return quat

    def _generate_gates(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate virtual gate positions and orientations along the trajectory.
        
        Gates are placed at regular intervals along the arc-length of the trajectory.
        Each gate is oriented perpendicular to the trajectory tangent.
        
        Returns:
            gate_positions: [num_gates, 3] positions of gate centers
            gate_orientations: [num_gates, 4] orientations as quaternions (w, x, y, z)
        """
        num_gates = int(self.total_arc_length / self.cfg.gate_spacing)
        gate_positions = torch.zeros(num_gates, 3, device=self.device)
        gate_orientations = torch.zeros(num_gates, 4, device=self.device)
        
        for i in range(num_gates):
            # Target arc-length for this gate
            target_arc_length = i * self.cfg.gate_spacing
            
            # Find waypoint closest to this arc-length
            idx = torch.searchsorted(self.cumulative_arc_length, target_arc_length)
            idx = torch.clamp(idx, 0, self.num_waypoints - 1)
            
            # Interpolate between waypoints if needed
            if idx > 0 and idx < self.num_waypoints:
                # Linear interpolation factor
                prev_arc = self.cumulative_arc_length[idx - 1]
                next_arc = self.cumulative_arc_length[idx]
                t = (target_arc_length - prev_arc) / (next_arc - prev_arc + 1e-8)
                
                # Interpolate position
                gate_positions[i] = (1 - t) * self.waypoints[idx - 1] + t * self.waypoints[idx]
                
                # Use tangent at the closer waypoint
                tangent = self.tangents[idx] if t > 0.5 else self.tangents[idx - 1]
            else:
                gate_positions[i] = self.waypoints[idx]
                tangent = self.tangents[idx]
            
            # Compute gate orientation (perpendicular to trajectory)
            # Gate normal should point along the trajectory (for passing through)
            gate_orientations[i] = self._compute_orientation_from_direction(tangent.unsqueeze(0)).squeeze(0)
        
        return gate_positions, gate_orientations

    def get_gate_info(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Get gate positions and orientations for visualization and collision checking.
        
        Returns:
            gate_positions: [num_gates, 3] positions of gate centers
            gate_orientations: [num_gates, 4] orientations as quaternions (w, x, y, z)
            num_gates: number of gates
        """
        return self.gate_positions, self.gate_orientations, self.num_gates

    def check_gate_collision(self, current_pos: torch.Tensor, contour_error: torch.Tensor) -> torch.Tensor:
        """Check if drones have collided with gates.
        
        A collision occurs when the drone is near a gate and its contour error
        exceeds the gate opening size (drone passed outside the gate).
        
        Args:
            current_pos: [num_envs, 3] current drone positions
            contour_error: [num_envs] perpendicular distance from trajectory
            
        Returns:
            collision_mask: [num_envs] boolean mask indicating gate collisions
        """
        if not self.gates_enabled or self.num_gates == 0:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Find nearest gate for each environment
        # [num_envs, num_gates, 3]
        gate_diffs = current_pos.unsqueeze(1) - self.gate_positions.unsqueeze(0)
        gate_distances = torch.norm(gate_diffs, dim=2)  # [num_envs, num_gates]
        
        # Get distance to nearest gate
        nearest_gate_dist, _ = torch.min(gate_distances, dim=1)  # [num_envs]
        
        # Check collision: near a gate AND contour error exceeds gate size
        near_gate = nearest_gate_dist < self.cfg.gate_collision_threshold
        outside_gate = contour_error > (self.cfg.gate_size / 2.0)
        
        collision_mask = near_gate & outside_gate
        
        return collision_mask

    def get_gate_penalty_multiplier(self, current_pos: torch.Tensor) -> torch.Tensor:
        """Get penalty multiplier for contour error based on proximity to gates.
        
        Contour error penalties are amplified when near gates to encourage
        precise passage through gate openings.
        
        Args:
            current_pos: [num_envs, 3] current drone positions
            
        Returns:
            multiplier: [num_envs] penalty multiplier (1.0 baseline, higher near gates)
        """
        if not self.gates_enabled or self.num_gates == 0:
            return torch.ones(self.num_envs, device=self.device)
        
        # Find distance to nearest gate
        gate_diffs = current_pos.unsqueeze(1) - self.gate_positions.unsqueeze(0)
        gate_distances = torch.norm(gate_diffs, dim=2)  # [num_envs, num_gates]
        nearest_gate_dist, _ = torch.min(gate_distances, dim=1)  # [num_envs]
        
        # Smooth falloff from gate center: multiplier highest at gate, decays with distance
        # Use exponential decay
        decay_distance = self.cfg.gate_collision_threshold
        multiplier = 1.0 + (self.cfg.gate_penalty_multiplier - 1.0) * torch.exp(
            -nearest_gate_dist / decay_distance
        )
        
        return multiplier
