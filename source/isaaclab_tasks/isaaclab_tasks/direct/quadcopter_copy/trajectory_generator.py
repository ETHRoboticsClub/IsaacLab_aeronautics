# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Advanced trajectory generation for racing tracks using splines and procedural generation."""

from __future__ import annotations

import torch
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrajectoryGeneratorCfg:
    """Configuration for trajectory generation parameters."""
    
    # Library parameters
    num_libraries: int = 30  # Number of independent trajectory libraries
    trajectories_per_library: int = 100  # Trajectories per library
    
    # Library type distribution (must sum to 1.0)
    # Each library will specialize in one trajectory type
    library_type_ratios: dict[str, float] | None = None  # If None, uses default
    # Default: {"banked_loop": 0.20, "figure8": 0.20, "cloverleaf": 0.20, 
    #          "racing_circuit": 0.20, "spline": 0.20}
    
    # Size parameters
    radius_range: tuple[float, float] = (1.5, 3.5)  # Min/max radius for trajectories
    height_range: tuple[float, float] = (1.0, 2.0)  # Min/max base height
    height_amplitude_range: tuple[float, float] = (0.2, 0.7)  # Vertical variation
    
    # Complexity parameters
    banking_angle_range: tuple[float, float] = (-0.2, 0.2)  # Banking angle in radians
    num_oscillations_range: tuple[int, int] = (1, 3)  # Vertical waves for loops
    num_lobes_range: tuple[int, int] = (3, 4)  # Lobes for cloverleaf
    num_control_points_range: tuple[int, int] = (5, 8)  # Control points for splines
    
    # Circuit parameters  
    straight_length_range: tuple[float, float] = (2.0, 4.0)  # Length of straights
    corner_radius_range: tuple[float, float] = (0.5, 0.8)  # Corner tightness
    banking_range: tuple[float, float] = (0.1, 0.3)  # Banking in corners
    
    # Smoothing
    smooth_window: int = 5  # Window size for trajectory smoothing


class TrajectoryLibrary:
    """Manager for multiple trajectory libraries with per-environment assignment.
    
    Creates multiple independent libraries, each specialized in one trajectory type.
    Library types are distributed according to configured ratios.
    Environments are assigned to libraries using env_idx % num_libraries,
    ensuring diversity while keeping memory usage reasonable.
    """
    
    # Trajectory type names
    TRAJECTORY_TYPES = ["banked_loop", "figure8", "cloverleaf", "racing_circuit", "spline"]
    
    @staticmethod
    def _get_default_type_ratios() -> dict[str, float]:
        """Get default library type distribution."""
        return {
            "banked_loop": 0.20,
            "figure8": 0.20,
            "cloverleaf": 0.20,
            "racing_circuit": 0.20,
            "spline": 0.20,
        }

    def __init__(
        self,
        num_waypoints: int = 100,
        device: str = "cuda",
        seed: Optional[int] = None,
        cfg: Optional[TrajectoryGeneratorCfg] = None,
    ):
        """Initialize multiple trajectory libraries.

        Args:
            num_waypoints: Number of discrete waypoints per trajectory.
            device: Device for tensor operations.
            seed: Random seed for reproducibility.
            cfg: Configuration for trajectory generation parameters.
        """
        self.num_waypoints = num_waypoints
        self.device = device
        self.cfg = cfg if cfg is not None else TrajectoryGeneratorCfg()
        self.num_libraries = self.cfg.num_libraries
        self.trajectories_per_library = self.cfg.trajectories_per_library

        if seed is not None:
            torch.manual_seed(seed)

        # Get library type distribution
        type_ratios = self.cfg.library_type_ratios if self.cfg.library_type_ratios is not None else self._get_default_type_ratios()
        
        # Validate ratios
        ratio_sum = sum(type_ratios.values())
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError(f"Library type ratios must sum to 1.0, got {ratio_sum}")
        
        # Assign types to libraries based on ratios
        self.library_types = self._assign_library_types(type_ratios)
        
        # Print distribution
        type_counts = {t: self.library_types.count(t) for t in self.TRAJECTORY_TYPES}
        print(f"Generating {self.num_libraries} specialized trajectory libraries:")
        for traj_type, count in type_counts.items():
            if count > 0:
                print(f"  {traj_type}: {count} libraries ({count/self.num_libraries*100:.1f}%)")

        # Generate multiple libraries [num_libraries, trajectories_per_library, num_waypoints, 3]
        self.trajectories, self.tangents, self.difficulties = self._generate_all_libraries()
        
        # Sort trajectories by difficulty within each library [num_libraries, trajectories_per_library]
        self.difficulty_sorted_indices = torch.argsort(self.difficulties, dim=1)
        
        print(f"Total trajectories: {self.num_libraries * self.trajectories_per_library}")
        print(f"Difficulty range per library: [{self.difficulties.min(dim=1)[0].mean():.3f}, {self.difficulties.max(dim=1)[0].mean():.3f}]")

    def _assign_library_types(self, type_ratios: dict[str, float]) -> list[str]:
        """Assign trajectory types to libraries based on configured ratios.
        
        Args:
            type_ratios: Dictionary mapping trajectory type to ratio (0-1).
            
        Returns:
            List of trajectory types, one per library.
        """
        library_types = []
        
        for traj_type in self.TRAJECTORY_TYPES:
            if traj_type in type_ratios:
                num_libs = int(type_ratios[traj_type] * self.num_libraries)
                library_types.extend([traj_type] * num_libs)
        
        # Fill remaining libraries (due to rounding) with most common type
        while len(library_types) < self.num_libraries:
            most_common_type = max(type_ratios.keys(), key=lambda k: type_ratios[k])
            library_types.append(most_common_type)
        
        # Trim if we went over (due to rounding)
        library_types = library_types[:self.num_libraries]
        
        return library_types

    def _generate_all_libraries(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate multiple independent trajectory libraries.
        
        Returns:
            trajectories: [num_libraries, trajectories_per_library, num_waypoints, 3]
            tangents: [num_libraries, trajectories_per_library, num_waypoints, 3]
            difficulties: [num_libraries, trajectories_per_library]
        """
        all_trajectories = []
        all_tangents = []
        all_difficulties = []
        
        for lib_idx in range(self.num_libraries):
            # Use different seed offset for each library to ensure diversity
            seed_offset = lib_idx * 10000
            library_type = self.library_types[lib_idx]
            trajectories, tangents, difficulties = self._generate_single_library(seed_offset, library_type)
            all_trajectories.append(trajectories)
            all_tangents.append(tangents)
            all_difficulties.append(difficulties)
        
        # Stack into [num_libraries, trajectories_per_library, ...]
        all_trajectories = torch.stack(all_trajectories, dim=0)
        all_tangents = torch.stack(all_tangents, dim=0)
        all_difficulties = torch.stack(all_difficulties, dim=0)
        
        return all_trajectories, all_tangents, all_difficulties

    def get_library_for_env(self, env_idx: int) -> int:
        """Get the library index assigned to an environment.
        
        Args:
            env_idx: Environment index.
            
        Returns:
            Library index for this environment.
        """
        return env_idx % self.num_libraries
    
    def get_library_type(self, env_idx: int) -> str:
        """Get the trajectory type for an environment's assigned library.
        
        Args:
            env_idx: Environment index.
            
        Returns:
            Trajectory type string.
        """
        lib_idx = self.get_library_for_env(env_idx)
        return self.library_types[lib_idx]

    def get_trajectory_by_difficulty(self, env_idx: int, difficulty: float) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Get a trajectory based on environment and difficulty level.
        
        Args:
            env_idx: Environment index (determines which library to use).
            difficulty: Difficulty level from 0.0 (easiest) to 1.0 (hardest).
            
        Returns:
            waypoints: [num_waypoints, 3] trajectory positions
            tangents: [num_waypoints, 3] tangent vectors  
            index: The trajectory index that was selected within the library
        """
        # Select library based on environment
        lib_idx = self.get_library_for_env(env_idx)
        
        # Map difficulty to trajectory index within library (sorted by difficulty)
        traj_idx = int(difficulty * (self.trajectories_per_library - 1))
        traj_idx = min(max(traj_idx, 0), self.trajectories_per_library - 1)
        sorted_traj_idx = self.difficulty_sorted_indices[lib_idx, traj_idx].item()
        
        return self.trajectories[lib_idx, sorted_traj_idx], self.tangents[lib_idx, sorted_traj_idx], sorted_traj_idx

    def _generate_single_library(self, seed_offset: int = 0, library_type: str = "banked_loop") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a single library of racing trajectories of a specific type.
        
        Args:
            seed_offset: Offset to add to seeds for unique trajectories per library.
            library_type: Type of trajectories to generate in this library.

        Returns:
            trajectories: [trajectories_per_library, num_waypoints, 3] waypoint positions
            tangents: [trajectories_per_library, num_waypoints, 3] tangent vectors
            difficulties: [trajectories_per_library] difficulty score for each track (0-1)
        """
        trajectories = []
        tangents = []
        difficulties = []

        # Generate all trajectories of the specified type
        for i in range(self.trajectories_per_library):
            if library_type == "banked_loop":
                traj, tang, diff = self._generate_banked_loop(seed_offset + i)
            elif library_type == "figure8":
                traj, tang, diff = self._generate_figure8_3d(seed_offset + i)
            elif library_type == "cloverleaf":
                traj, tang, diff = self._generate_cloverleaf(seed_offset + i)
            elif library_type == "racing_circuit":
                traj, tang, diff = self._generate_racing_circuit(seed_offset + i)
            elif library_type == "spline":
                traj, tang, diff = self._generate_spline_track(seed_offset + i)
            else:
                raise ValueError(f"Unknown library type: {library_type}")
            
            trajectories.append(traj)
            tangents.append(tang)
            difficulties.append(diff)

        # Stack all trajectories
        trajectories = torch.stack(trajectories, dim=0)  # [num_traj, num_waypoints, 3]
        tangents = torch.stack(tangents, dim=0)  # [num_traj, num_waypoints, 3]
        difficulties = torch.tensor(difficulties, device=self.device)  # [num_traj]

        return trajectories, tangents, difficulties

    def _generate_banked_loop(self, seed: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Generate a banked circular or elliptical loop with vertical variation.

        Args:
            seed: Seed for variation.

        Returns:
            waypoints: [num_waypoints, 3] trajectory positions
            tangents: [num_waypoints, 3] normalized tangent vectors
            difficulty: Difficulty score (0-1)
        """
        torch.manual_seed(seed + 1000)
        
        # Random parameters from config
        radius_x = self.cfg.radius_range[0] + torch.rand(1).item() * (self.cfg.radius_range[1] - self.cfg.radius_range[0])
        radius_y = self.cfg.radius_range[0] + torch.rand(1).item() * (self.cfg.radius_range[1] - self.cfg.radius_range[0])
        height_base = self.cfg.height_range[0] + torch.rand(1).item() * (self.cfg.height_range[1] - self.cfg.height_range[0])
        height_amplitude = self.cfg.height_amplitude_range[0] + torch.rand(1).item() * (
            self.cfg.height_amplitude_range[1] - self.cfg.height_amplitude_range[0]
        )
        banking_angle = (torch.rand(1).item() - 0.5) * (self.cfg.banking_angle_range[1] - self.cfg.banking_angle_range[0])
        num_oscillations = torch.randint(self.cfg.num_oscillations_range[0], self.cfg.num_oscillations_range[1] + 1, (1,)).item()

        theta = torch.linspace(0, 2 * math.pi, self.num_waypoints, device=self.device)

        # Elliptical path with banking
        x = radius_x * torch.cos(theta + banking_angle * torch.sin(theta))
        y = radius_y * torch.sin(theta + banking_angle * torch.sin(theta))
        z = height_base + height_amplitude * torch.sin(num_oscillations * theta)

        waypoints = torch.stack([x, y, z], dim=1)
        tangents = self._compute_tangents(waypoints)
        
        # Compute difficulty (0-1): based on banking, oscillations, and height variation
        difficulty = (
            abs(banking_angle) / max(abs(self.cfg.banking_angle_range[1]), 1e-6) * 0.3 +
            (num_oscillations - self.cfg.num_oscillations_range[0]) / max(self.cfg.num_oscillations_range[1] - self.cfg.num_oscillations_range[0], 1) * 0.3 +
            height_amplitude / max(self.cfg.height_amplitude_range[1], 1e-6) * 0.4
        )

        return waypoints, tangents, difficulty

    def _generate_figure8_3d(self, seed: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Generate a 3D figure-8 (lemniscate) with vertical crossings.

        Args:
            seed: Seed for variation.

        Returns:
            waypoints: [num_waypoints, 3] trajectory positions
            tangents: [num_waypoints, 3] normalized tangent vectors
            difficulty: Difficulty score (0-1)
        """
        torch.manual_seed(seed + 2000)

        scale = self.cfg.radius_range[0] + torch.rand(1).item() * (self.cfg.radius_range[1] - self.cfg.radius_range[0])
        height_base = self.cfg.height_range[0] + torch.rand(1).item() * (self.cfg.height_range[1] - self.cfg.height_range[0])
        height_scale = self.cfg.height_amplitude_range[0] + torch.rand(1).item() * (
            self.cfg.height_amplitude_range[1] - self.cfg.height_amplitude_range[0]
        )
        twist = (torch.rand(1).item() - 0.5) * math.pi  # Add twist to the 8

        theta = torch.linspace(0, 2 * math.pi, self.num_waypoints, device=self.device)

        # Lemniscate equations with 3D extension
        x = scale * torch.sin(theta + twist)
        y = scale * torch.sin(theta + twist) * torch.cos(theta + twist)
        z = height_base + height_scale * torch.sin(2 * theta)  # Cross over in Z

        waypoints = torch.stack([x, y, z], dim=1)
        tangents = self._compute_tangents(waypoints)
        
        # Difficulty based on scale, height variation, and twist
        difficulty = 0.2 + (
            (scale - self.cfg.radius_range[0]) / max(self.cfg.radius_range[1] - self.cfg.radius_range[0], 1e-6) * 0.3 +
            height_scale / max(self.cfg.height_amplitude_range[1], 1e-6) * 0.3 +
            abs(twist) / math.pi * 0.2
        )

        return waypoints, tangents, difficulty

    def _generate_cloverleaf(self, seed: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Generate a cloverleaf pattern (3 or 4 lobes).

        Args:
            seed: Seed for variation.

        Returns:
            waypoints: [num_waypoints, 3] trajectory positions
            tangents: [num_waypoints, 3] normalized tangent vectors
            difficulty: Difficulty score (0-1)
        """
        torch.manual_seed(seed + 3000)

        num_lobes = torch.randint(self.cfg.num_lobes_range[0], self.cfg.num_lobes_range[1] + 1, (1,)).item()
        scale = self.cfg.radius_range[0] + torch.rand(1).item() * (self.cfg.radius_range[1] - self.cfg.radius_range[0])
        height_base = self.cfg.height_range[0] + torch.rand(1).item() * (self.cfg.height_range[1] - self.cfg.height_range[0])
        height_var = self.cfg.height_amplitude_range[0] + torch.rand(1).item() * (
            self.cfg.height_amplitude_range[1] - self.cfg.height_amplitude_range[0]
        )

        theta = torch.linspace(0, 2 * math.pi, self.num_waypoints, device=self.device)

        # Polar rose equations (r = cos(k*theta))
        r = scale * (1 + 0.5 * torch.cos(num_lobes * theta))
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        z = height_base + height_var * torch.sin(num_lobes * theta)

        waypoints = torch.stack([x, y, z], dim=1)
        tangents = self._compute_tangents(waypoints)
        
        # Difficulty based on number of lobes and height variation
        difficulty = 0.4 + (
            (num_lobes - self.cfg.num_lobes_range[0]) / max(self.cfg.num_lobes_range[1] - self.cfg.num_lobes_range[0], 1) * 0.4 +
            height_var / max(self.cfg.height_amplitude_range[1], 1e-6) * 0.2
        )

        return waypoints, tangents, difficulty

    def _generate_racing_circuit(self, seed: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Generate a racing circuit with straights, chicanes, and hairpins.

        Args:
            seed: Seed for variation.

        Returns:
            waypoints: [num_waypoints, 3] trajectory positions
            tangents: [num_waypoints, 3] normalized tangent vectors
            difficulty: Difficulty score (0-1)
        """
        torch.manual_seed(seed + 4000)

        # Use a rounded rectangle approach with random corner radii
        straight_length = self.cfg.straight_length_range[0] + torch.rand(1).item() * (
            self.cfg.straight_length_range[1] - self.cfg.straight_length_range[0]
        )
        width = self.cfg.radius_range[0] + torch.rand(1).item() * (self.cfg.radius_range[1] - self.cfg.radius_range[0])
        corner_radius_factor = self.cfg.corner_radius_range[0] + torch.rand(1).item() * (
            self.cfg.corner_radius_range[1] - self.cfg.corner_radius_range[0]
        )
        height_base = self.cfg.height_range[0] + torch.rand(1).item() * (self.cfg.height_range[1] - self.cfg.height_range[0])
        banking = self.cfg.banking_range[0] + torch.rand(1).item() * (
            self.cfg.banking_range[1] - self.cfg.banking_range[0]
        )

        # Create track sections
        waypoints_list = []

        # Four corners with straights between them
        corners = [
            (straight_length / 2, width / 2),  # Top-right
            (-straight_length / 2, width / 2),  # Top-left
            (-straight_length / 2, -width / 2),  # Bottom-left
            (straight_length / 2, -width / 2),  # Bottom-right
        ]

        points_per_section = self.num_waypoints // 4

        for i in range(4):
            start_corner = corners[i]
            end_corner = corners[(i + 1) % 4]

            # Generate curved transition
            for j in range(points_per_section):
                t = j / points_per_section
                
                # Blend between corners with smooth curve
                x = start_corner[0] + (end_corner[0] - start_corner[0]) * self._smoothstep(t)
                y = start_corner[1] + (end_corner[1] - start_corner[1]) * self._smoothstep(t)
                
                # Add banking in corners
                corner_factor = 4 * t * (1 - t)  # Peaks at 0.5
                z = height_base + banking * corner_factor
                
                waypoints_list.append(torch.tensor([x, y, z], device=self.device))

        waypoints = torch.stack(waypoints_list, dim=0)
        
        # Smooth the trajectory
        waypoints = self._smooth_trajectory(waypoints, window=self.cfg.smooth_window)
        tangents = self._compute_tangents(waypoints)
        
        # Difficulty based on tight corners (high corner_radius_factor = easier, low = harder) and banking
        difficulty = 0.5 + (
            (1.0 - corner_radius_factor) / max(1.0 - self.cfg.corner_radius_range[0], 1e-6) * 0.3 +
            banking / max(self.cfg.banking_range[1], 1e-6) * 0.2
        )

        return waypoints, tangents, difficulty

    def _generate_spline_track(self, seed: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Generate complex track using Catmull-Rom splines through random control points.

        Args:
            seed: Seed for variation.

        Returns:
            waypoints: [num_waypoints, 3] trajectory positions
            tangents: [num_waypoints, 3] normalized tangent vectors
            difficulty: Difficulty score (0-1)
        """
        torch.manual_seed(seed + 5000)

        # Generate random control points
        num_control_points = torch.randint(
            self.cfg.num_control_points_range[0], 
            self.cfg.num_control_points_range[1] + 1, 
            (1,)
        ).item()
        
        # Control points in polar coordinates for closed loop
        angles = torch.linspace(0, 2 * math.pi, num_control_points + 1, device=self.device)[:-1]
        radii = self.cfg.radius_range[0] + torch.rand(num_control_points, device=self.device) * (
            self.cfg.radius_range[1] - self.cfg.radius_range[0]
        )
        heights = self.cfg.height_range[0] + torch.rand(num_control_points, device=self.device) * (
            self.cfg.height_range[1] - self.cfg.height_range[0]
        )

        control_points_x = radii * torch.cos(angles)
        control_points_y = radii * torch.sin(angles)
        control_points_z = heights

        control_points = torch.stack([control_points_x, control_points_y, control_points_z], dim=1)

        # Interpolate using Catmull-Rom spline
        waypoints = self._catmull_rom_spline(control_points, self.num_waypoints)
        tangents = self._compute_tangents(waypoints)
        
        # Difficulty based on number of control points (more = more complex) and height variation
        height_variation = (heights.max() - heights.min()).item()
        difficulty = 0.7 + (
            (num_control_points - self.cfg.num_control_points_range[0]) / 
            max(self.cfg.num_control_points_range[1] - self.cfg.num_control_points_range[0], 1) * 0.2 +
            height_variation / max(self.cfg.height_range[1] - self.cfg.height_range[0], 1e-6) * 0.1
        )

        return waypoints, tangents, difficulty

    def _catmull_rom_spline(self, control_points: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Interpolate points using Catmull-Rom spline.

        Args:
            control_points: [num_control, 3] control point positions
            num_samples: Number of interpolated points

        Returns:
            interpolated: [num_samples, 3] smooth trajectory
        """
        num_control = control_points.shape[0]
        
        # Extend control points for closed loop (wrap around)
        extended = torch.cat([
            control_points[-1:],
            control_points,
            control_points[:2]
        ], dim=0)

        samples = []
        points_per_segment = num_samples // num_control

        for i in range(num_control):
            p0 = extended[i]
            p1 = extended[i + 1]
            p2 = extended[i + 2]
            p3 = extended[i + 3]

            for j in range(points_per_segment):
                t = j / points_per_segment
                t2 = t * t
                t3 = t2 * t

                # Catmull-Rom basis functions
                point = 0.5 * (
                    (2 * p1) +
                    (-p0 + p2) * t +
                    (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                    (-p0 + 3 * p1 - 3 * p2 + p3) * t3
                )
                samples.append(point)

        # Fill remaining points
        while len(samples) < num_samples:
            samples.append(samples[0])

        return torch.stack(samples[:num_samples], dim=0)

    def _compute_tangents(self, waypoints: torch.Tensor) -> torch.Tensor:
        """Compute normalized tangent vectors for waypoints.

        Args:
            waypoints: [num_waypoints, 3] trajectory positions

        Returns:
            tangents: [num_waypoints, 3] normalized tangent vectors
        """
        tangents = torch.zeros_like(waypoints)
        tangents[:-1] = waypoints[1:] - waypoints[:-1]
        tangents[-1] = waypoints[0] - waypoints[-1]  # Close the loop
        
        # Normalize
        tangents = tangents / (torch.norm(tangents, dim=1, keepdim=True) + 1e-8)
        
        return tangents

    def _smooth_trajectory(self, waypoints: torch.Tensor, window: int = 5) -> torch.Tensor:
        """Smooth trajectory using moving average.

        Args:
            waypoints: [num_waypoints, 3] trajectory positions
            window: Window size for smoothing

        Returns:
            smoothed: [num_waypoints, 3] smoothed trajectory
        """
        smoothed = torch.zeros_like(waypoints)
        half_window = window // 2

        for i in range(len(waypoints)):
            indices = [(i + j - half_window) % len(waypoints) for j in range(window)]
            smoothed[i] = waypoints[indices].mean(dim=0)

        return smoothed

    def _smoothstep(self, t: float) -> float:
        """Smooth interpolation function (3t^2 - 2t^3).

        Args:
            t: Parameter in [0, 1]

        Returns:
            Smoothed value in [0, 1]
        """
        return t * t * (3.0 - 2.0 * t)

    def get_trajectory(self, env_idx: int, traj_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a specific trajectory from the library assigned to an environment.

        Args:
            env_idx: Environment index (determines which library to use).
            traj_index: Trajectory index within the library (will be wrapped if out of bounds).

        Returns:
            waypoints: [num_waypoints, 3] trajectory positions
            tangents: [num_waypoints, 3] tangent vectors
        """
        lib_idx = self.get_library_for_env(env_idx)
        traj_index = traj_index % self.trajectories_per_library
        return self.trajectories[lib_idx, traj_index], self.tangents[lib_idx, traj_index]

    def get_random_trajectory(self, env_idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Get a random trajectory from the library assigned to an environment.
        
        Args:
            env_idx: Environment index (determines which library to use).

        Returns:
            waypoints: [num_waypoints, 3] trajectory positions
            tangents: [num_waypoints, 3] tangent vectors
            index: The trajectory index that was selected within the library
        """
        lib_idx = self.get_library_for_env(env_idx)
        traj_index = torch.randint(0, self.trajectories_per_library, (1,), device=self.device).item()
        return self.trajectories[lib_idx, traj_index], self.tangents[lib_idx, traj_index], traj_index
