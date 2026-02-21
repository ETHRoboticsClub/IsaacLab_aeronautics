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
from tqdm import tqdm


@dataclass
class TrajectoryGeneratorCfg:
    """Configuration for trajectory generation parameters."""
    
    # Library parameters
    num_libraries: int = -1  # Number of independent trajectory libraries (-1 = one per environment)
    trajectories_per_library: int = 3  # Trajectories per library
    
    # Library type distribution (must sum to 1.0)
    # Each library will specialize in one trajectory type
    library_type_ratios: dict[str, float] | None = None  # If None, uses default
    # Default: {"banked_loop": 0.20, "figure8": 0.20, "cloverleaf": 0.20, 
    #          "racing_circuit": 0.20, "spline": 0.20}
    
    # Size parameters
    radius_range: tuple[float, float] = (1.0, 4.0)  # Min/max radius for trajectories (expanded)
    height_range: tuple[float, float] = (1.8, 5.5)  # Min/max base height (expanded)
    height_amplitude_range: tuple[float, float] = (0.1, 1.0)  # Vertical variation (expanded)
    
    # Complexity parameters
    banking_angle_range: tuple[float, float] = (-0.4, 0.4)  # Banking angle in radians (doubled)
    num_oscillations_range: tuple[int, int] = (1, 4)  # Vertical waves for loops (expanded)
    num_lobes_range: tuple[int, int] = (2, 5)  # Lobes for cloverleaf (expanded)
    num_control_points_range: tuple[int, int] = (4, 10)  # Control points for splines (expanded)
    
    # Circuit parameters  
    straight_length_range: tuple[float, float] = (1.5, 5.0)  # Length of straights (expanded)
    corner_radius_range: tuple[float, float] = (0.3, 1.2)  # Corner tightness (expanded for sharper corners)
    banking_range: tuple[float, float] = (0.05, 0.5)  # Banking in corners (expanded)
    
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
        waypoints_density: float = 0.5,
        device: str = "cuda",
        seed: Optional[int] = None,
        cfg: Optional[TrajectoryGeneratorCfg] = None,
        num_environments: Optional[int] = None,
    ):
        """Initialize multiple trajectory libraries.

        Args:
            waypoints_density: Density of waypoints per meter of trajectory.
            device: Device for tensor operations.
            seed: Random seed for reproducibility.
            cfg: Configuration for trajectory generation parameters.
            num_environments: Number of environments. If provided and cfg.num_libraries == -1,
                            will set num_libraries = num_environments.
        """
        self.waypoints_density = waypoints_density
        self.device = device
        self.cfg = cfg if cfg is not None else TrajectoryGeneratorCfg()
        
        # If num_libraries is -1, use num_environments (one library per env), capped at 50
        if self.cfg.num_libraries == -1:
            if num_environments is None:
                raise ValueError("num_environments must be provided when cfg.num_libraries == -1")
            self.num_libraries = min(num_environments, 50)
        else:
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
                print(f"  {traj_type}: {count} libraries")

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
        
        for lib_idx in tqdm(range(self.num_libraries), desc="Generating trajectory libraries"):
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
        
        # Normalize difficulties to span [0, 1] for this library
        min_diff = difficulties.min()
        max_diff = difficulties.max()
        if max_diff - min_diff > 0.02:
            difficulties = (difficulties - min_diff) / (max_diff - min_diff)
        else:
            # All tracks have similar difficulty, map to mid-range
            difficulties = torch.ones_like(difficulties) * 0.5

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
        torch.manual_seed(seed * 10007 + 1000)
        
        # Random parameters from config with better seed diversity
        radius_x = self.cfg.radius_range[0] + torch.rand(1).item() * (self.cfg.radius_range[1] - self.cfg.radius_range[0])
        radius_y = self.cfg.radius_range[0] + torch.rand(1).item() * (self.cfg.radius_range[1] - self.cfg.radius_range[0])
        height_base = self.cfg.height_range[0] + torch.rand(1).item() * (self.cfg.height_range[1] - self.cfg.height_range[0])
        height_amplitude = self.cfg.height_amplitude_range[0] + torch.rand(1).item() * (
            self.cfg.height_amplitude_range[1] - self.cfg.height_amplitude_range[0]
        )
        banking_angle = (torch.rand(1).item() - 0.5) * (self.cfg.banking_angle_range[1] - self.cfg.banking_angle_range[0])
        num_oscillations = torch.randint(self.cfg.num_oscillations_range[0], self.cfg.num_oscillations_range[1] + 1, (1,)).item()

        # First generate coarse trajectory to compute length
        coarse_n = 100
        theta = torch.linspace(0, 2 * math.pi, coarse_n, device=self.device)

        # Elliptical path with banking
        x = radius_x * torch.cos(theta + banking_angle * torch.sin(theta))
        y = radius_y * torch.sin(theta + banking_angle * torch.sin(theta))
        z = height_base + height_amplitude * torch.sin(num_oscillations * theta)

        coarse_waypoints = torch.stack([x, y, z], dim=1)
        
        # Resample with proper density
        waypoints = self._resample_trajectory_with_density(coarse_waypoints)
        tangents = self._compute_tangents(waypoints)
        
        # Compute difficulty (0-1): based on banking, oscillations, height variation, and radius extremes
        banking_difficulty = abs(banking_angle) / max(abs(self.cfg.banking_angle_range[1]), 1e-6)
        oscillation_difficulty = (num_oscillations - self.cfg.num_oscillations_range[0]) / max(self.cfg.num_oscillations_range[1] - self.cfg.num_oscillations_range[0], 1)
        height_difficulty = height_amplitude / max(self.cfg.height_amplitude_range[1], 1e-6)
        radius_difficulty = (min(radius_x, radius_y) - self.cfg.radius_range[0]) / max(self.cfg.radius_range[1] - self.cfg.radius_range[0], 1e-6)
        
        difficulty = (
            banking_difficulty * 0.25 +
            oscillation_difficulty * 0.25 +
            height_difficulty * 0.25 +
            radius_difficulty * 0.25
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
        torch.manual_seed(seed * 10009 + 2000)

        scale = self.cfg.radius_range[0] + torch.rand(1).item() * (self.cfg.radius_range[1] - self.cfg.radius_range[0])
        height_base = self.cfg.height_range[0] + torch.rand(1).item() * (self.cfg.height_range[1] - self.cfg.height_range[0])
        height_scale = self.cfg.height_amplitude_range[0] + torch.rand(1).item() * (
            self.cfg.height_amplitude_range[1] - self.cfg.height_amplitude_range[0]
        )
        twist = (torch.rand(1).item() - 0.5) * math.pi * 2  # Add more variation to twist

        # First generate coarse trajectory to compute length
        coarse_n = 100
        theta = torch.linspace(0, 2 * math.pi, coarse_n, device=self.device)

        # Lemniscate equations with 3D extension
        x = scale * torch.sin(theta + twist)
        y = scale * torch.sin(theta + twist) * torch.cos(theta + twist)
        z = height_base + height_scale * torch.sin(2 * theta)  # Cross over in Z

        coarse_waypoints = torch.stack([x, y, z], dim=1)
        
        # Resample with proper density
        waypoints = self._resample_trajectory_with_density(coarse_waypoints)
        tangents = self._compute_tangents(waypoints)
        
        # Difficulty based on scale, height variation, and twist
        scale_difficulty = (scale - self.cfg.radius_range[0]) / max(self.cfg.radius_range[1] - self.cfg.radius_range[0], 1e-6)
        height_difficulty = height_scale / max(self.cfg.height_amplitude_range[1], 1e-6)
        twist_difficulty = abs(twist) / (math.pi * 2)
        
        difficulty = (
            scale_difficulty * 0.3 +
            height_difficulty * 0.4 +
            twist_difficulty * 0.3
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
        torch.manual_seed(seed * 10007 + 3000)

        num_lobes = torch.randint(self.cfg.num_lobes_range[0], self.cfg.num_lobes_range[1] + 1, (1,)).item()
        scale = self.cfg.radius_range[0] + torch.rand(1).item() * (self.cfg.radius_range[1] - self.cfg.radius_range[0])
        height_base = self.cfg.height_range[0] + torch.rand(1).item() * (self.cfg.height_range[1] - self.cfg.height_range[0])
        height_var = self.cfg.height_amplitude_range[0] + torch.rand(1).item() * (
            self.cfg.height_amplitude_range[1] - self.cfg.height_amplitude_range[0]
        )

        # First generate coarse trajectory to compute length
        coarse_n = 100
        theta = torch.linspace(0, 2 * math.pi, coarse_n, device=self.device)

        # Polar rose equations (r = cos(k*theta))
        r = scale * (1 + 0.5 * torch.cos(num_lobes * theta))
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        z = height_base + height_var * torch.sin(num_lobes * theta)

        coarse_waypoints = torch.stack([x, y, z], dim=1)
        
        # Resample with proper density
        waypoints = self._resample_trajectory_with_density(coarse_waypoints)
        tangents = self._compute_tangents(waypoints)
        
        # Difficulty based on number of lobes, scale, and height variation
        lobe_difficulty = (num_lobes - self.cfg.num_lobes_range[0]) / max(self.cfg.num_lobes_range[1] - self.cfg.num_lobes_range[0], 1)
        scale_difficulty = (scale - self.cfg.radius_range[0]) / max(self.cfg.radius_range[1] - self.cfg.radius_range[0], 1e-6)
        height_difficulty = height_var / max(self.cfg.height_amplitude_range[1], 1e-6)
        
        difficulty = (
            lobe_difficulty * 0.35 +
            scale_difficulty * 0.35 +
            height_difficulty * 0.30
        )

        return waypoints, tangents, difficulty

    def _generate_racing_circuit(self, seed: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Generate a racing circuit using random walk with velocity vector perturbations.
        
        The trajectory is generated by starting with an initial velocity vector, then at each
        step: (1) advance by the velocity, (2) perturb the velocity magnitude and direction.
        The last portion smoothly closes back to the start point.

        Args:
            seed: Seed for variation.

        Returns:
            waypoints: [num_waypoints, 3] trajectory positions
            tangents: [num_waypoints, 3] normalized tangent vectors
            difficulty: Difficulty score (0-1)
        """
        # Use multiple random generators with different seeds for better variety at all difficulty levels
        # Use CPU generators for random number generation (PyTorch requirement)
        rng1 = torch.Generator(device="cpu").manual_seed(seed + 4000)
        rng2 = torch.Generator(device="cpu").manual_seed(seed + 2000)
        rng3 = torch.Generator(device="cpu").manual_seed(seed + 3000)

        # Random parameters controlling trajectory characteristics
        base_speed = self.cfg.radius_range[0] * 0.3 + torch.rand(1, generator=rng1).item() * (self.cfg.radius_range[1] - self.cfg.radius_range[0]) * 0.3
        speed_variation = 0.15 + torch.rand(1, generator=rng2).item() * 0.35  # How much speed varies (0.15-0.5)
        angular_variation = 0.2 + torch.rand(1, generator=rng3).item() * 0.6  # How sharply direction changes (0.2-0.8 rad)
        height_base = self.cfg.height_range[0] + torch.rand(1, generator=rng1).item() * (self.cfg.height_range[1] - self.cfg.height_range[0])
        height_variation = 0.1 + torch.rand(1, generator=rng2).item() * 0.4  # Vertical variation
        
        # Add shape bias parameters - these affect trajectory shape but not difficulty
        # This ensures variety even at same difficulty level
        turn_bias = (torch.rand(1, generator=rng1).item() - 0.5) * 0.3  # Bias to turn left or right
        spiral_factor = (torch.rand(1, generator=rng2).item() - 0.5) * 0.1  # Tendency to spiral outward/inward
        speed_phase = torch.rand(1, generator=rng3).item() * 2 * math.pi  # Phase offset for speed variations
        
        # Add base variations that ensure diversity even at low difficulty
        # These provide minimum variation regardless of difficulty
        min_speed_variation = 0.1
        min_angular_variation = 0.15
        min_height_variation = 0.08
        
        # Determine closure point (start closing early at 70-80% of trajectory for smooth return)
        closure_start_ratio = 0.70 + torch.rand(1, generator=rng3).item() * 0.10
        
        # Use coarse sampling for initial generation
        coarse_n = 100
        closure_start_idx = int(coarse_n * closure_start_ratio)
        
        # Initialize trajectory
        waypoints_list = []
        current_pos = torch.tensor([0.0, 0.0, height_base], device=self.device)
        
        # Initial velocity: random direction in XY plane
        initial_angle = torch.rand(1, generator=rng1).item() * 2 * math.pi
        velocity = torch.tensor([
            base_speed * math.cos(initial_angle),
            base_speed * math.sin(initial_angle),
            0.0
        ], device=self.device)
        
        # Use different random generators in loop for maximum variety (use CPU for PyTorch compatibility)
        loop_rng1 = torch.Generator(device="cpu").manual_seed(seed + 1000)
        loop_rng2 = torch.Generator(device="cpu").manual_seed(seed + 5000)
        loop_rng3 = torch.Generator(device="cpu").manual_seed(seed + 7000)
        
        # Track metrics for difficulty computation
        total_turn_magnitude = 0.0
        total_speed_changes = 0.0
        max_height_change = 0.0
        min_height = height_base
        max_height = height_base
        
        # Generate waypoints via random walk
        for i in range(coarse_n - 1):  # Generate one less point since we'll close the loop
            waypoints_list.append(current_pos.clone())
            
            # Check if we're in the closure phase
            if i >= closure_start_idx:
                # Gradually steer back to origin
                closure_progress = (i - closure_start_idx) / (coarse_n - 1 - closure_start_idx)
                
                # Target is the starting position
                to_start = -current_pos  # Vector pointing to origin
                to_start[2] = height_base - current_pos[2]  # Return to base height
                
                # Very gentle blend - use cubic smoothing for even softer transition
                blend_weight = closure_progress ** 3  # Even gentler than smoothstep
                
                # Continue random walk but gently bias toward origin
                # Still apply random perturbations during closure
                speed = torch.norm(velocity[:2])
                speed_change = (torch.rand(1, generator=loop_rng1).item() - 0.5) * 2 * speed_variation * base_speed * (1 - blend_weight * 0.5)
                new_speed = torch.clamp(speed + speed_change, base_speed * 0.5, base_speed * 2.0)
                total_speed_changes += abs(speed_change)
                
                current_angle = math.atan2(velocity[1].item(), velocity[0].item())
                angle_change = (torch.rand(1, generator=loop_rng2).item() - 0.5) * 2 * angular_variation * (1 - blend_weight * 0.7)
                total_turn_magnitude += abs(angle_change)
                new_angle = current_angle + angle_change
                
                # Random walk velocity component
                velocity[0] = new_speed * math.cos(new_angle)
                velocity[1] = new_speed * math.sin(new_angle)
                
                z_change = (torch.rand(1, generator=loop_rng3).item() - 0.5) * 2 * height_variation * base_speed * (1 - blend_weight)
                velocity[2] = torch.clamp(velocity[2] + z_change, -base_speed * 0.5, base_speed * 0.5)
                
                # Gently blend in the return-to-origin component
                to_start_norm = to_start / (torch.norm(to_start) + 1e-8)
                target_velocity = to_start_norm * base_speed
                velocity = (1 - blend_weight) * velocity + blend_weight * target_velocity
            else:
                # Random walk phase: perturb velocity
                
                # Perturb speed (magnitude) - use loop_rng1
                # Always apply at least minimum variation for diversity
                speed = torch.norm(velocity[:2])  # XY speed
                speed_change = (torch.rand(1, generator=loop_rng1).item() - 0.5) * 2 * max(speed_variation, min_speed_variation) * base_speed
                new_speed = torch.clamp(speed + speed_change, base_speed * 0.5, base_speed * 2.0)
                total_speed_changes += abs(speed_change)
                
                # Perturb direction (angular change in XY plane) - use loop_rng2
                # Always apply at least minimum variation for diversity
                # Add turn bias to create consistent directional preference
                current_angle = math.atan2(velocity[1].item(), velocity[0].item())
                angle_change = (torch.rand(1, generator=loop_rng2).item() - 0.5) * 2 * max(angular_variation, min_angular_variation)
                angle_change += turn_bias  # Add consistent directional bias
                total_turn_magnitude += abs(angle_change)
                new_angle = current_angle + angle_change
                
                # Apply spiral factor - gradually change radius
                distance_from_origin = torch.norm(current_pos[:2])
                spiral_adjustment = spiral_factor * (1.0 if distance_from_origin > 1.0 else -1.0)
                new_speed = new_speed * (1.0 + spiral_adjustment)
                
                # Update XY velocity
                velocity[0] = new_speed * math.cos(new_angle)
                velocity[1] = new_speed * math.sin(new_angle)
                
                # Perturb Z velocity (vertical) - use loop_rng3
                # Always apply at least minimum variation for diversity
                z_change = (torch.rand(1, generator=loop_rng3).item() - 0.5) * 2 * max(height_variation, min_height_variation) * base_speed
                velocity[2] = torch.clamp(velocity[2] + z_change, -base_speed * 0.5, base_speed * 0.5)
            
            # Advance position
            current_pos = current_pos + velocity
            
            # Track height statistics
            min_height = min(min_height, current_pos[2].item())
            max_height = max(max_height, current_pos[2].item())
        
        max_height_change = max_height - min_height
        
        waypoints = torch.stack(waypoints_list, dim=0)
        
        # Smooth the trajectory to reduce jitter
        waypoints = self._smooth_trajectory(waypoints, window=self.cfg.smooth_window)
        
        # Close the loop by appending the first point as the last point
        coarse_waypoints = torch.cat([waypoints, waypoints[0:1]], dim=0)
        
        # Resample with proper density
        waypoints = self._resample_trajectory_with_density(coarse_waypoints)
        tangents = self._compute_tangents(waypoints)
        
        # Compute difficulty based on trajectory characteristics
        # Normalize metrics with better scaling to spread difficulty range
        turn_difficulty = min(total_turn_magnitude / (coarse_n * 0.2), 1.0)  # More sensitive to turns
        speed_difficulty = min(total_speed_changes / (coarse_n * base_speed * 0.3), 1.0)  # More sensitive to speed changes
        height_difficulty = min(max_height_change / 1.5, 1.0)  # More sensitive to height changes
        
        # Add variation parameter difficulty
        variation_difficulty = (speed_variation - 0.15) / 0.35  # Normalized to 0-1
        angular_difficulty = (angular_variation - 0.2) / 0.6  # Normalized to 0-1
        
        difficulty = (
            turn_difficulty * 0.25 +
            speed_difficulty * 0.20 +
            height_difficulty * 0.20 +
            variation_difficulty * 0.175 +
            angular_difficulty * 0.175
        )

        return waypoints, tangents, difficulty

    def _generate_spline_track(self, seed: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Generate complex track using Catmull-Rom splines through random control points with varied turns.

        Args:
            seed: Seed for variation.

        Returns:
            waypoints: [num_waypoints, 3] trajectory positions
            tangents: [num_waypoints, 3] normalized tangent vectors
            difficulty: Difficulty score (0-1)
        """
        torch.manual_seed(seed * 10007 + 5000)

        # Generate random control points with non-uniform angular distribution for more turns
        num_control_points = torch.randint(
            self.cfg.num_control_points_range[0], 
            self.cfg.num_control_points_range[1] + 1, 
            (1,)
        ).item()
        
        # Non-uniform angles to create varied turn density
        # Add random perturbations to create clusters and gaps
        base_angles = torch.linspace(0, 2 * math.pi, num_control_points + 1, device=self.device)[:-1]
        angle_perturbations = (torch.rand(num_control_points, device=self.device) - 0.5) * (math.pi / num_control_points) * 2
        angles = base_angles + angle_perturbations
        angles = angles.sort()[0]  # Keep them sorted to maintain connectivity
        
        # Vary radii significantly to create sharp and gentle turns
        radii = self.cfg.radius_range[0] + torch.rand(num_control_points, device=self.device) * (
            self.cfg.radius_range[1] - self.cfg.radius_range[0]
        )
        # Add some extreme radius variations for interesting turns
        radii = radii * (1.0 + torch.rand(num_control_points, device=self.device) * 1.0)
        
        # Vary heights more dramatically
        heights = self.cfg.height_range[0] + torch.rand(num_control_points, device=self.device) * (
            self.cfg.height_range[1] - self.cfg.height_range[0]
        )

        control_points_x = radii * torch.cos(angles)
        control_points_y = radii * torch.sin(angles)
        control_points_z = heights

        control_points = torch.stack([control_points_x, control_points_y, control_points_z], dim=1)

        # First generate coarse spline to compute length
        coarse_waypoints = self._catmull_rom_spline(control_points, 100)
        
        # Resample with proper density
        waypoints = self._resample_trajectory_with_density(coarse_waypoints)
        tangents = self._compute_tangents(waypoints)
        
        # Difficulty based on number of control points, radius variation, and height variation
        height_variation = (heights.max() - heights.min()).item()
        radius_variation = (radii.max() - radii.min()).item()
        
        control_point_difficulty = (num_control_points - self.cfg.num_control_points_range[0]) / max(
            self.cfg.num_control_points_range[1] - self.cfg.num_control_points_range[0], 1
        )
        height_difficulty = height_variation / max(self.cfg.height_range[1] - self.cfg.height_range[0], 1e-6)
        radius_difficulty = radius_variation / max(self.cfg.radius_range[1] - self.cfg.radius_range[0], 1e-6)
        
        difficulty = (
            control_point_difficulty * 0.4 +
            height_difficulty * 0.3 +
            radius_difficulty * 0.3
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

    def _resample_trajectory_with_density(self, coarse_waypoints: torch.Tensor) -> torch.Tensor:
        """Resample trajectory with proper waypoint density based on arc length.
        
        Args:
            coarse_waypoints: [coarse_n, 3] coarsely sampled trajectory points
            
        Returns:
            waypoints: [n, 3] properly resampled trajectory points with FIXED count
        """
        # Compute arc length from coarse sampling
        segment_lengths = torch.norm(coarse_waypoints[1:] - coarse_waypoints[:-1], dim=1)
        trajectory_length = segment_lengths.sum().item()
        
        # Use a FIXED number of waypoints (100) for consistency across all trajectories
        # This ensures all trajectories have the same waypoint count for stacking in library
        n = 100
        
        # Create cumulative arc length for interpolation
        cumulative_lengths = torch.cat([torch.zeros(1, device=self.device), segment_lengths.cumsum(0)])
        total_length = cumulative_lengths[-1]
        
        # Create target arc lengths for resampling
        target_lengths = torch.linspace(0, total_length, n, device=self.device)
        
        # Interpolate waypoints at target lengths
        resampled_waypoints = torch.zeros(n, 3, device=self.device)
        
        for i, target_length in enumerate(target_lengths):
            # Find segment containing this arc length
            idx = torch.searchsorted(cumulative_lengths, target_length)
            idx = torch.clamp(idx, 1, len(cumulative_lengths) - 1)
            
            # Linear interpolation within segment
            t = (target_length - cumulative_lengths[idx-1]) / (cumulative_lengths[idx] - cumulative_lengths[idx-1] + 1e-8)
            resampled_waypoints[i] = coarse_waypoints[idx-1] * (1 - t) + coarse_waypoints[idx] * t
            
        return resampled_waypoints
