# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_rotate_inverse

from .trajectory import RacingTrajectory, TrajectoryConfig
from .managers import (
    RewardManager,
    RewardConfig,
    ObservationManager,
    ObservationConfig,
    DisturbanceManager,
)
from .curriculum import TrajectoryCurriculumManager, CurriculumCfg

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: "RacingQuadcopterEnv", window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class RacingQuadcopterEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 15.0
    decimation = 2
    action_space = 4
    # observation_space calculation:
    # base: lin_vel(3) + ang_vel(3) + gravity(3) + velocity_limit(1) = 10
    # optional: contour_error(3) + speed_error(1) + progress(1) + gate_proximity(1) = 6
    # lookahead points: 6 * 3 = 18
    # history: (lin_vel(3) + ang_vel(3) + actions(4)) * history_length
    state_space = 0
    debug_vis = True

    lookahead_distances = (0.0, 0.1, 0.3, 0.5, 0.9, 1.4)

    # velocity limit configuration
    velocity_limit: float = 30.0  # Maximum allowed speed (m/s)
    randomize_velocity_limit: bool = False  # Randomize limit per episode
    velocity_limit_range: tuple[float, float] = (1.5, 4.0)  # Random range for velocity limit

    # trajectory config
    trajectory: TrajectoryConfig = TrajectoryConfig(
        trajectory_type="library",
        radius=5.0,
        height=1.5,
        waypoints_density=6.0,
        desired_speed=1.0,
        lookahead_distances=lookahead_distances,
        use_trajectory_library=True,
        track_seed=42,
        trajectories_per_env=1,  # Each environment gets 1 unique trajectory (set to 3 for curriculum learning)
        gate_spacing=4.0,
        gate_size=0.5,
    )

    # observation config
    observation: ObservationConfig = ObservationConfig(
        history_length=0,
        lookahead_distances=lookahead_distances,
        include_velocity_limit=True,
        include_contour_error=True,
        include_speed_error=True,
        include_gate_proximity=True,
    )

    observation_space = ObservationManager.get_observation_dim(observation, action_space)

    # reward config
    reward: RewardConfig = RewardConfig(
        progress_scale=0.2,
        contour_error_scale=0.04,
        velocity_alignment_scale=0.02,
        speed_tracking_scale=0.00,
        velocity_limit_penalty_scale=-0.02,
        orientation_penalty_scale=-0.002,
        ang_vel_penalty_scale=-0.0005,
        action_smoothness_scale=-0.0001,
    )
    
    # disturbances for robustness
    enable_disturbances: bool = False
    force_disturbance_scale: float = 0.05  # Scale of random force as fraction of weight
    torque_disturbance_scale: float = 0.02  # Scale of random torque

    # curriculum learning
    enable_curriculum: bool = False  # Enable curriculum learning for trajectories
    curriculum: CurriculumCfg = CurriculumCfg(
        # Good performance thresholds (advance difficulty)
        good_contour_error_threshold=0.10,
        good_progress_velocity_threshold=1.0,
        # Bad performance thresholds (demote difficulty)
        bad_contour_error_threshold=0.25,
        bad_progress_velocity_threshold=0.5,
        # Difficulty settings
        initial_difficulty=0.0,
        max_difficulty=1.0,
        # Random adjustment amounts
        max_advancement_increment=0.10,  # Random [0, 0.10] when good
        max_demotion_decrement=0.10,  # Random [0, 0.10] when bad
    )
    
    # Termination thresholds
    off_track_threshold: float = 2.0  # Max contour error before termination (meters)
    upside_down_threshold: float = 0.75  # Gravity z-component threshold (>0.5 means >60° tilt)
    min_height: float = 0.3  # Minimum height before ground crash termination (meters)
    max_velocity: float = 100.0  # Maximum velocity before runaway termination (m/s)

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01


class RacingQuadcopterEnv(DirectRLEnv):
    """Quadcopter racing environment with time-independent trajectory tracking.
    
    This environment trains a quadcopter to follow a racing trajectory with:
    - Time-independent progress tracking
    - Look-ahead trajectory points for anticipation
    - Velocity limits and disturbances for robustness
    - Temporal observation history for better control
    """
    
    cfg: RacingQuadcopterEnvCfg

    def __init__(self, cfg: RacingQuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Action tracking
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Get robot properties
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Initialize trajectory AFTER parent init (when we have device and num_envs)
        self._trajectory = RacingTrajectory(cfg.trajectory, self.device, self.num_envs)
        # Set environment origins now that terrain is created
        self._trajectory.set_environment_origins(self._terrain.env_origins)

        # Per-environment velocity limit
        self._velocity_limit = torch.ones(self.num_envs, device=self.device) * cfg.velocity_limit

        # Initialize managers
        self._reward_manager = RewardManager(
            cfg.reward,
            self.num_envs,
            self.device,
            self._trajectory,
        )
        self._observation_manager = ObservationManager(
            cfg.observation,
            self.num_envs,
            cfg.action_space,
            self.device,
            self._trajectory,
            self._velocity_limit,
        )
        self._disturbance_manager = DisturbanceManager(
            self.num_envs,
            self.device,
            self._robot_weight,
            cfg.force_disturbance_scale,
            cfg.torque_disturbance_scale,
            cfg.enable_disturbances,
            difficulty_dependent=cfg.enable_curriculum,  # Enable difficulty scaling if curriculum enabled
            difficulty_threshold=0.33,
            max_force_scale_at_max_difficulty=0.15,
            max_torque_scale_at_max_difficulty=0.06,
        )

        # Initialize curriculum learning manager
        self._curriculum_manager = None
        if cfg.enable_curriculum:
            self._curriculum_manager = TrajectoryCurriculumManager(
                self.num_envs,
                self.device,
                cfg.curriculum,
            )
            print("Curriculum learning enabled for trajectory difficulty progression")

        # Episode tracking for curriculum
        self._episode_contour_errors = torch.zeros(self.num_envs, device=self.device)
        self._episode_progress_velocities = torch.zeros(self.num_envs, device=self.device)
        self._episode_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)

        # Termination reason tracking (for logging)
        self._last_off_track = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_gate_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_upside_down = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_ground_crash = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_runaway = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Add handle for debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self) -> None:
        """Setup the scene with robot, terrain, and lighting."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        
        # Trajectory will be initialized in __init__ after this method completes
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions and update state before physics step."""
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

        # Update observation history
        self._observation_manager.update_history(self._robot, self._actions)
        
        # Track episode metrics for curriculum (if enabled)
        if self._curriculum_manager is not None:
            # Accumulate contour error and progress velocity
            contour_error = self._reward_manager.contour_error
            # Use proper progress velocity calculation (dot product with tangent)
            progress_velocity = self._trajectory.get_progress_velocity(self._robot.data.root_lin_vel_w)
            
            self._episode_contour_errors += contour_error
            self._episode_progress_velocities += progress_velocity
            self._episode_steps += 1

    def _apply_action(self) -> None:
        """Apply control forces and disturbances to robot."""
        # Sample disturbances
        force_disturbance, torque_disturbance = self._disturbance_manager.sample_disturbances()

        # Combine control forces with disturbances
        total_force = self._thrust + force_disturbance
        total_torque = self._moment + torque_disturbance
        self._robot.set_external_force_and_torque(total_force, total_torque, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        """Compute observations for the policy network."""
        # Pass tracking info from reward manager to observation manager
        self._observation_manager.set_tracking_info(
            self._reward_manager.closest_point,
            self._reward_manager.contour_error_vector,
        )
        obs = self._observation_manager.compute_observations(self._robot, self._actions)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for current state."""
        total_reward, _ = self._reward_manager.compute_rewards(
            self._robot,
            self._actions,
            self._previous_actions,
            self._velocity_limit,
            self.step_dt,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions.
        
        Returns:
            died: Boolean tensor indicating premature termination.
            time_out: Boolean tensor indicating episode timeout.
        """
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Check contour error (off-track)
        contour_error = self._reward_manager.contour_error
        off_track = contour_error > self.cfg.off_track_threshold
        
        # Check gate collisions
        check_gate_collision = False
        if check_gate_collision:
            gate_collision = self._trajectory.check_gate_collision(
                self._robot.data.root_pos_w,
                contour_error
            )
        else:
            gate_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Safety terminations
        # Upside down: gravity in body frame pointing up (z > 0.5 means >60° tilt)
        upside_down = self._robot.data.projected_gravity_b[:, 2] > self.cfg.upside_down_threshold
        
        # Ground collision: below minimum height
        ground_crash = self._robot.data.root_pos_w[:, 2] < self.cfg.min_height
        
        # Runaway velocity: exceeding maximum safe speed
        runaway = torch.norm(self._robot.data.root_lin_vel_w, dim=1) > self.cfg.max_velocity
        
        # Combine all death conditions (first match wins for categorization)
        died = off_track | gate_collision | upside_down | ground_crash | runaway
        
        # Store termination reasons for logging in _reset_idx
        self._last_off_track = off_track
        self._last_gate_collision = gate_collision
        self._last_upside_down = upside_down
        self._last_ground_crash = ground_crash
        self._last_runaway = runaway
        
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """Reset specified environments."""
        # print(f"[TRACE] _reset_idx START: num_envs={len(env_ids) if env_ids is not None else 'all'}")
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Gather logging statistics
        final_contour_error = self._reward_manager.contour_error[env_ids].mean()
        mean_contour_error = self._episode_contour_errors[env_ids] / torch.clamp(self._episode_steps[env_ids], min=1)
        mean_contour_error_value = mean_contour_error.mean()
        
        # Compute mean progress velocity for the episode
        mean_progress_velocity = self._episode_progress_velocities[env_ids] / torch.clamp(self._episode_steps[env_ids], min=1)
        mean_progress_velocity_value = mean_progress_velocity.mean()

        # Update curriculum based on episode performance
        if self._curriculum_manager is not None:
            # Compute mean episode metrics for curriculum
            
            # Update curriculum
            self._curriculum_manager.update(mean_contour_error, mean_progress_velocity, env_ids)
            
            # Get curriculum statistics for logging
            curriculum_stats = self._curriculum_manager.get_statistics()
            
            # Reset episode tracking
            self._episode_contour_errors[env_ids] = 0.0
            self._episode_progress_velocities[env_ids] = 0.0
            self._episode_steps[env_ids] = 0

        # Get episode reward statistics from reward manager
        extras = self._reward_manager.get_episode_stats(env_ids, self.max_episode_length_s)
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        # Add termination statistics with detailed breakdown
        terminated_mask = self.reset_terminated[env_ids]
        num_terminated = torch.count_nonzero(terminated_mask).item()
        
        # Count each termination reason
        off_track_count = torch.count_nonzero(self._last_off_track[env_ids] & terminated_mask).item()
        gate_collision_count = torch.count_nonzero(self._last_gate_collision[env_ids] & terminated_mask).item()
        upside_down_count = torch.count_nonzero(self._last_upside_down[env_ids] & terminated_mask).item()
        ground_crash_count = torch.count_nonzero(self._last_ground_crash[env_ids] & terminated_mask).item()
        runaway_count = torch.count_nonzero(self._last_runaway[env_ids] & terminated_mask).item()
        
        extras = {
            "Episode_Termination/died": num_terminated,
            "Episode_Termination/died_off_track": off_track_count,
            "Episode_Termination/died_gate_collision": gate_collision_count,
            "Episode_Termination/died_upside_down": upside_down_count,
            "Episode_Termination/died_ground_crash": ground_crash_count,
            "Episode_Termination/died_runaway": runaway_count,
            "Episode_Termination/time_out": torch.count_nonzero(self.reset_time_outs[env_ids]).item(),
            "Metrics/final_contour_error": final_contour_error.item(),
            "Metrics/mean_contour_error": mean_contour_error_value.item(),
            "Metrics/mean_progress_velocity": mean_progress_velocity_value.item(),
        }
        self.extras["log"].update(extras)
        
        # Add curriculum statistics
        if self._curriculum_manager is not None:
            self.extras["log"].update(curriculum_stats)

        # Reset robot
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Spread out resets to avoid training spikes
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Reset actions
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # Reset managers
        # Use curriculum-based reset if enabled
        if self._curriculum_manager is not None:
            difficulties = self._curriculum_manager.get_difficulty(env_ids)
            self._trajectory.reset_with_difficulty(env_ids, difficulties)
            # Reset disturbances with difficulty-dependent scaling
            self._disturbance_manager.reset_episode_disturbances(env_ids, self._curriculum_manager.difficulty_levels)
        else:
            self._trajectory.reset(env_ids)
            # Reset disturbances without difficulty scaling
            self._disturbance_manager.reset_episode_disturbances(env_ids, None)
            
        # Randomize velocity limit if enabled
        if self.cfg.randomize_velocity_limit:
            vl_min, vl_max = self.cfg.velocity_limit_range
            self._velocity_limit[env_ids] = torch.rand(len(env_ids), device=self.device) * (vl_max - vl_min) + vl_min
        else:
            self._velocity_limit[env_ids] = self.cfg.velocity_limit
            
        self._reward_manager.reset(env_ids)
        self._observation_manager.reset(env_ids)

        # Get starting pose from trajectory (already in world coordinates after offset)
        start_positions, start_orientations = self._trajectory.get_starting_pose(env_ids)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]

        root_pose = torch.cat([start_positions, start_orientations], dim=1)
        root_vel = torch.zeros(len(env_ids), 6, device=self.device)

        self._robot.write_root_pose_to_sim(root_pose, env_ids)
        self._robot.write_root_velocity_to_sim(root_vel, env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Note: Observation history is properly initialized to zero in reset().
        # The first observation will use zero history, which is correct since
        # there is no prior state after a reset.
        # print(f"[TRACE] _reset_idx END")

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Set debug visualization on or off.
        
        Args:
            debug_vis: Whether to enable debug visualization.
        """
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "closest_point_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.08, 0.08, 0.08)
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 0.0)  # Green
                # -- closest trajectory point
                marker_cfg.prim_path = "/Visuals/Command/closest_point"
                self.closest_point_visualizer = VisualizationMarkers(marker_cfg)
                
            if not hasattr(self, "lookahead_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (1.0, 0.0, 0.0)  # Red
                # -- lookahead points (will visualize first env only for clarity)
                marker_cfg.prim_path = "/Visuals/Command/lookahead_points"
                self.lookahead_visualizer = VisualizationMarkers(marker_cfg)
            
            # Create 4-sided gate visualization (top, bottom, left, right bars)
            gate_size = self.cfg.trajectory.gate_size
            bar_thickness = 0.015  # Thickness of gate bars
            
            if not hasattr(self, "gate_top_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (gate_size, bar_thickness, bar_thickness)  # Horizontal bar
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (1.0, 0.3, 0.0)  # Orange
                marker_cfg.prim_path = "/Visuals/Command/gates_top"
                self.gate_top_visualizer = VisualizationMarkers(marker_cfg)
            
            if not hasattr(self, "gate_bottom_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (gate_size, bar_thickness, bar_thickness)  # Horizontal bar
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (1.0, 0.3, 0.0)  # Orange
                marker_cfg.prim_path = "/Visuals/Command/gates_bottom"
                self.gate_bottom_visualizer = VisualizationMarkers(marker_cfg)
            
            if not hasattr(self, "gate_left_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (bar_thickness, bar_thickness, gate_size)  # Vertical bar
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (1.0, 0.3, 0.0)  # Orange
                marker_cfg.prim_path = "/Visuals/Command/gates_left"
                self.gate_left_visualizer = VisualizationMarkers(marker_cfg)
            
            if not hasattr(self, "gate_right_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (bar_thickness, bar_thickness, gate_size)  # Vertical bar
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (1.0, 0.3, 0.0)  # Orange
                marker_cfg.prim_path = "/Visuals/Command/gates_right"
                self.gate_right_visualizer = VisualizationMarkers(marker_cfg)
            
            # Create trajectory line visualizer using small spheres
            if not hasattr(self, "trajectory_line_visualizer"):
                from isaaclab.markers import SPHERE_MARKER_CFG
                marker_cfg = SPHERE_MARKER_CFG.copy()
                marker_cfg.markers["sphere"].radius = 0.01  # Small spheres
                marker_cfg.markers["sphere"].visual_material.diffuse_color = (0.3, 0.7, 1.0)  # Light blue
                marker_cfg.prim_path = "/Visuals/Command/trajectory_line"
                self.trajectory_line_visualizer = VisualizationMarkers(marker_cfg)
                
            # set their visibility to true
            self.closest_point_visualizer.set_visibility(False)
            self.lookahead_visualizer.set_visibility(False)
            self.gate_top_visualizer.set_visibility(True)
            self.gate_bottom_visualizer.set_visibility(True)
            self.gate_left_visualizer.set_visibility(True)
            self.gate_right_visualizer.set_visibility(True)
            self.trajectory_line_visualizer.set_visibility(False)
        else:
            if hasattr(self, "closest_point_visualizer"):
                self.closest_point_visualizer.set_visibility(False)
            if hasattr(self, "lookahead_visualizer"):
                self.lookahead_visualizer.set_visibility(False)
            if hasattr(self, "gate_top_visualizer"):
                self.gate_top_visualizer.set_visibility(False)
            if hasattr(self, "gate_bottom_visualizer"):
                self.gate_bottom_visualizer.set_visibility(False)
            if hasattr(self, "gate_left_visualizer"):
                self.gate_left_visualizer.set_visibility(False)
            if hasattr(self, "gate_right_visualizer"):
                self.gate_right_visualizer.set_visibility(False)
            if hasattr(self, "trajectory_line_visualizer"):
                self.trajectory_line_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        """Update debug visualization markers.
        
        Args:
            event: Simulation event triggering the callback.
        """
        # Show closest trajectory point from reward manager
        closest_point = self._reward_manager._closest_traj_point
        self.closest_point_visualizer.visualize(closest_point)
        
        # Show lookahead points for first few environments
        lookahead_points = self._trajectory.get_lookahead_points(self._robot.data.root_pos_w, self.cfg.lookahead_distances)
        vis_points = lookahead_points[:self.num_envs].reshape(-1, 3)
        self.lookahead_visualizer.visualize(vis_points)
        
        # Show virtual gates if enabled (4-sided gate visualization)
        if self._trajectory.gates_enabled and self._trajectory.num_gates_per_env is not None and self._trajectory.num_gates_per_env.sum() > 0:
            gate_positions, gate_orientations, num_gates = self._trajectory.get_all_gates_flattened()
            # Gate positions are already in world coordinates (offset applied during trajectory generation)
            gate_centers = gate_positions
            
            # Compute positions for 4 bars of each gate
            gate_size = self.cfg.trajectory.gate_size
            half_size = gate_size / 2.0
            
            # Transform local offsets to world coordinates using gate orientations
            from isaaclab.utils.math import quat_rotate
            
            # Gate body frame: x=right, y=tangent(through), z=up
            # Top bar: offset upward by half_size in local z (body-z = up)
            top_offset = torch.zeros(num_gates, 3, device=self.device)
            top_offset[:, 2] = half_size
            top_positions = gate_centers + quat_rotate(gate_orientations, top_offset)
            
            # Bottom bar: offset downward by half_size in local z
            bottom_offset = torch.zeros(num_gates, 3, device=self.device)
            bottom_offset[:, 2] = -half_size
            bottom_positions = gate_centers + quat_rotate(gate_orientations, bottom_offset)
            
            # Left bar: offset left by half_size in local x (body-x = right, so -x = left)
            left_offset = torch.zeros(num_gates, 3, device=self.device)
            left_offset[:, 0] = -half_size
            left_positions = gate_centers + quat_rotate(gate_orientations, left_offset)
            
            # Right bar: offset right by half_size in local x
            right_offset = torch.zeros(num_gates, 3, device=self.device)
            right_offset[:, 0] = half_size
            right_positions = gate_centers + quat_rotate(gate_orientations, right_offset)
            
            # Visualize each bar with appropriate orientation
            self.gate_top_visualizer.visualize(top_positions, gate_orientations)
            self.gate_bottom_visualizer.visualize(bottom_positions, gate_orientations)
            self.gate_left_visualizer.visualize(left_positions, gate_orientations)
            self.gate_right_visualizer.visualize(right_positions, gate_orientations)
        
        # Visualize trajectory lines using waypoints from all environments
        if self._trajectory.current_waypoints is not None:
            # Get all waypoints from all environments and flatten them
            # Subsample waypoints to reduce visual clutter (every 3rd waypoint)
            all_waypoints = []
            for env_id in range(self.num_envs):  # Limit to first 100 envs for performance
                waypoints = self._trajectory.current_waypoints[env_id]  # [num_waypoints, 3]
                # Subsample every Xth waypoint
                # subsampled = waypoints[::1]
                all_waypoints.append(waypoints)
            
            # Flatten all waypoints
            trajectory_points = torch.cat(all_waypoints, dim=0)
            self.trajectory_line_visualizer.visualize(trajectory_points)
