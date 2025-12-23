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

    def __init__(self, env: "QuadcopterEnvCopy", window_name: str = "IsaacLab"):
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
class QuadcopterEnvCfgCopy(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    action_space = 4
    # observation_space calculation:
    # base: lin_vel(3) + ang_vel(3) + gravity(3) + velocity_limit(1) = 10
    # lookahead points: 6 * 3 = 18
    # history: (lin_vel(3) + ang_vel(3) + actions(4)) * history_length
    history_length = 3  # Number of past timesteps to include
    observation_space = 10 + 18 + (3 + 3 + 4) * history_length  # 28 + 30 = 58
    state_space = 0
    debug_vis = True

    # trajectory config
    trajectory: TrajectoryConfig = TrajectoryConfig(
        trajectory_type="library",
        radius=2.0,
        height=1.5,
        num_waypoints=100,
        desired_speed=2.0,
        lookahead_distances=(0.3, 0.6, 1.0, 1.3, 1.6, 2.0),
        velocity_limit=3.0,
        randomize_velocity_limit=True,
        velocity_limit_range=(1.5, 4.0),
        use_trajectory_library=True,
        track_seed=42,
    )

    # observation config
    observation: ObservationConfig = ObservationConfig(
        history_length=3,
        lookahead_distances=(0.3, 0.6, 1.0, 1.3, 1.6, 2.0),
    )

    # reward config
    reward: RewardConfig = RewardConfig(
        progress_scale=10.0,
        contour_error_scale=-2.0,
        velocity_alignment_scale=1.0,
        speed_tracking_scale=0.5,
        velocity_limit_penalty_scale=-2.0,
        orientation_penalty_scale=-0.5,
        ang_vel_penalty_scale=-0.01,
        action_smoothness_scale=-0.001,
    )
    
    # disturbances for robustness
    enable_disturbances: bool = True
    force_disturbance_scale: float = 0.05  # Scale of random force as fraction of weight
    torque_disturbance_scale: float = 0.02  # Scale of random torque

    # curriculum learning
    enable_curriculum: bool = True  # Enable curriculum learning for trajectories
    curriculum: CurriculumCfg = CurriculumCfg(
        contour_error_threshold=0.15,
        progress_velocity_threshold=0.8,
        success_streak_required=50,
        initial_difficulty=0.0,
        max_difficulty=1.0,
        use_continuous_progression=True,
        base_increment=0.05,
        max_increment=0.15,
        base_decrement=0.05,
        max_decrement=0.15,
        enable_demotion=True,
        failure_streak_threshold=100,
    )

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


class QuadcopterEnvCopy(DirectRLEnv):
    """Quadcopter racing environment with time-independent trajectory tracking.
    
    This environment trains a quadcopter to follow a racing trajectory with:
    - Time-independent progress tracking
    - Look-ahead trajectory points for anticipation
    - Velocity limits and disturbances for robustness
    - Temporal observation history for better control
    """
    
    cfg: QuadcopterEnvCfgCopy

    def __init__(self, cfg: QuadcopterEnvCfgCopy, render_mode: str | None = None, **kwargs):
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

        # Initialize trajectory
        self._trajectory = RacingTrajectory(cfg.trajectory, self.device, self.num_envs)

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
            self.cfg.action_space,
            self.device,
            self._trajectory,
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
            progress_velocity = torch.norm(self._robot.data.root_lin_vel_w[:, :2], dim=1) * \
                               torch.sum(self._robot.data.root_lin_vel_w * self._trajectory.tangents[self._trajectory.closest_idx][:, :3], dim=1).sign()
            
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
        obs = self._observation_manager.compute_observations(self._robot, self._actions)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for current state."""
        total_reward, _ = self._reward_manager.compute_rewards(
            self._robot,
            self._actions,
            self._previous_actions,
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
        
        # Check altitude boundaries
        altitude_died = torch.logical_or(
            self._robot.data.root_pos_w[:, 2] < 0.1,
            self._robot.data.root_pos_w[:, 2] > 2.0
        )
        
        # Check gate collisions
        contour_error = self._reward_manager.contour_error
        gate_collision = self._trajectory.check_gate_collision(
            self._robot.data.root_pos_w,
            contour_error
        )
        
        # Combine all death conditions
        died = torch.logical_or(altitude_died, gate_collision)
        
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """Reset specified environments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Gather logging statistics
        final_contour_error = self._reward_manager.contour_error[env_ids].mean()
        final_progress = self._trajectory.progress[env_ids].mean()

        # Update curriculum based on episode performance
        if self._curriculum_manager is not None:
            # Compute mean episode metrics for curriculum
            mean_contour_error = self._episode_contour_errors[env_ids] / torch.clamp(self._episode_steps[env_ids], min=1)
            mean_progress_velocity = self._episode_progress_velocities[env_ids] / torch.clamp(self._episode_steps[env_ids], min=1)
            
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

        # Add termination statistics
        extras = {
            "Episode_Termination/died": torch.count_nonzero(self.reset_terminated[env_ids]).item(),
            "Episode_Termination/time_out": torch.count_nonzero(self.reset_time_outs[env_ids]).item(),
            "Metrics/final_contour_error": final_contour_error.item(),
            "Metrics/final_progress": final_progress.item(),
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
            
        self._reward_manager.reset(env_ids)
        self._observation_manager.reset(env_ids)

        # Get starting pose from trajectory
        start_positions, start_orientations = self._trajectory.get_starting_pose(env_ids)
        start_positions += self._terrain.env_origins[env_ids]

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]

        root_pose = torch.cat([start_positions, start_orientations], dim=1)
        root_vel = torch.zeros(len(env_ids), 6, device=self.device)

        self._robot.write_root_pose_to_sim(root_pose, env_ids)
        self._robot.write_root_velocity_to_sim(root_vel, env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

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
                # -- closest trajectory point
                marker_cfg.prim_path = "/Visuals/Command/closest_point"
                self.closest_point_visualizer = VisualizationMarkers(marker_cfg)
                
            if not hasattr(self, "lookahead_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- lookahead points (will visualize first env only for clarity)
                marker_cfg.prim_path = "/Visuals/Command/lookahead_points"
                self.lookahead_visualizer = VisualizationMarkers(marker_cfg)
            
            # Create 4-sided gate visualization (top, bottom, left, right bars)
            gate_size = self.cfg.trajectory.gate_size
            bar_thickness = 0.08  # Thickness of gate bars
            
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
                
            # set their visibility to true
            self.closest_point_visualizer.set_visibility(True)
            self.lookahead_visualizer.set_visibility(True)
            self.gate_top_visualizer.set_visibility(True)
            self.gate_bottom_visualizer.set_visibility(True)
            self.gate_left_visualizer.set_visibility(True)
            self.gate_right_visualizer.set_visibility(True)
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

    def _debug_vis_callback(self, event) -> None:
        """Update debug visualization markers.
        
        Args:
            event: Simulation event triggering the callback.
        """
        # Show closest trajectory point from reward manager
        closest_point = self._reward_manager._closest_traj_point
        self.closest_point_visualizer.visualize(closest_point)
        
        # Show lookahead points for first few environments
        lookahead_points = self._trajectory.get_lookahead_points(self._robot.data.root_pos_w)
        vis_points = lookahead_points[:self.num_envs].reshape(-1, 3)
        self.lookahead_visualizer.visualize(vis_points)
        
        # Show virtual gates if enabled (4-sided gate visualization)
        if self._trajectory.gates_enabled and self._trajectory.num_gates > 0:
            gate_positions, gate_orientations, num_gates = self._trajectory.get_gate_info()
            # Add environment origin offset for first environment
            gate_centers = gate_positions + self._terrain.env_origins[0]
            
            # Compute positions for 4 bars of each gate
            gate_size = self.cfg.trajectory.gate_size
            half_size = gate_size / 2.0
            
            # Transform local offsets to world coordinates using gate orientations
            from isaaclab.utils.math import quat_rotate
            
            # Top bar: offset upward by half_size in local z
            top_offset = torch.zeros(num_gates, 3, device=self.device)
            top_offset[:, 2] = half_size
            top_positions = gate_centers + quat_rotate(gate_orientations, top_offset)
            
            # Bottom bar: offset downward by half_size in local z
            bottom_offset = torch.zeros(num_gates, 3, device=self.device)
            bottom_offset[:, 2] = -half_size
            bottom_positions = gate_centers + quat_rotate(gate_orientations, bottom_offset)
            
            # Left bar: offset left by half_size in local y
            left_offset = torch.zeros(num_gates, 3, device=self.device)
            left_offset[:, 1] = -half_size
            left_positions = gate_centers + quat_rotate(gate_orientations, left_offset)
            
            # Right bar: offset right by half_size in local y
            right_offset = torch.zeros(num_gates, 3, device=self.device)
            right_offset[:, 1] = half_size
            right_positions = gate_centers + quat_rotate(gate_orientations, right_offset)
            
            # Visualize each bar with appropriate orientation
            self.gate_top_visualizer.visualize(top_positions, gate_orientations)
            self.gate_bottom_visualizer.visualize(bottom_positions, gate_orientations)
            self.gate_left_visualizer.visualize(left_positions, gate_orientations)
            self.gate_right_visualizer.visualize(right_positions, gate_orientations)