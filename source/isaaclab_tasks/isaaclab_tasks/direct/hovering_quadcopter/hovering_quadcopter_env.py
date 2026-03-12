# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from .hovering_quadcopter_env_cfg import HoveringQuadcopterEnvCfg, DRONE_MODEL
from .reward_manager import RewardManager, RewardWeights


class HoveringQuadcopterEnv(DirectRLEnv):
    cfg: HoveringQuadcopterEnvCfg

    def __init__(self, cfg: HoveringQuadcopterEnvCfg,
                 render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._reward_manager = RewardManager(self.num_envs, self.device)

        # Eval metrics
        if self.cfg.eval_mode:
            torch.manual_seed(42)
            self._mean_distances = None

        # Logging
        self._episode_sums = self._reward_manager.get_episode_sums()

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Persistent disturbance forces/torques (set at reset, applied every step alongside thrust)
        self._disturbance_force = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._disturbance_torque = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)


        # Get specific body indices
        body_name = "body" if DRONE_MODEL == "crazyflie" else "base_link"
        self._body_id = self._robot.find_bodies(body_name)[0] # 'body' for crazyflie, 'base_link' for our drone
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # DEBUG: inspect what got spawned
        '''import omni.usd
        stage = omni.usd.get_context().get_stage()
        print("=" * 60)
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if "envs" in path and ("Robot" in path or "env_" in path):
                # Only print top-level env prims, not deep children
                depth = path.count("/")
                if depth <= 5:
                    print(f"  {path}  ({prim.GetTypeName()})")
        print("=" * 60)'''

        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _change_goal_pos(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        pos_offset = torch.zeros(len(env_ids), 3, device=self.device)
        pos_offset[:, 0].uniform_(*self.cfg.goal_offset_range_x)
        pos_offset[:, 1].uniform_(*self.cfg.goal_offset_range_y)
        pos_offset[:, 2].uniform_(*self.cfg.goal_offset_range_z)
        self._desired_pos_w[env_ids] = self._robot.data.root_pos_w[env_ids] + pos_offset
        self._desired_pos_w[env_ids, 2] = torch.clamp(self._desired_pos_w[env_ids, 2], min=0.5, max=1.5)

    def _pre_physics_step(self, actions: torch.Tensor):
        if self.cfg.eval_mode:
            rand_vals = torch.rand(self.num_envs, device=self.device)
            env_ids = torch.nonzero(rand_vals < self.cfg.goal_swap_prob_step, as_tuple=False).squeeze(-1)
            self._change_goal_pos(env_ids)

        self._unnormalized_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(
            self._thrust + self._disturbance_force,
            self._moment + self._disturbance_torque,
            body_ids=self._body_id,
        )

    def _get_observations(self) -> dict:
        if self.cfg.eval_mode:
            dist = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
            if self._mean_distances is None:
                self._mean_distances = dist.mean().unsqueeze(0)
            else:
                self._mean_distances = torch.cat([self._mean_distances, dist.mean().unsqueeze(0)])

        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        reward, components = self._reward_manager.compute_rewards(
            self._robot,
            self._desired_pos_w,
            self._actions,
            self._unnormalized_actions,
            self._prev_actions if hasattr(self, "_prev_actions") else self._actions,
            self.step_dt
        )
        self._prev_actions = self._actions.clone()
        # Logging
        self._episode_sums = self._reward_manager.get_episode_sums()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        # Eval metrics
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        if self.cfg.eval_mode:
            extras["Metrics/mean_distance_to_goal"] = \
                self._mean_distances.mean().item() if self._mean_distances is not None else 0.0
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        pos_offset = torch.zeros(len(env_ids), 3, device=self.device)
        pos_offset[:, 0].uniform_(-0.5, 0.5)   # X: ±0.5 m
        pos_offset[:, 1].uniform_(-0.5, 0.5)   # Y: ±0.5 m
        pos_offset[:, 2].uniform_(-0.2, 0.2)   # Z: ±0.2 m
        default_root_state[:, :3] += pos_offset
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._change_goal_pos(env_ids)

        # Sample persistent disturbance for the episode (summed with thrust in _apply_action)
        self._disturbance_force[env_ids, 0, :].uniform_(*self.cfg.disturbance_force_range)
        self._disturbance_torque[env_ids, 0, :].uniform_(*self.cfg.disturbance_torque_range)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
