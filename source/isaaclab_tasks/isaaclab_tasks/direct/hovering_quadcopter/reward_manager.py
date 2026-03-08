from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class RewardWeights:
    lin_vel = -0.2
    ang_vel = -0.05  # It learns to circle around when this is too low
    distance_to_goal = -1.0
    crashed = -100.0
    action_magnitude = -0.02
    oversaturation = -0.03
    action_smoothness = -0.02

    reward_scale = 0.2  # overall scaling factor

    def __post_init__(self):
        assert self.lin_vel <= 0.0
        assert self.ang_vel <= 0.0
        assert self.distance_to_goal <= 0.0
        assert self.crashed <= 0.0
        assert self.action_magnitude <= 0.0
        assert self.oversaturation <= 0.0
        assert self.action_smoothness <= 0.0


class RewardManager:
    def __init__(
            self,
            num_envs: int,
            device: str | torch.device,
            weights: RewardWeights | None = None,
    ):
        self.num_envs = num_envs
        self.device = device
        self.weights = weights if weights is not None else RewardWeights()

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "crashed",
                "action_magnitude",
                "oversaturation",
                "action_smoothness",
            ]
        }

    def compute_rewards(
            self,
            robot,
            desired_pos_w,
            actions,
            unnormalized_actions,
            prev_actions,
            step_dt: float,

        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        lin_vel = torch.sum(torch.square(robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(desired_pos_w - robot.data.root_pos_w, dim=1)
        distance_to_goal_scaled = torch.pow(distance_to_goal + 2.0, 3.0)
        crashed = torch.logical_or(robot.data.root_pos_w[:, 2] < 0.1, robot.data.root_pos_w[:, 2] > 2.0)

        components = {}

        p_lin_vel = lin_vel * self.weights.lin_vel * step_dt
        components["lin_vel"] = p_lin_vel

        p_ang_vel = ang_vel * self.weights.ang_vel * step_dt
        components["ang_vel"] = p_ang_vel

        p_distance_to_goal = distance_to_goal_scaled * self.weights.distance_to_goal * step_dt
        components["distance_to_goal"] = p_distance_to_goal

        p_crashed = crashed * self.weights.crashed
        components["crashed"] = p_crashed

        p_action_magnitude = self.weights.action_magnitude * actions.square().sum(dim=1)
        components["action_magnitude"] = p_action_magnitude

        action_clip = unnormalized_actions - actions
        p_oversaturation = self.weights.oversaturation * action_clip.square().sum(dim=1)
        components["oversaturation"] = p_oversaturation

        smoothness = actions - prev_actions
        p_smoothness = self.weights.action_smoothness * smoothness.square().sum(dim=1)
        components["action_smoothness"] = p_smoothness

        reward = torch.sum(torch.stack(list(components.values())), dim=0) * self.weights.reward_scale
        
        # Logging
        for key, value in components.items():
            self._episode_sums[key] += value
        
        return reward, components

    def get_episode_sums(self) -> dict[str, torch.Tensor]:
        ret = self._episode_sums.copy()
        ret['total'] = torch.stack(list(self._episode_sums.values())).sum(dim=0) * self.weights.reward_scale
        return ret