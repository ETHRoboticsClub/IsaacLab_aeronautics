# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

_entry_point = f"{__name__}.hovering_quadcopter_env:HoveringQuadcopterEnv"
_cfg_module = f"{__name__}.hovering_quadcopter_env_cfg"
_agent_kwargs = {
    "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
    "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
}

# Training
gym.register(
    id="Isaac-Hovering-Quadcopter-Direct-v0",
    entry_point=_entry_point,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{_cfg_module}:HoveringQuadcopterEnvCfg", **_agent_kwargs},
)


gym.register(
    id="Isaac-Hovering-Quadcopter-Direct-Play-Train-v0",
    entry_point=_entry_point,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{_cfg_module}:HoveringQuadcopterEnvCfg_PLAY_TRAIN", **_agent_kwargs},
)


gym.register(
    id="Isaac-Hovering-Quadcopter-Direct-Play-Ideal-v0",
    entry_point=_entry_point,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{_cfg_module}:HoveringQuadcopterEnvCfg_PLAY_IDEAL", **_agent_kwargs},
)


gym.register(
    id="Isaac-Hovering-Quadcopter-Direct-Play-Easy-v0",
    entry_point=_entry_point,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{_cfg_module}:HoveringQuadcopterEnvCfg_PLAY_EASY", **_agent_kwargs},
)


gym.register(
    id="Isaac-Hovering-Quadcopter-Direct-Play-Hard-v0",
    entry_point=_entry_point,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{_cfg_module}:HoveringQuadcopterEnvCfg_PLAY_HARD", **_agent_kwargs},
)