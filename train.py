#!/usr/bin/env python3
from unityagents import UnityEnvironment

import agents
import config as cfg
from environment import Environment
from training import train

env_unity = UnityEnvironment(
    file_name=cfg.PATH_TO_BANANA, worker_id=1, no_graphics=True
)
env = Environment(env_unity)


hpara = {
    "eps_start": 1.0,
    "eps_range": 0.99,
    "eps_decay": 0.995,
    "lr": 5e-4,
    "tau": 1e-3,
    "gamma": 0.99,
    "target_update_interval": 4,
}
agent = agents.DQN(env.state_space_size, env.action_space_size, seed=123, hpara=hpara)

train(
    num_episodes=1800,
    early_end_sliding_score=13.0,
    early_end_num_episodes=100,
    env=env,
    agent=agent,
)

env.close()
