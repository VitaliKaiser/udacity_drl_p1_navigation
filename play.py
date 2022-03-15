#!/usr/bin/env python3
from unityagents import UnityEnvironment
import numpy as np

import agents
import config as cfg
from environment import Environment

env_unity = UnityEnvironment(file_name=cfg.PATH_TO_BANANA, worker_id=1)
env = Environment(env_unity)

# Hyperparameter not really relevant for playing/inference.
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

agent.restore()

for i in range(10):
    state = env.reset(train_mode=False)

    while True:

        action = agent.act(state, train_mode=False)

        # Send the action to the environment.
        env_info = env.step(action)

        # Roll over the state to next time step.
        state = env_info.vector_observations
        i = i + 1
        if np.any(env_info.local_done):
            break
