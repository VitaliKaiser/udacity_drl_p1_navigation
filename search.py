"""I am to lazy to search for hyperparameters, so let the computer do it."""

from collections import deque
from datetime import datetime
import os

import numpy as np
from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.schedulers.async_hyperband import ASHAScheduler
from unityagents import UnityEnvironment

import agents
import config as cfg
from environment import Environment
from training import episode

EARLY_END_NUM_EPISODES = 100
NUM_MAX_EPISODES = 700


def trainable(hpara):
    env_unity = UnityEnvironment(
        file_name=cfg.PATH_TO_BANANA,
        worker_id=os.getpid(),
        seed=datetime.now().microsecond,
        base_port=1,
        no_graphics=True,
    )
    env = Environment(env_unity)
    agent = agents.DQN(
        env.state_space_size,
        env.action_space_size,
        seed=datetime.now().microsecond,
        hpara=hpara,
    )

    # Track the mean score
    scores_window = deque(maxlen=EARLY_END_NUM_EPISODES)
    for i in range(NUM_MAX_EPISODES):

        score, _ = episode(env, agent)
        scores_window.append(score)

        # print the current score
        mean_score = np.mean(scores_window)

        yield {"mean_score": mean_score, "episodes": i}
    env.close()


bayesopt = BayesOptSearch(metric="mean_score", mode="max")
asha_scheduler = ASHAScheduler(
    time_attr="episodes",
    metric="mean_score",
    mode="max",
    max_t=700,
    grace_period=10,
    reduction_factor=3,
    brackets=1,
)
analysis = tune.run(
    trainable,
    config=agents.DQN.hyperparamter_space(),
    search_alg=bayesopt,
    scheduler=asha_scheduler,
    resources_per_trial={"cpu": 2},
    num_samples=100,
)

print(
    "best config: ",
    analysis.get_best_config(
        metric="mean_score",
        mode="max",
    ),
)
