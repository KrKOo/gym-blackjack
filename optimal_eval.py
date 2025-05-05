from common import OPTIMAL_POLICY, fitness, plot_strategy

import numpy as np
import gymnasium as gym


env = gym.make("Blackjack-v1", sab=True)

zero_policy = np.zeros(len(OPTIMAL_POLICY), dtype=int)

res = fitness(
	env,
	policy=OPTIMAL_POLICY,
	episodes=500
)

plot_strategy(OPTIMAL_POLICY)
print(res)