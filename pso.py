import numpy as np
import gymnasium as gym
from pyswarm import pso
from matplotlib import pyplot as plt

from common import fitness, STATES, plot_strategy

# Inicializuj prostredie
env = gym.make("Blackjack-v1", sab=True)


def evaluate(policy_vector):
    return -fitness(env, policy_vector)

# Spusti PSO optimalizáciu
lb = [0] * len(STATES)
ub = [1] * len(STATES)

best_policy_vector, best_score = pso(evaluate, lb, ub, swarmsize=100, maxiter=100)

# Preveď na binárnu politiku
final_policy = (best_policy_vector > 0.5).astype(int)



# Výpis
print(f"\nNajlepšie skóre: {-best_score:.3f}")

# Vykreslenie politiky
plot_strategy(final_policy, usable_ace=False)
plot_strategy(final_policy, usable_ace=True)
