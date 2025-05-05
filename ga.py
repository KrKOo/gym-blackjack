import numpy as np
import gymnasium as gym
import pygad
from matplotlib import pyplot as plt

from common import fitness, STATES, plot_strategy

# Vytvorenie prostredia Blackjack
env = gym.make("Blackjack-v1")

chromosome_length = len(STATES)

def fitness_func(ga_instance, solution, solution_idx):
    return fitness(env, solution, episodes=5000 )

gen = 0
def on_generation(ga_instance):
    global gen
    gen += 1
    if gen % 10 == 0:
        print(f"Generation: {gen}")
        best_solution, best_fitness, _ = ga_instance.best_solution()
        print(f"Fitness: {best_fitness:.2f}")
        plot_strategy(best_solution, filename=f"data/strategy_GA_{gen}.png")
        ga_instance.plot_fitness(filename=f"data/fitness_GA_{gen}.png")


# GA konfigurácia
ga_instance = pygad.GA(
    num_generations=300,
    num_parents_mating=30,
    fitness_func=fitness_func,
    initial_population=np.random.randint(0, 2, size=(50, chromosome_length)),
    sol_per_pop=50,
    keep_parents=25,
    num_genes=chromosome_length,
    gene_type=int,
    init_range_low=0,
    init_range_high=2,
    mutation_percent_genes=50,
    mutation_type="inversion",
    crossover_type="two_points",
    parent_selection_type="tournament",
    on_generation=on_generation
)

ga_instance.run()

solution, solution_fitness, _ = ga_instance.best_solution()
# plot_strategy(solution)
print(f"Najlepšia stratégia má priemernú odmenu: {solution_fitness:.2f}")