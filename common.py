
import numpy as np
from matplotlib import pyplot as plt

player_sum_range = range(4, 22)
dealer_card_range = range(1, 11)
usable_ace_range = [False, True]

# 18 * 10 * 2 = 360 states
STATES = [(ps, dc, ua) for ps in player_sum_range for dc in dealer_card_range for ua in usable_ace_range]
STATE_TO_INDEX = {s: i for i, s in enumerate(STATES)}

OPTIMAL_POLICY = np.zeros(len(STATES), dtype=int)

# 0 = stick, 1 = hit
for i, (player, dealer, usable_ace) in enumerate(STATES):
    if usable_ace:
        if player >= 19:
            OPTIMAL_POLICY[i] = 0
        elif player == 18:
            OPTIMAL_POLICY[i] = 1 if dealer in [1, 9, 10] else 0
        else:
            OPTIMAL_POLICY[i] = 1
    else:
        if player >= 17:
            OPTIMAL_POLICY[i] = 0
        elif 13 <= player <= 16:
            OPTIMAL_POLICY[i] = 0 if 2 <= dealer <= 6 else 1
        elif player == 12:
            OPTIMAL_POLICY[i] = 0 if 4 <= dealer <= 6 else 1
        else:
            OPTIMAL_POLICY[i] = 1
            
def policy_action(policy, observation):
    obs = (min(observation[0], 21), observation[1], observation[2])
    index = STATE_TO_INDEX.get(obs)
    if index is None:
        print(f"Unknown state: {observation}")
        return 0
    return int(round(policy[index]))

def fitness(env, policy, episodes=500, runs=1):
    total_reward = 0
    episodes = 500
    results = []

    for _ in range(runs):
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action = policy_action(policy, obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            total_reward += reward
        results.append(total_reward / episodes)
    
    return np.mean(results)

def plot_strategy(policy, filename=None):
    player_sums = list(range(4, 22))   # meaningful player totals
    dealer_cards = list(range(1, 11))
    usable_ace_options = [True, False]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True,
                             gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.1})

    # Create a separate axis for the colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]

    for idx, usable_ace in enumerate(usable_ace_options):
        action_matrix = np.zeros((len(player_sums), len(dealer_cards)))

        for i, ps in enumerate(player_sums):
            for j, dc in enumerate(dealer_cards):
                state = (ps, dc, usable_ace)
                action = policy_action(policy, state)
                action_matrix[i, j] = action

        ax = axes[idx]
        im = ax.imshow(action_matrix, cmap='coolwarm', aspect='auto', origin='lower', vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(dealer_cards)))
        ax.set_xticklabels(dealer_cards)
        ax.set_yticks(np.arange(len(player_sums)))
        ax.set_yticklabels(player_sums)
        ax.set_xlabel("Dealer's Visible Card")
        if idx == 0:
            ax.set_ylabel("Player Sum")
        ax.set_title(f"Usable Ace: {usable_ace}")

    fig.suptitle("Strategy Heatmap (0=Stick, 1=Hit)", fontsize=16)
    fig.colorbar(im, cax=cbar_ax, ticks=[0, 1], label='Action')
    plt.subplots_adjust(right=0.9)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()