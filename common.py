
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
    index = STATE_TO_INDEX.get(observation)
    if index is None:
        print(f"Unknown state: {observation}")
        return 0
    return int(round(policy[index]))

def fitness(env, policy, episodes=500):
    total_reward = 0
    episodes = 500
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = policy_action(policy, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        total_reward += reward
    return total_reward / episodes

def plot_strategy(policy, usable_ace=False):
    player_sums = list(range(12, 22))   # iba zmysluplné súčty
    dealer_cards = list(range(1, 11))
    action_matrix = np.zeros((len(player_sums), len(dealer_cards)))

    for i, ps in enumerate(player_sums):
        for j, dc in enumerate(dealer_cards):
            state = (ps, dc, usable_ace)

            action = policy_action(policy, state)
            action_matrix[i, j] = action

    title = f"Strategy Heatmap (Usable Ace: {usable_ace})"
    plt.figure(figsize=(10, 6))
    plt.imshow(action_matrix, cmap='coolwarm', aspect='auto', origin='lower')
    plt.colorbar(ticks=[0, 1], label="Action (0=Stick, 1=Hit)")
    plt.xticks(np.arange(len(dealer_cards)), dealer_cards)
    plt.yticks(np.arange(len(player_sums)), player_sums)
    plt.xlabel("Dealer's Visible Card")
    plt.ylabel("Player Sum")
    plt.title(title)
    plt.grid(False)
    plt.show()