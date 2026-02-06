import json
import os
import matplotlib.pyplot as plt
from collections import deque


def env_to_state_vector(env):
    N = env.size
    state = [0] * (N * N * 2 + 2)
    w_r, w_c = env.white_pos
    b_r, b_c = env.black_pos
    state[w_r * N + w_c] = 1
    state[N * N + b_r * N + b_c] = 1
    state[-2] = env.white_walls
    state[-1] = env.black_walls
    return state


def index_to_action(index, env):
    N = env.size
    if index < 4:
        dr_dc = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        r, c = env.white_pos
        dr, dc = dr_dc[index]
        return ('move', r + dr, c + dc)
    wall_index = index - 4
    r = wall_index // N
    c = wall_index % N
    return ('h_wall', r, c)


def action_to_index(action, env):
    N = env.size
    if action[0] == 'move':
        dr_dc = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        r, c = env.white_pos
        target_r, target_c = action[1], action[2]
        dr, dc = target_r - r, target_c - c
        try:
            return dr_dc.index((dr, dc))
        except ValueError:
            return 0
    else:
        wall_r, wall_c = action[1], action[2]
        return 4 + wall_r * N + wall_c


def blocking_walls_ahead(env):
    r, c = env.white_pos
    count = 0
    if (r - 1, c) in env.h_walls:
        count += 1
    if (r - 1, c - 1) in env.h_walls:
        count += 1
    return count


def compute_reward(env, prev_state, visit_count):
    prev_white_pos, prev_black_pos, prev_h_walls, prev_v_walls = prev_state
    progress_reward = (prev_white_pos[0] - env.white_pos[0]) * 10.0
    wall_penalty = blocking_walls_ahead(env) * -2.0
    placed_wall = len(env.h_walls) + \
        len(env.v_walls) > len(prev_h_walls) + len(prev_v_walls)
    block_opponent_reward = 0.0
    if placed_wall:
        prev_black_dist = (env.size - 1) - prev_black_pos[0]
        curr_black_dist = (env.size - 1) - env.black_pos[0]
        if curr_black_dist > prev_black_dist:
            block_opponent_reward = 3.0
    repeat_penalty = 0.0
    if visit_count.get(env.white_pos, 0) > 1:
        repeat_penalty = -1.5 * (visit_count[env.white_pos] - 1)
    win_reward = 0.0
    if env.is_game_over() == "white":
        win_reward = 500.0
    elif env.is_game_over() == "black":
        win_reward = -500.0
    return progress_reward + wall_penalty + block_opponent_reward + repeat_penalty + win_reward


def snapshot(env):
    return {
        "white_pos": list(env.white_pos),
        "black_pos": list(env.black_pos),
        "h_walls": [list(w) for w in env.h_walls],
        "v_walls": [list(w) for w in env.v_walls],
        "white_walls": env.white_walls,
        "black_walls": env.black_walls,
        "turn": env.turn
    }


def save_episode(states, ep, env, model_type="actor_critic"):
    os.makedirs(f"saved_episodes_{model_type}", exist_ok=True)
    with open(f"saved_episodes_{model_type}/ep_{ep}.json", "w") as f:
        json.dump({
            "size": env.size,
            "num_walls": env.num_walls,
            "states": states
        }, f)


def plot_training(rewards, lengths, model_type="actor_critic"):
    os.makedirs("plots", exist_ok=True)
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"{model_type.upper()} Training Reward")
    plt.savefig(f"plots/{model_type}_training_reward.png")
    plt.close()
    plt.figure()
    plt.plot(lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title(f"{model_type.upper()} Episode Length")
    plt.savefig(f"plots/{model_type}_episode_length.png")
    plt.close()
