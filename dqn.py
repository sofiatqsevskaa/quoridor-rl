import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pygame
import matplotlib.pyplot as plt
from collections import deque
from quoridor_env import QuoridorEnv
from visualizer import QuoridorVisualizer
import os
import json


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)


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


def save_episode(states, ep, env):
    os.makedirs("saved_episodes", exist_ok=True)
    with open(f"saved_episodes/ep_{ep}.json", "w") as f:
        json.dump({
            "size": env.size,
            "num_walls": env.num_walls,
            "states": states
        }, f)


def train_dqn(episodes=5000, size=5, batch_size=64):
    env = QuoridorEnv(size=size, num_walls=3)

    state_dim = size * size * 2 + 2
    action_dim = 4 + size * size

    buffer = ReplayBuffer()
    policy = DQN(state_dim, action_dim)
    target = DQN(state_dim, action_dim)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    epsilon = 1.0

    episode_rewards = []
    episode_lengths = []

    for ep in range(1, episodes + 1):
        env.reset()
        state = env_to_state_vector(env)
        episode_states = [snapshot(env)]
        visit_count = {}
        step = 0
        total_reward = 0.0

        while not env.is_game_over() and step < 150:
            step += 1

            visit_count[env.white_pos] = visit_count.get(env.white_pos, 0) + 1

            prev_state = (
                env.white_pos,
                env.black_pos,
                set(env.h_walls),
                set(env.v_walls)
            )

            if random.random() < epsilon:
                a = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    a = policy(torch.tensor(
                        state).float().unsqueeze(0)).argmax().item()

            action = index_to_action(a, env)
            if action[0] == 'move':
                env.move_pawn((action[1], action[2]))
            else:
                env.place_wall(action[0][0], (action[1], action[2]))

            done = env.is_game_over() is not None

            opp = env.get_legal_moves()
            if opp and not done:
                o = random.choice(opp)
                if o[0] == 'move':
                    env.move_pawn((o[1], o[2]))
                else:
                    env.place_wall(o[0][0], (o[1], o[2]))

            episode_states.append(snapshot(env))
            next_state = env_to_state_vector(env)
            reward = compute_reward(env, prev_state, visit_count)
            total_reward += reward

            buffer.push(state, a, reward, next_state, done)
            state = next_state

            if len(buffer) >= batch_size:
                s, a_b, r, ns, d = buffer.sample(batch_size)
                q = policy(s).gather(1, a_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    qn = target(ns).max(1)[0]
                    qt = r + 0.99 * qn * (1 - d)
                loss = nn.MSELoss()(q, qt)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        episode_rewards.append(total_reward)
        episode_lengths.append(step)

        if ep < episodes - 1000:
            epsilon = max(epsilon * 0.996, 0.05)
        else:
            epsilon = 0.01

        if ep % 100 == 0:
            save_episode(episode_states, ep, env)
            print(
                f"episode {ep} reward={total_reward:.1f} steps={step} epsilon={epsilon:.3f}")

    torch.save(policy, "trained_dqn.pth")
    plot_training(episode_rewards, episode_lengths)


def plot_training(rewards, lengths):
    os.makedirs("plots", exist_ok=True)

    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward")
    plt.savefig("plots/training_reward"+plt.savefig("plots/training_reward_" +
                str(int(time.time())) + ".png")+".png")
    plt.close()

    plt.figure()
    plt.plot(lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length")
    plt.savefig("plots/episode_length.png")
    plt.close()


if __name__ == "__main__":
    train_dqn()
