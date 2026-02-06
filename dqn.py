import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from quoridor_env import QuoridorEnv
from helper_functions import *
import os


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


if __name__ == "__main__":
    train_dqn(100000)
