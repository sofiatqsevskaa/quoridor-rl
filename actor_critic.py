import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import random

from collections import deque


def count_shortest_paths(env, start_pos, goal_row):
    visited = set()
    queue = deque([(start_pos, 0)])
    min_steps = None
    path_count = 0

    while queue:
        pos, steps = queue.popleft()
        if pos in visited:
            continue
        visited.add(pos)

        r, c = pos
        if r == goal_row:
            if min_steps is None:
                min_steps = steps
            if steps == min_steps:
                path_count += 1
            continue

        for nr, nc in env.get_legal_moves_from(pos):
            if (nr, nc) not in visited:
                queue.append(((nr, nc), steps + 1))

    return path_count, min_steps


class ActorCriticNetwork(nn.Module):

    def __init__(self, board_size=9, hidden_size=256):
        super(ActorCriticNetwork, self).__init__()
        self.board_size = board_size

        input_channels = 5
        input_size = input_channels * board_size * board_size + 2

        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_size // 2 + 3, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )

    def forward(self, state_tensor, action_masks=None):
        shared_features = self.shared(state_tensor)
        state_value = self.critic(shared_features)
        return shared_features, state_value

    def get_action_value(self, shared_features, action_tensor):
        combined = torch.cat([shared_features, action_tensor], dim=-1)
        return self.actor(combined)


class QuoridorActorCritic:

    def __init__(self, board_size=9, learning_rate=0.0003, gamma=0.99,
                 entropy_coef=0.01, value_coef=0.5):
        self.board_size = board_size
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.network = ActorCriticNetwork(board_size).to(self.device)
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=learning_rate)

        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.prev_positions = set()

    def state_to_tensor(self, env, board_size=9):
        state = np.zeros((5, board_size, board_size), dtype=np.float32)
        state[0, env.white_pos[0], env.white_pos[1]] = 1.0
        state[1, env.black_pos[0], env.black_pos[1]] = 1.0
        for r, c in env.h_walls:
            if r < board_size - 1 and c < board_size - 1:
                state[2, r, c] = 1.0
                state[2, r, c + 1] = 1.0
        for r, c in env.v_walls:
            if r < board_size - 1 and c < board_size - 1:
                state[3, r, c] = 1.0
                state[3, r + 1, c] = 1.0
        state[4, :, :] = float(env.turn)
        flat_state = state.flatten()
        wall_info = np.array(
            [env.white_walls / 10.0, env.black_walls / 10.0], dtype=np.float32)
        return torch.FloatTensor(np.concatenate([flat_state, wall_info]))

    def action_to_tensor(self, action, board_size=9):
        if action[0] == 'move':
            return torch.FloatTensor([0.0, action[1]/board_size, action[2]/board_size])
        elif action[0] == 'h_wall':
            return torch.FloatTensor([1.0, action[1]/board_size, action[2]/board_size])
        else:
            return torch.FloatTensor([2.0, action[1]/board_size, action[2]/board_size])

    def select_action(self, env, epsilon=0.1):
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            return None, None, None

        state_tensor = self.state_to_tensor(env).unsqueeze(0).to(self.device)

        with torch.no_grad():
            shared_features, value = self.network(state_tensor)

        action_tensors = torch.stack(
            [self.action_to_tensor(a).to(self.device) for a in legal_actions])
        combined = torch.cat([shared_features.repeat(
            len(legal_actions), 1), action_tensors], dim=1)
        logits = self.network.actor(combined).squeeze(-1)
        probs = F.softmax(logits, dim=0)

        if random.random() < epsilon:
            move_idxs = [i for i, a in enumerate(
                legal_actions) if a[0] == 'move']
            wall_idxs = [i for i, a in enumerate(
                legal_actions) if a[0] != 'move']

            if move_idxs and wall_idxs:
                idx = random.choice(move_idxs) if random.random(
                ) < 0.5 else random.choice(wall_idxs)
            elif move_idxs:
                idx = random.choice(move_idxs)
            else:
                idx = random.choice(wall_idxs)
        else:
            dist = Categorical(probs)
            idx = dist.sample().item()

        action = legal_actions[idx]
        log_prob = torch.log(probs[idx] + 1e-10)
        return action, log_prob, value.item()

    def train_step(self, states, actions, returns, advantages):
        states_tensor = torch.stack(states).to(self.device)
        action_tensors = torch.stack(
            [self.action_to_tensor(a).to(self.device) for a in actions])
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        advantages_tensor = (
            advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        self.optimizer.zero_grad()
        shared_features, values = self.network(states_tensor)
        values = values.squeeze(-1)

        combined = torch.cat([shared_features, action_tensors], dim=1)
        logits = self.network.actor(combined).squeeze(-1)
        probs = F.softmax(logits, dim=0)
        log_probs = torch.log(probs + 1e-10)

        policy_loss = -(log_probs * advantages_tensor).mean()
        value_loss = F.mse_loss(values, returns_tensor)
        entropy_loss = -(probs * log_probs).sum(dim=-1).mean()

        loss = policy_loss + self.value_coef * \
            value_loss - self.entropy_coef * entropy_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        self.losses.append(loss.item())
        return loss.item()

    def compute_returns(self, rewards, values, next_value, dones, gamma=0.99, lam=0.95):
        returns = []
        advantages = []
        gae = 0
        values = values + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * \
                values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        return returns, advantages

    def wall_block_score(self, env, pos, goal_row, max_dist=4):
        r, c = pos
        direction = -1 if goal_row < r else 1
        score = 0.0

        for d in range(1, max_dist + 1):
            nr = r + d * direction
            if nr < 0 or nr >= env.size:
                break

            if env.is_wall_between(r + (d - 1) * direction, c, nr, c):
                score += 1.0 / d
                break

        return score

    def shaped_reward(self, env, action, prev_white_pos, prev_black_pos, player_turn):
        reward = -0.01

        pos = env.white_pos if player_turn == 0 else env.black_pos
        prev_pos = prev_white_pos if player_turn == 0 else prev_black_pos
        opp_pos = env.black_pos if player_turn == 0 else env.white_pos

        goal_row = self.board_size - 1 if player_turn == 0 else 0
        opp_goal_row = 0 if player_turn == 0 else self.board_size - 1

        winner = env.is_game_over()
        if winner == "white":
            return 10.0 if player_turn == 0 else -10.0
        if winner == "black":
            return 10.0 if player_turn == 1 else -10.0

        self_dist_prev = abs(goal_row - prev_pos[0])
        self_dist_curr = abs(goal_row - pos[0])

        if action[0] == 'move':
            # reward for progress
            progress = self_dist_prev - self_dist_curr
            reward += 0.6 * progress

            # reward for path diversity
            paths, _ = count_shortest_paths(env, tuple(pos), goal_row)
            reward += 0.05 * min(paths, 5)  # cap contribution

            if tuple(pos) in self.prev_positions:
                reward -= 0.03
            self.prev_positions.add(tuple(pos))

        elif action[0] in ['h_wall', 'v_wall']:
            # evaluate wall by how much it reduces opponent paths
            opp_paths_before, _ = count_shortest_paths(
                env, opp_pos, opp_goal_row)
            # temporarily place wall
            env.place_wall(
                'h' if action[0] == 'h_wall' else 'v', (action[1], action[2]))
            opp_paths_after, _ = count_shortest_paths(
                env, opp_pos, opp_goal_row)
            env.remove_wall(
                'h' if action[0] == 'h_wall' else 'v', (action[1], action[2]))

            wall_effect = max(0, opp_paths_before - opp_paths_after)
            reward += 0.3 * wall_effect  # reward for blocking opponent

            # small penalty for placing walls in general
            reward -= 0.1

        return reward

    def train_episode(self, env, max_steps=200, epsilon=0.1):
        env.reset()
        self.prev_positions = set()
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        episode_reward = 0

        for step in range(max_steps):
            prev_white_pos, prev_black_pos = list(
                env.white_pos), list(env.black_pos)

            state_tensor = self.state_to_tensor(env)
            action, log_prob, value = self.select_action(env, epsilon)

            if action is None:
                break

            states.append(state_tensor)
            actions.append(action)
            values.append(value)
            log_probs.append(log_prob)

            if action[0] == 'move':
                env.move_pawn((action[1], action[2]))
            elif action[0] == 'h_wall':
                env.place_wall('h', (action[1], action[2]))
            else:
                env.place_wall('v', (action[1], action[2]))

            winner = env.is_game_over()
            done = winner is not None

            if done:
                reward = 10.0 if winner == 'white' else -10.0
            else:
                reward = self.shaped_reward(
                    env, action, prev_white_pos, prev_black_pos, env.turn)

            rewards.append(reward)
            dones.append(1.0 if done else 0.0)
            episode_reward += reward

            if done:
                break

        next_value = 0.0
        if not done:
            with torch.no_grad():
                final_state = self.state_to_tensor(
                    env).unsqueeze(0).to(self.device)
                _, next_value_tensor = self.network(final_state)
                next_value = next_value_tensor.item()

        returns, advantages = self.compute_returns(
            rewards, values, next_value, dones)

        if len(states) > 0:
            self.train_step(states, actions, returns, advantages)

        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(len(states))
        return episode_reward, len(states)

    def save_model(self, filepath):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        print(f"Model loaded from {filepath}")

    def get_training_stats(self):
        if not self.episode_rewards:
            return {}
        recent = min(100, len(self.episode_rewards))
        return {
            'episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards[-recent:]),
            'avg_length': np.mean(self.episode_lengths[-recent:]),
            'total_reward': sum(self.episode_rewards),
            'recent_avg_loss': np.mean(self.losses[-recent:]) if self.losses else 0,
        }
