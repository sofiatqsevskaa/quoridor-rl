import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from quoridor_env import QuoridorEnv
import helper_functions as helpers


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, output_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        features = self.shared(x)
        action_probs = torch.softmax(self.actor(features), dim=-1)
        state_value = self.critic(features)
        return action_probs, state_value


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.batch_size = batch_size

    def push(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def get_batches(self):
        n = len(self.states)
        batch_start = np.arange(0, n, self.batch_size)
        indices = np.arange(n, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return batches


def train_ppo(episodes=10000, size=9, gamma=0.99, lam=0.95, clip_epsilon=0.2,
              lr=3e-4, epochs=10, batch_size=64, update_timestep=2000):

    env = QuoridorEnv(size=size, num_walls=3)
    state_dim = size * size * 2 + 2
    action_dim = 4 + size * size

    policy = ActorCriticNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    memory = PPOMemory(batch_size)

    episode_rewards = []
    episode_lengths = []
    timestep = 0

    for ep in range(1, episodes + 1):
        env.reset()
        state = helpers.env_to_state_vector(env)
        episode_states = [helpers.snapshot(env)]
        total_reward = 0
        done = False
        step = 0
        visit_count = {}

        while not done and step < 200:
            timestep += 1
            step += 1
            visit_count[env.white_pos] = visit_count.get(env.white_pos, 0) + 1

            prev_state = (
                env.white_pos,
                env.black_pos,
                set(env.h_walls),
                set(env.v_walls)
            )

            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs, value = policy(state_tensor)

            dist = torch.distributions.Categorical(action_probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)

            action = helpers.index_to_action(action_idx.item(), env)

            if action[0] == 'move':
                env.move_pawn((action[1], action[2]))
            else:
                env.place_wall(action[0][0], (action[1], action[2]))

            opp = env.get_legal_moves()
            if opp and not env.is_game_over():
                o = random.choice(opp)
                if o[0] == 'move':
                    env.move_pawn((o[1], o[2]))
                else:
                    env.place_wall(o[0][0], (o[1], o[2]))

            episode_states.append(helpers.snapshot(env))
            next_state = helpers.env_to_state_vector(env)
            reward = helpers.compute_reward(env, prev_state, visit_count)

            done = env.is_game_over() is not None
            total_reward += reward

            memory.push(state, action_idx.item(), reward,
                        value.item(), log_prob.item(), done)

            state = next_state

        if done:
            if env.is_game_over() == "white":
                final_reward = 500.0
            else:
                final_reward = -500.0
        else:
            final_reward = 0.0

        total_reward += final_reward
        episode_rewards.append(total_reward)
        episode_lengths.append(step)

        if timestep >= update_timestep:
            print(f"Updating PPO at episode {ep}, timestep {timestep}")
            learn(policy, memory, optimizer, gamma, lam, clip_epsilon, epochs)
            memory.clear()
            timestep = 0

        if ep % 100 == 0:
            helpers.save_episode(episode_states, ep, env, "ppo")
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            print(
                f"Episode {ep} Avg Reward={avg_reward:.1f} Avg Steps={avg_length:.1f}")

    torch.save(policy.state_dict(), "ppo_model.pth")
    helpers.plot_training(episode_rewards, episode_lengths, "ppo")
    return policy


def learn(policy, memory, optimizer, gamma, lam, clip_epsilon, epochs):
    for _ in range(epochs):
        returns = compute_gae(memory, gamma, lam)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.tensor(memory.values, dtype=torch.float32)
        old_log_probs = torch.tensor(memory.log_probs, dtype=torch.float32)
        states = torch.tensor(np.array(memory.states), dtype=torch.float32)
        actions = torch.tensor(memory.actions)

        advantage = returns - values
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        for batch in memory.get_batches():
            state_batch = states[batch]
            action_batch = actions[batch]
            old_log_prob_batch = old_log_probs[batch]
            advantage_batch = advantage[batch]
            return_batch = returns[batch]

            action_probs, critic_value = policy(state_batch)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(action_batch)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_prob_batch)
            surr1 = ratio * advantage_batch
            surr2 = torch.clamp(ratio, 1 - clip_epsilon,
                                1 + clip_epsilon) * advantage_batch
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.MSELoss()(critic_value.squeeze(), return_batch)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()


def compute_gae(memory, gamma, lam):
    rewards = memory.rewards
    values = memory.values
    dones = memory.dones
    gae = 0
    returns = []

    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[step + 1]

        delta = rewards[step] + gamma * next_value * \
            (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        returns.insert(0, gae + values[step])

    return returns


if __name__ == "__main__":
    train_ppo(episodes=10000, update_timestep=2048, epochs=10, batch_size=64)
