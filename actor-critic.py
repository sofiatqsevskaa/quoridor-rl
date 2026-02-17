import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
from quoridor_env import QuoridorEnv
import helper_functions as helpers


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)


class Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def train_actor_critic(episodes=5000, size=9, gamma=0.99, lr_actor=1e-4, lr_critic=1e-3):
    env = QuoridorEnv(size=size, num_walls=3)
    state_dim = size * size * 2 + 2
    action_dim = 4 + size * size

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)

    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    episode_rewards = []
    episode_lengths = []

    for ep in range(1, episodes + 1):
        env.reset()
        state = helpers.env_to_state_vector(env)
        episode_states = [helpers.snapshot(env)]
        visit_count = {}
        step = 0
        total_reward = 0.0

        log_probs = []
        values = []
        rewards = []

        while not env.is_game_over() and step < 150:
            step += 1
            visit_count[env.white_pos] = visit_count.get(env.white_pos, 0) + 1

            prev_state = (
                env.white_pos,
                env.black_pos,
                set(env.h_walls),
                set(env.v_walls)
            )

            state_tensor = torch.tensor(state).float().unsqueeze(0)

            action_probs = actor(state_tensor)
            value = critic(state_tensor)

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
            total_reward += reward

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            state = next_state

        if env.is_game_over() == "white":
            final_reward = 500.0
        elif env.is_game_over() == "black":
            final_reward = -500.0
        else:
            final_reward = 0.0

        rewards.append(final_reward)
        total_reward += final_reward

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns[:-1])
        values = torch.cat(values).squeeze()

        advantage = returns - values

        actor_loss = 0
        for log_prob, adv in zip(log_probs, advantage):
            actor_loss += -log_prob * adv.detach()

        critic_loss = advantage.pow(2).mean()

        optimizer_actor.zero_grad()
        actor_loss.backward()
        optimizer_actor.step()

        optimizer_critic.zero_grad()
        critic_loss.backward()
        optimizer_critic.step()

        episode_rewards.append(total_reward)
        episode_lengths.append(step)

        if ep % 100 == 0:
            helpers.save_episode(episode_states, ep, env, "actor_critic")
            print(f"Episode {ep} Reward={total_reward:.1f} Steps={step}")

    torch.save(actor.state_dict(), "actor_model.pth")
    torch.save(critic.state_dict(), "critic_model.pth")

    helpers.plot_training(episode_rewards, episode_lengths, "actor_critic")


if __name__ == "__main__":
    train_actor_critic(episodes=10000)
