import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, player_id, state_size=200,
                 lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 batch_size=64, memory_size=10000, device=None):
        self.player_id = player_id
        self.state_size = state_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        print(f"Agent {player_id} using device: {self.device}")

        self.action_size = 209

        self.policy_net = DQN(state_size, self.action_size).to(self.device)
        self.target_net = DQN(state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(memory_size)

        self.action_to_index = {}
        self.index_to_action = {}
        idx = 0

        for r in range(9):
            for c in range(9):
                self.action_to_index[('move', r, c)] = idx
                self.index_to_action[idx] = ('move', r, c)
                idx += 1

        for r in range(8):
            for c in range(8):
                self.action_to_index[('h_wall', r, c)] = idx
                self.index_to_action[idx] = ('h_wall', r, c)
                idx += 1

        for r in range(8):
            for c in range(8):
                self.action_to_index[('v_wall', r, c)] = idx
                self.index_to_action[idx] = ('v_wall', r, c)
                idx += 1

    def state_to_vector(self, state):
        vector = []

        white_pos = state['white_pos']
        black_pos = state['black_pos']
        vector.extend([white_pos[0] / 8.0, white_pos[1] / 8.0])
        vector.extend([black_pos[0] / 8.0, black_pos[1] / 8.0])

        h_wall_grid = np.zeros(81)
        for wall in state['h_walls']:
            if 0 <= wall[0] < 9 and 0 <= wall[1] < 9:
                idx = wall[0] * 9 + wall[1]
                h_wall_grid[idx] = 1
        vector.extend(h_wall_grid.tolist())

        v_wall_grid = np.zeros(81)
        for wall in state['v_walls']:
            if 0 <= wall[0] < 9 and 0 <= wall[1] < 9:
                idx = wall[0] * 9 + wall[1]
                v_wall_grid[idx] = 1
        vector.extend(v_wall_grid.tolist())

        vector.extend([state['white_walls'] / 10.0,
                      state['black_walls'] / 10.0])

        vector.append(float(state['turn']))

        while len(vector) < self.state_size:
            vector.append(0.0)

        return np.array(vector[:self.state_size], dtype=np.float32)

    def get_legal_actions(self, env):
        actions = []

        legal_moves = env.get_legal_moves()
        for move in legal_moves:
            actions.append(('move', move[0], move[1]))

        walls_remaining = env.white_walls if env.turn == 0 else env.black_walls
        if walls_remaining > 0:
            for r in range(8):
                for c in range(8):
                    if env.is_valid_wall('h', (r, c)):
                        actions.append(('h_wall', r, c))

            for r in range(8):
                for c in range(8):
                    if env.is_valid_wall('v', (r, c)):
                        actions.append(('v_wall', r, c))

        return actions

    def choose_action(self, env):
        legal_actions = self.get_legal_actions(env)

        if not legal_actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        state = env.get_state()
        state_vector = self.state_to_vector(state)
        state_tensor = torch.FloatTensor(
            state_vector).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]

        legal_indices = [self.action_to_index[action] for action in legal_actions
                         if action in self.action_to_index]

        if not legal_indices:
            return random.choice(legal_actions)

        legal_q_values = [(idx, q_values[idx]) for idx in legal_indices]
        best_idx = max(legal_q_values, key=lambda x: x[1])[0]

        return self.index_to_action[best_idx]

    def execute_action(self, env, action):
        if action[0] == 'move':
            env.move_pawn((action[1], action[2]))
        elif action[0] == 'h_wall':
            env.place_wall('h', (action[1], action[2]))
        elif action[0] == 'v_wall':
            env.place_wall('v', (action[1], action[2]))

    def remember(self, state, action, reward, next_state, done):
        state_vec = self.state_to_vector(state)
        next_state_vec = self.state_to_vector(next_state)
        action_idx = self.action_to_index.get(action, 0)

        self.memory.push(state_vec, action_idx, reward, next_state_vec, done)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states).gather(
            1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


def train_agents(episodes=10000, print_every=100, target_update=10, use_device=None):
    from quoridor_env import QuoridorEnv

    if use_device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(use_device)

    print(f"device: {device}")

    env = QuoridorEnv(size=9, num_walls=10)

    white_agent = DQNAgent(player_id=0, device=device)
    black_agent = DQNAgent(player_id=1, device=device)

    white_wins = 0
    black_wins = 0
    draws = 0
    walls_placed = 0

    import time
    start_time = time.time()

    for episode in range(episodes):
        state = env.reset()
        done = False
        move_count = 0
        max_moves = 200

        episode_loss_white = []
        episode_loss_black = []
        episode_walls = 0

        while not done and move_count < max_moves:
            current_agent = white_agent if env.turn == 0 else black_agent

            current_state = env.get_state()
            action = current_agent.choose_action(env)

            if action is None:
                break

            if action[0] in ['h_wall', 'v_wall']:
                episode_walls += 1
                walls_placed += 1

            prev_white_pos = env.white_pos
            prev_black_pos = env.black_pos
            current_agent.execute_action(env, action)
            next_state = env.get_state()

            winner = env.is_game_over()
            if winner:
                done = True
                if winner == "white":
                    white_reward = 100
                    black_reward = -100
                    white_wins += 1
                elif winner == "black":
                    white_reward = -100
                    black_reward = 100
                    black_wins += 1

                white_agent.remember(current_state, action,
                                     white_reward, next_state, done)
                black_agent.remember(current_state, action,
                                     black_reward, next_state, done)
            else:
                reward = -0.1

                if action[0] == 'move':
                    if env.turn == 1:
                        distance_reward = (prev_white_pos[0] - action[1]) * 2
                        reward += distance_reward
                    else:
                        distance_reward = (action[1] - prev_black_pos[0]) * 2
                        reward += distance_reward

                elif action[0] in ['h_wall', 'v_wall']:
                    reward += 0.5

                if env.turn == 1:
                    white_agent.remember(
                        current_state, action, reward, next_state, done)
                else:
                    black_agent.remember(
                        current_state, action, reward, next_state, done)

            if env.turn == 1:
                loss = white_agent.replay()
                if loss is not None:
                    episode_loss_white.append(loss)
            else:
                loss = black_agent.replay()
                if loss is not None:
                    episode_loss_black.append(loss)

            move_count += 1

        if move_count >= max_moves:
            draws += 1

        white_agent.decay_epsilon()
        black_agent.decay_epsilon()

        if episode % target_update == 0:
            white_agent.update_target_network()
            black_agent.update_target_network()

        if (episode + 1) % print_every == 0:
            elapsed_time = time.time() - start_time
            episodes_per_sec = (episode + 1) / elapsed_time

            avg_loss_white = np.mean(
                episode_loss_white) if episode_loss_white else 0
            avg_loss_black = np.mean(
                episode_loss_black) if episode_loss_black else 0
            avg_walls = walls_placed / print_every

            print(
                f"\nEpisode {episode + 1}/{episodes} ({episodes_per_sec:.2f} eps/sec)")
            print(
                f"  White wins: {white_wins}, Black wins: {black_wins}, Draws: {draws}")
            print(
                f"  Epsilon: White={white_agent.epsilon:.4f}, Black={black_agent.epsilon:.4f}")
            print(
                f"  Avg Loss: White={avg_loss_white:.4f}, Black={avg_loss_black:.4f}")
            print(f"  Walls placed (avg): {avg_walls:.2f} per game")
            print(
                f"  Memory size: White={len(white_agent.memory)}, Black={len(black_agent.memory)}")
            print(f"  Time elapsed: {elapsed_time/60:.2f} minutes")

            white_wins = 0
            black_wins = 0
            draws = 0
            walls_placed = 0

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Average: {episodes/total_time:.2f} episodes/second")
    print(f"{'='*60}\n")

    return white_agent, black_agent


def play_game(white_agent, black_agent, render=True):
    from quoridor_env import QuoridorEnv

    env = QuoridorEnv(size=9, num_walls=10)
    env.reset()

    if render:
        print("\nStarting game:")
        env.print_board()

    done = False
    move_count = 0
    max_moves = 200

    while not done and move_count < max_moves:
        current_agent = white_agent if env.turn == 0 else black_agent

        old_epsilon = current_agent.epsilon
        current_agent.epsilon = 0
        action = current_agent.choose_action(env)
        current_agent.epsilon = old_epsilon

        if action is None:
            print("No legal actions available!")
            break

        player = "White" if env.turn == 0 else "Black"
        if render:
            if action[0] == 'move':
                print(f"\n{player} moves to ({action[1]}, {action[2]})")
            elif action[0] == 'h_wall':
                print(
                    f"\n{player} places horizontal wall at ({action[1]}, {action[2]})")
            elif action[0] == 'v_wall':
                print(
                    f"\n{player} places vertical wall at ({action[1]}, {action[2]})")

        current_agent.execute_action(env, action)

        if render:
            env.print_board()

        winner = env.is_game_over()
        if winner:
            done = True
            if render:
                print(f"\n{winner.capitalize()} wins!")
            return winner

        move_count += 1

    if render:
        print("\nGame ended in a draw (max moves reached)")
    return "draw"


if __name__ == "__main__":
    print("Training DQN agents with wall placement...")

    device = "mps" if torch.backends.mps.is_available() else None

    white_agent, black_agent = train_agents(
        episodes=5000,
        print_every=100,
        target_update=10,
        use_device=device
    )

    white_agent.save("white_dqn_agent.pth")
    black_agent.save("black_dqn_agent.pth")

    print("\n" + "="*50)
    print("Training complete! Playing demonstration games...")
    print("="*50)

    results = {"white": 0, "black": 0, "draw": 0}
    for i in range(10):
        print(f"\n--- Game {i+1} ---")
        result = play_game(white_agent, black_agent, render=(i < 3))
        results[result] += 1

    print("\n" + "="*50)
    print("Final Results:")
    print(f"  White wins: {results['white']}")
    print(f"  Black wins: {results['black']}")
    print(f"  Draws: {results['draw']}")
