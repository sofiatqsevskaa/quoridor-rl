import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from quoridor_env import QuoridorEnv
from actor_critic import QuoridorActorCritic


def save_episode_states(env, states, actions, episode_num, winner, folder='saved_episodes'):
    os.makedirs(folder, exist_ok=True)

    episode_data = {
        'metadata': {
            'episode_number': episode_num,
            'winner': winner,
            'total_steps': len(states) - 1,
            'timestamp': datetime.now().isoformat()
        },
        'states': states,
        'actions': actions
    }

    filename = f"episode_{episode_num:04d}_{winner if winner else 'incomplete'}.json"
    filepath = os.path.join(folder, filename)

    with open(filepath, 'w') as f:
        json.dump(episode_data, f, indent=2)

    return filepath


def record_state(env):
    return {
        "white_pos": list(env.white_pos),
        "black_pos": list(env.black_pos),
        "h_walls": [list(w) for w in env.h_walls],
        "v_walls": [list(w) for w in env.v_walls],
        "white_walls": env.white_walls,
        "black_walls": env.black_walls,
        "turn": env.turn
    }


def save_training_episode(env, agent, episode_num):
    env.reset()

    states = [record_state(env)]
    actions = []

    max_steps = 200
    for step in range(max_steps):
        action, _, _ = agent.select_action(env, epsilon=0.0)

        if action is None:
            break

        if episode_num <= 1000 and action[0] in ['h_wall', 'v_wall']:
            continue

        actions.append(list(action))

        if action[0] == 'move':
            env.move_pawn((action[1], action[2]))
        elif action[0] == 'h_wall':
            env.place_wall('h', (action[1], action[2]))
        else:
            env.place_wall('v', (action[1], action[2]))

        states.append(record_state(env))

        winner = env.is_game_over()
        if winner:
            save_episode_states(env, states, actions, episode_num, winner)
            print(f"  Episode saved: {winner} wins in {step+1} steps")
            return winner

    return None


def train_agent(num_episodes=1000, save_episodes_every=50, save_model_every=100):
    env = QuoridorEnv(size=9, num_walls=10)
    agent = QuoridorActorCritic(board_size=9, learning_rate=0.001, gamma=0.99)

    print("Starting training...")
    print(f"Device: {agent.device}")
    print("-" * 50)

    for episode in range(1, num_episodes + 1):
        epsilon = max(0.01, 0.5 - (episode / num_episodes) * 0.49)

        env.num_walls = 0 if episode <= 1000 else 10

        episode_reward, episode_length = agent.train_episode(
            env, max_steps=200, epsilon=epsilon)

        stats = agent.get_training_stats()

        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Length: {episode_length} | "
                  f"Avg Reward (last 100): {stats['avg_reward']:.2f} | "
                  f"Epsilon: {epsilon:.3f}")

        if episode % save_episodes_every == 0:
            print(f"Saving episode {episode}...")
            save_training_episode(env, agent, episode)

        if episode % save_model_every == 0:
            agent.save_model(f"models/quoridor_ac_ep{episode}.pt")

    print("\nTraining completed!")
    print(f"Final statistics:")
    print(f"  Total episodes: {stats['episodes']}")
    print(f"  Average reward (last 100): {stats['avg_reward']:.2f}")
    print(f"  Average length (last 100): {stats['avg_length']:.2f}")

    agent.save_model("models/quoridor_ac_final.pt")

    plot_training_curves(agent)

    return agent


def plot_training_curves(agent):
    if not agent.episode_rewards:
        print("No training data to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Actor-Critic Training Statistics')

    axes[0, 0].plot(agent.episode_rewards, alpha=0.3, label='Episode Reward')
    window = min(50, len(agent.episode_rewards) // 10)
    if window > 0:
        moving_avg = [sum(agent.episode_rewards[max(0, i-window):i+1]) /
                      min(window, i+1) for i in range(len(agent.episode_rewards))]
        axes[0, 0].plot(
            moving_avg, label=f'Moving Avg ({window})', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(agent.episode_lengths, alpha=0.3, label='Episode Length')
    if window > 0:
        moving_avg_len = [sum(agent.episode_lengths[max(0, i-window):i+1]) /
                          min(window, i+1) for i in range(len(agent.episode_lengths))]
        axes[0, 1].plot(
            moving_avg_len, label=f'Moving Avg ({window})', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    if agent.losses:
        axes[1, 0].plot(agent.losses, alpha=0.5)
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].grid(True, alpha=0.3)

    cumulative_reward = [sum(agent.episode_rewards[:i+1])
                         for i in range(len(agent.episode_rewards))]
    axes[1, 1].plot(cumulative_reward)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Cumulative Reward')
    axes[1, 1].set_title('Cumulative Reward Over Time')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("Training curves saved to 'training_curves.png'")
    plt.show()


if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('saved_episodes', exist_ok=True)

    num_episodes = int(input("Number of episodes (default 1000): ") or "1000")
    save_episodes_every = int(
        input("Save episode states every N episodes (default 50): ") or "50")

    agent = train_agent(
        num_episodes=num_episodes,
        save_episodes_every=save_episodes_every,
        save_model_every=100
    )
