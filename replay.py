import os
import json
from quoridor_env import QuoridorEnv
from visualizer import QuoridorVisualizer
import pygame
import sys
import time


def replay_episode(filepath, auto_play=True, delay_ms=200):
    with open(filepath, 'r') as f:
        episode_data = json.load(f)

    size = episode_data["size"]
    num_walls = episode_data["num_walls"]
    states = episode_data["states"]

    env = QuoridorEnv(size=size, num_walls=num_walls)
    visualizer = QuoridorVisualizer(board_size=size, cell_size=60)

    step = 0

    while step < len(states):
        state = states[step]

        env.white_pos = tuple(state['white_pos'])
        env.black_pos = tuple(state['black_pos'])
        env.h_walls = set(tuple(w) for w in state['h_walls'])
        env.v_walls = set(tuple(w) for w in state['v_walls'])
        env.white_walls = state['white_walls']
        env.black_walls = state['black_walls']
        env.turn = state['turn']

        visualizer.draw_board(env, step=step, total_steps=len(states) - 1)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                visualizer.close()
                return False

        pygame.time.delay(delay_ms)
        step += 1

    pygame.time.delay(500)
    visualizer.close()
    return True


def replay_all_episodes(folder_path='saved_episodes', auto_play=True, delay_ms=200, reverse=True):
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found!")
        return

    json_files = []
    for file in os.listdir(folder_path):
        if file.endswith('.json'):
            json_files.append(file)

    if not json_files:
        print(f"No JSON files found in '{folder_path}'")
        return

    json_files.sort(reverse=reverse)

    print(f"Found {len(json_files)} episodes to replay")
    print(f"Order: {'Newest to Oldest' if reverse else 'Oldest to Newest'}")
    print(f"Auto-play: {'ON' if auto_play else 'OFF'} (delay: {delay_ms}ms)")
    print("=" * 50)

    for i, filename in enumerate(json_files):
        filepath = os.path.join(folder_path, filename)
        print(f"\n[{i+1}/{len(json_files)}]", end=" ")

        continue_playing = replay_episode(
            filepath, auto_play=auto_play, delay_ms=delay_ms)

        if not continue_playing:
            print("\nReplay stopped by user")
            break

    print("\nAll episodes replayed!")


if __name__ == "__main__":
    delay_ms = 200
    replay_all_episodes(auto_play=True, delay_ms=delay_ms, reverse=True)
