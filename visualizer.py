import pygame
import sys
import json
import pickle
from datetime import datetime
import os


class QuoridorVisualizer:
    def __init__(self, board_size=9, cell_size=60):
        pygame.init()
        self.board_size = board_size
        self.cell_size = cell_size
        self.window_size = board_size * cell_size + 40
        self.screen = pygame.display.set_mode(
            (self.window_size, self.window_size))
        pygame.display.set_caption(f"Quoridor")

        self.bg_color = (210, 210, 210)
        self.grid_color = (180, 180, 180)
        self.white_color = (255, 255, 255)
        self.black_color = (30, 30, 30)
        self.wall_color = (160, 120, 80)
        self.text_color = (50, 50, 50)

        self.font = pygame.font.SysFont(None, 24)
        self.history = []
        self.move_history = []
        self.current_episode = 1
        self.update_title()

    def update_title(self):
        pygame.display.set_caption(
            f"Quoridor - Episode {self.current_episode}")

    def record_state(self, env):
        state = {
            "white_pos": env.white_pos,
            "black_pos": env.black_pos,
            "h_walls": list(env.h_walls),
            "v_walls": list(env.v_walls),
            "white_walls": env.white_walls,
            "black_walls": env.black_walls,
            "turn": env.turn,
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(state)

    def record_move(self, action, env):
        move_data = {
            "action": action,
            "turn": env.turn,
            "state_before": self.history[-1] if self.history else None
        }
        self.move_history.append(move_data)

    def save_episode(self, env, filename=None, format='json'):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quoridor_episode_{timestamp}"

        episode_data = {
            'metadata': {
                'size': env.size,
                'num_walls': env.num_walls,
                'created_at': datetime.now().isoformat(),
                'total_states': len(self.history),
                'total_moves': len(self.move_history),
                'episode_number': self.current_episode
            },
            'initial_state': self.history[0] if self.history else None,
            'states': self.history,
            'moves': self.move_history,
            'final_state': self.history[-1] if self.history else None,
            'winner': env.is_game_over()
        }

        os.makedirs('saved_episodes', exist_ok=True)
        filepath = os.path.join('saved_episodes', filename)

        if format == 'json':
            if not filepath.endswith('.json'):
                filepath += '.json'

            def convert_for_json(obj):
                if isinstance(obj, tuple):
                    return list(obj)
                elif isinstance(obj, set):
                    return list(obj)
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj

            json_data = convert_for_json(episode_data)
            with open(filepath, 'w') as f:
                json.dump(json_data, f, indent=2)

        elif format == 'pickle':
            if not filepath.endswith('.pkl'):
                filepath += '.pkl'
            with open(filepath, 'wb') as f:
                pickle.dump(episode_data, f)

        print(f"Episode {self.current_episode} saved to {filepath}")
        return filepath

    def load_episode(self, filepath):
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
        elif filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError("Unsupported file format")

        return data

    def replay_episode(self, env, episode_data):
        print(f"Replaying episode...")

        states = episode_data['states']
        moves = episode_data['moves']

        self.screen.fill((50, 50, 50))
        font = pygame.font.SysFont(None, 36)
        text = font.render("Press SPACE to start replay",
                           True, (255, 255, 255))
        text_rect = text.get_rect(
            center=(self.window_size//2, self.window_size//2))
        self.screen.blit(text, text_rect)
        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

        step = 0
        playing = True

        while playing and step < len(states):
            state = states[step]

            env.white_pos = tuple(state['white_pos'])
            env.black_pos = tuple(state['black_pos'])
            env.h_walls = set(tuple(w) for w in state['h_walls'])
            env.v_walls = set(tuple(w) for w in state['v_walls'])
            env.white_walls = state['white_walls']
            env.black_walls = state['black_walls']
            env.turn = state['turn']

            self.draw_board(env, replay=True, step=step,
                            total_steps=len(states)-1)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    playing = False
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        step = min(step + 1, len(states) - 1)
                    elif event.key == pygame.K_LEFT:
                        step = max(step - 1, 0)
                    elif event.key == pygame.K_SPACE:
                        step += 1
                    elif event.key == pygame.K_r:
                        step = 0
                    elif event.key == pygame.K_ESCAPE:
                        playing = False

            pygame.time.delay(500)

        print("Replay finished")

    def draw_board(self, env, replay=False, step=0, total_steps=0):
        self.screen.fill(self.bg_color)

        for i in range(self.board_size):
            for j in range(self.board_size):
                rect = pygame.Rect(
                    j * self.cell_size + 20,
                    i * self.cell_size + 20,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, (210, 210, 210), rect, 0)
                pygame.draw.rect(self.screen, self.grid_color, rect, 1)

        for wall in env.h_walls:
            r, c = wall
            x_start = c * self.cell_size + 20
            y = r * self.cell_size + 20 + self.cell_size
            pygame.draw.rect(
                self.screen,
                self.wall_color,
                (x_start, y - 2, self.cell_size * 2, 4)
            )

        for wall in env.v_walls:
            r, c = wall
            x = c * self.cell_size + 20 + self.cell_size
            y_start = r * self.cell_size + 20
            pygame.draw.rect(
                self.screen,
                self.wall_color,
                (x - 2, y_start, 4, self.cell_size * 2)
            )

        w_r, w_c = env.white_pos
        white_center = (
            w_c * self.cell_size + 20 + self.cell_size // 2,
            w_r * self.cell_size + 20 + self.cell_size // 2
        )
        pygame.draw.circle(self.screen, self.white_color,
                           white_center, self.cell_size // 3)

        b_r, b_c = env.black_pos
        black_center = (
            b_c * self.cell_size + 20 + self.cell_size // 2,
            b_r * self.cell_size + 20 + self.cell_size // 2
        )
        pygame.draw.circle(self.screen, self.black_color,
                           black_center, self.cell_size // 3)

        turn_text = f"Turn: {'White' if env.turn == 0 else 'Black'}"
        walls_text = f"Walls - White: {env.white_walls}, Black: {env.black_walls}"
        episode_text = f"Episode: {self.current_episode}"

        turn_surface = self.font.render(turn_text, True, self.text_color)
        walls_surface = self.font.render(walls_text, True, self.text_color)
        episode_surface = self.font.render(episode_text, True, self.text_color)

        self.screen.blit(turn_surface, (10, 5))
        self.screen.blit(walls_surface, (10, 30))
        self.screen.blit(episode_surface, (self.window_size - 150, 5))

        if replay:
            replay_text = f"Replay: Step {step}/{total_steps}"
            controls_text = "LEFT/RIGHT: Navigate, SPACE: Next, R: Reset, ESC: Exit"
            replay_surface = self.font.render(replay_text, True, (200, 50, 50))
            controls_surface = self.font.render(
                controls_text, True, self.text_color)
            self.screen.blit(replay_surface, (10, self.window_size - 60))
            self.screen.blit(controls_surface, (10, self.window_size - 30))

        pygame.display.flip()

    def check_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_s:
                    return 'save'
                if event.key == pygame.K_l:
                    return 'load'
                if event.key == pygame.K_n:
                    self.current_episode += 1
                    self.history = []
                    self.move_history = []
                    self.update_title()
                    return 'new'
        return None

    def reset_recording(self):
        self.history = []
        self.move_history = []

    def close(self):
        pygame.quit()
