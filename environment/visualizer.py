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
        pygame.display.set_caption("Quoridor Replay")

        self.bg_color = (210, 210, 210)
        self.grid_color = (180, 180, 180)
        self.white_color = (255, 255, 255)
        self.black_color = (30, 30, 30)
        self.wall_color = (160, 120, 80)
        self.text_color = (50, 50, 50)
        self.font = pygame.font.SysFont(None, 24)

    def draw_board(self, env, step=0, total_steps=0):
        self.screen.fill(self.bg_color)
        for i in range(self.board_size):
            for j in range(self.board_size):
                rect = pygame.Rect(
                    j * self.cell_size + 20,
                    i * self.cell_size + 20,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, self.bg_color, rect)
                pygame.draw.rect(self.screen, self.grid_color, rect, 1)

        for r, c in env.h_walls:
            pygame.draw.rect(
                self.screen,
                self.wall_color,
                (c * self.cell_size + 20, r * self.cell_size +
                 20 + self.cell_size, self.cell_size * 2, 4)
            )
        for r, c in env.v_walls:
            pygame.draw.rect(
                self.screen,
                self.wall_color,
                (c * self.cell_size + 20 + self.cell_size, r *
                 self.cell_size + 20, 4, self.cell_size * 2)
            )

        w_r, w_c = env.white_pos
        pygame.draw.circle(
            self.screen,
            self.white_color,
            (w_c * self.cell_size + 20 + self.cell_size // 2,
             w_r * self.cell_size + 20 + self.cell_size // 2),
            self.cell_size // 3
        )

        b_r, b_c = env.black_pos
        pygame.draw.circle(
            self.screen,
            self.black_color,
            (b_c * self.cell_size + 20 + self.cell_size // 2,
             b_r * self.cell_size + 20 + self.cell_size // 2),
            self.cell_size // 3
        )

        if total_steps > 0:
            step_text = f"Step {step}/{total_steps}"
            step_surface = self.font.render(step_text, True, self.text_color)
            self.screen.blit(step_surface, (10, self.window_size - 30))

        pygame.display.flip()

    def close(self):
        pygame.quit()

    def load_episode(self, filepath):
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                return json.load(f)
        elif filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError("Unsupported file format")
