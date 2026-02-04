from quoridor_env import QuoridorEnv
from minimax import minimax
from visualizer import QuoridorVisualizer
import pygame


def play_game(size):
    env = QuoridorEnv(size=size, num_walls=10)
    viz = QuoridorVisualizer(board_size=size)
    moves = 0

    while not env.is_game_over():
        actions = env.get_legal_moves()
        best_action = None
        best_value = -float('inf') if env.turn == 0 else float('inf')

        for action in actions:
            new_env = env.copy()
            if action[0] == 'move':
                new_env.move_pawn((action[1], action[2]))
            else:
                new_env.place_wall(action[0][0], (action[1], action[2]))

            score, _ = minimax(
                new_env, depth=2, maximizing_player=(env.turn == 0))

            if env.turn == 0 and score > best_value:
                best_value = score
                best_action = action
            elif env.turn == 1 and score < best_value:
                best_value = score
                best_action = action

        if best_action[0] == 'move':
            env.move_pawn((best_action[1], best_action[2]))
        else:
            env.place_wall(best_action[0][0], (best_action[1], best_action[2]))

        moves += 1
        viz.draw_board(env)
        pygame.time.delay(500)

    winner = env.is_game_over()
    print(f"Board {size}x{size}: Winner = {winner}, Total moves = {moves}")
    viz.close()


if __name__ == "__main__":
    play_game(5)
    play_game(9)
