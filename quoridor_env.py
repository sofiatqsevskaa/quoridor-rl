import random


class QuoridorEnv:
    possible_moves = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1)
    }

    def __init__(self, size=9, num_walls=10):
        self.size = size
        self.num_walls = num_walls
        self.reset()

    def reset(self):
        self.possible_vertical_walls = set()
        self.possible_horizontal_walls = set()
        for row_idx in range(self.size*2-1):
            for col_idx in range(self.size*2-1):
                if row_idx % 2 == 0 and col_idx % 2 == 1:
                    self.possible_vertical_walls.add((row_idx//2, col_idx//2))

        self.board = [['.' for _ in range(self.size)]
                      for _ in range(self.size)]
        self.white_pos = (self.size - 1, self.size // 2)
        self.black_pos = (0, self.size // 2)
        self.board[self.white_pos[0]][self.white_pos[1]] = 'W'
        self.board[self.black_pos[0]][self.black_pos[1]] = 'B'

        self.h_walls = set()
        self.v_walls = set()
        self.white_walls = self.num_walls
        self.black_walls = self.num_walls

        self.turn = 0

        return self.get_state()

    def draw_board(self):
        self.board = [['.' for _ in range(self.size)]
                      for _ in range(self.size)]
        self.board[self.white_pos[0]][self.white_pos[1]] = 'W'
        self.board[self.black_pos[0]][self.black_pos[1]] = 'B'

    def get_state(self):
        return {
            "white_pos": self.white_pos,
            "black_pos": self.black_pos,
            "h_walls": list(self.h_walls),
            "v_walls": list(self.v_walls),
            "white_walls": self.white_walls,
            "black_walls": self.black_walls,
            "turn": self.turn
        }

    def print_board(self):
        for row_idx in range(self.size*2-1):
            for col_idx in range(self.size*2-1):
                if row_idx % 2 == 0 and col_idx % 2 == 0:
                    if (row_idx//2, col_idx//2) == self.white_pos:
                        print(" W ", end="")
                    elif (row_idx//2, col_idx//2) == self.black_pos:
                        print(" B ", end="")
                    else:
                        print(" . ", end="")
                elif row_idx % 2 == 0:
                    if (row_idx//2, col_idx//2) in self.v_walls:
                        print(" | ", end="")
                    else:
                        print("   ", end="")
                elif col_idx % 2 == 0:
                    if (row_idx//2, col_idx//2) in self.h_walls:
                        print(" - ", end="")
                    else:
                        print("   ", end="")
                else:
                    print("   ", end="")
            print()

    def move_pawn(self, pos):
        if self.turn == 0:
            if pos in self.get_legal_moves():
                self.white_pos = pos
        else:
            if pos in self.get_legal_moves():
                self.black_pos = pos
        self.draw_board()
        self.turn = abs(self.turn - 1)

    def get_legal_moves(self):
        r, c = self.white_pos if self.turn == 0 else self.black_pos
        opponent_pos = self.black_pos if self.turn == 0 else self.white_pos
        moves = []

        for direction, (dr, dc) in self.possible_moves.items():
            nr, nc = r + dr, c + dc

            if not (0 <= nr < self.size and 0 <= nc < self.size):
                continue

            if self.is_wall_between(r, c, nr, nc):
                continue

            if (nr, nc) == opponent_pos:
                jump_r, jump_c = nr + dr, nc + dc

                if (0 <= jump_r < self.size and 0 <= jump_c < self.size and
                        not self.is_wall_between(nr, nc, jump_r, jump_c)):
                    moves.append((jump_r, jump_c))
                else:
                    for perp_dr, perp_dc in [(0, -1), (0, 1)] if dr != 0 else [(-1, 0), (1, 0)]:
                        diag_r, diag_c = nr + perp_dr, nc + perp_dc
                        if (0 <= diag_r < self.size and 0 <= diag_c < self.size and
                                not self.is_wall_between(nr, nc, diag_r, diag_c)):
                            moves.append((diag_r, diag_c))
            else:
                moves.append((nr, nc))

        return moves

    def is_wall_between(self, r1, c1, r2, c2):
        if r1 == r2:
            wall_row = r1
            wall_col = min(c1, c2)
            return (wall_row, wall_col) in self.v_walls
        else:
            wall_row = min(r1, r2)
            wall_col = c1
            return (wall_row, wall_col) in self.h_walls

    def place_wall(self, orientation, pos):
        if self.turn == 0 and self.white_walls > 0:
            if self.is_valid_wall(orientation, pos):
                if orientation == 'h':
                    self.h_walls.add(pos)
                else:
                    self.v_walls.add(pos)
                self.white_walls -= 1
        elif self.turn == 1 and self.black_walls > 0:
            if self.is_valid_wall(orientation, pos):
                if orientation == 'h':
                    self.h_walls.add(pos)
                else:
                    self.v_walls.add(pos)
                self.black_walls -= 1
        self.draw_board()
        self.turn = abs(self.turn - 1)

    def is_valid_wall(self, orientation, pos):
        r, c = pos

        if orientation == 'h':
            if r < 0 or r >= self.size - 1 or c < 0 or c >= self.size - 1:
                return False
            if (r, c) in self.h_walls:
                return False

            if (r, c) in self.v_walls or (r + 1, c) in self.v_walls:
                return False

            if (r, c - 1) in self.h_walls or (r, c + 1) in self.h_walls:
                return False

        else:
            if r < 0 or r >= self.size - 1 or c < 0 or c >= self.size - 1:
                return False

            if (r, c) in self.v_walls:
                return False

            if (r, c) in self.h_walls or (r, c + 1) in self.h_walls:
                return False

            if (r - 1, c) in self.v_walls or (r + 1, c) in self.v_walls:
                return False

        if orientation == 'h':
            self.h_walls.add(pos)
        else:
            self.v_walls.add(pos)

        white_can_reach = self._can_reach_goal(
            self.white_pos, 0)
        black_can_reach = self._can_reach_goal(
            self.black_pos, self.size - 1)

        if orientation == 'h':
            self.h_walls.remove(pos)
        else:
            self.v_walls.remove(pos)

        return white_can_reach and black_can_reach

    def _can_reach_goal(self, start_pos, goal_row):
        from collections import deque

        visited = set()
        queue = deque([start_pos])
        visited.add(start_pos)

        while queue:
            r, c = queue.popleft()

            if r == goal_row:
                return True

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc

                if not (0 <= nr < self.size and 0 <= nc < self.size):
                    continue

                if (nr, nc) in visited:
                    continue

                if self.is_wall_between(r, c, nr, nc):
                    continue

                visited.add((nr, nc))
                queue.append((nr, nc))

        return False

    def is_game_over(self):
        if self.white_pos[0] == 0:
            return "white"
        if self.black_pos[0] == self.size-1:
            return "black"
        return 0
