import heapq
from collections import deque


class QuoridorEnv:
    def __init__(self, size=9, num_walls=10):
        self.size = size
        self.num_walls = num_walls
        self.reset()

    def reset(self):
        self.white_pos = (self.size - 1, self.size // 2)
        self.black_pos = (0, self.size // 2)
        self.h_walls = set()
        self.v_walls = set()
        self.white_walls = self.num_walls
        self.black_walls = self.num_walls
        self.turn = 0
        return self.get_state()

    def copy(self):
        new_env = QuoridorEnv(self.size, self.num_walls)
        new_env.white_pos = self.white_pos
        new_env.black_pos = self.black_pos
        new_env.h_walls = self.h_walls.copy()
        new_env.v_walls = self.v_walls.copy()
        new_env.white_walls = self.white_walls
        new_env.black_walls = self.black_walls
        new_env.turn = self.turn
        return new_env

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

    def is_wall_between(self, r1, c1, r2, c2):
        if r1 == r2:
            row, col = r1, min(c1, c2)
            return ((row, col) in self.v_walls or (row - 1, col) in self.v_walls)
        if c1 == c2:
            row, col = min(r1, r2), c1
            return ((row, col) in self.h_walls or (row, col - 1) in self.h_walls)
        return False

    def get_legal_moves_from(self, pos):
        r, c = pos
        opponent = self.black_pos if pos == self.white_pos else self.white_pos
        moves = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r+dr, c+dc
            if not (0 <= nr < self.size and 0 <= nc < self.size):
                continue
            if self.is_wall_between(r, c, nr, nc):
                continue
            if (nr, nc) != opponent:
                moves.append((nr, nc))
                continue
            jr, jc = nr+dr, nc+dc
            if 0 <= jr < self.size and 0 <= jc < self.size and not self.is_wall_between(nr, nc, jr, jc):
                moves.append((jr, jc))
            else:
                if dr != 0:
                    for dc2 in [-1, 1]:
                        diag = (nr, nc+dc2)
                        if 0 <= diag[1] < self.size and not self.is_wall_between(nr, nc, diag[0], diag[1]):
                            moves.append(diag)
                else:
                    for dr2 in [-1, 1]:
                        diag = (nr+dr2, nc)
                        if 0 <= diag[0] < self.size and not self.is_wall_between(nr, nc, diag[0], diag[1]):
                            moves.append(diag)
        return list(set(moves))

    def get_legal_moves(self):
        pos = self.white_pos if self.turn == 0 else self.black_pos
        return [('move', r, c) for r, c in self.get_legal_moves_from(pos)]

    def move_pawn(self, pos):
        if pos not in [m[1:] for m in self.get_legal_moves()]:
            return False
        if self.turn == 0:
            self.white_pos = pos
        else:
            self.black_pos = pos
        self.turn = 1-self.turn
        return True

    def is_valid_wall(self, orientation, pos):
        r, c = pos
        if r < 0 or c < 0 or r >= self.size-1 or c >= self.size-1:
            return False
        if orientation == 'h':
            if pos in self.h_walls or (r, c-1) in self.h_walls or (r, c+1) in self.h_walls:
                return False
            if (r, c) in self.v_walls or (r, c+1) in self.v_walls:
                return False
            self.h_walls.add(pos)
        else:
            if pos in self.v_walls or (r-1, c) in self.v_walls or (r+1, c) in self.v_walls:
                return False
            if (r, c) in self.h_walls or (r+1, c) in self.h_walls:
                return False
            self.v_walls.add(pos)
        ok = self._can_reach_goal(self.white_pos, 0) and self._can_reach_goal(
            self.black_pos, self.size-1)
        if orientation == 'h':
            self.h_walls.remove(pos)
        else:
            self.v_walls.remove(pos)
        return ok

    def place_wall(self, orientation, pos):
        if not self.is_valid_wall(orientation, pos):
            return False
        if orientation == 'h':
            self.h_walls.add(pos)
        else:
            self.v_walls.add(pos)
        if self.turn == 0:
            self.white_walls -= 1
        else:
            self.black_walls -= 1
        self.turn = 1-self.turn
        return True

    def remove_wall(self, orientation, pos):
        if orientation == 'h':
            self.h_walls.discard(pos)
        else:
            self.v_walls.discard(pos)

    def _can_reach_goal(self, start, goal_row):
        visited = {start}
        q = deque([start])
        while q:
            r, c = q.popleft()
            if r == goal_row:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in visited and not self.is_wall_between(r, c, nr, nc):
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return False

    def shortest_path_length(self, start, goal_row):
        from collections import deque
        visited = {start}
        q = deque([(start, 0)])
        while q:
            (r, c), dist = q.popleft()
            if r == goal_row:
                return dist
            for nr, nc in self.get_legal_moves_from((r, c)):
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append(((nr, nc), dist+1))
        return 100

    def is_game_over(self):
        if self.white_pos[0] == 0:
            return "white"
        if self.black_pos[0] == self.size-1:
            return "black"
        return None
