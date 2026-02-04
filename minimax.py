import math


def evaluate(env, player=0):
    goal_row = 0 if player == 0 else env.size - 1
    opp_row = env.size - 1 if player == 0 else 0
    my_pos = env.white_pos if player == 0 else env.black_pos
    opp_pos = env.black_pos if player == 0 else env.white_pos
    my_dist = env.shortest_path_length(my_pos, goal_row)
    opp_dist = env.shortest_path_length(opp_pos, opp_row)
    my_walls = env.white_walls if player == 0 else env.black_walls
    opp_walls = env.black_walls if player == 0 else env.white_walls
    return (opp_dist - my_dist) + 0.1 * (my_walls - opp_walls)


def minimax(env, depth, maximizing_player=True, alpha=-math.inf, beta=math.inf):
    winner = env.is_game_over()
    if winner == 'white':
        return (math.inf if maximizing_player else -math.inf), None
    if winner == 'black':
        return (-math.inf if maximizing_player else math.inf), None

    if depth == 0:
        return evaluate(env, 0 if maximizing_player else 1), None

    best_action = None
    if maximizing_player:
        max_eval = -math.inf
        for action in env.get_legal_moves() + get_legal_walls(env):
            backup = backup_env(env)
            apply_action(env, action)
            eval_score, _ = minimax(env, depth-1, False, alpha, beta)
            restore_env(env, backup)
            if eval_score > max_eval:
                max_eval = eval_score
                best_action = action
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_action
    else:
        min_eval = math.inf
        for action in env.get_legal_moves() + get_legal_walls(env):
            backup = backup_env(env)
            apply_action(env, action)
            eval_score, _ = minimax(env, depth-1, True, alpha, beta)
            restore_env(env, backup)
            if eval_score < min_eval:
                min_eval = eval_score
                best_action = action
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_action


def get_legal_walls(env):
    walls = []
    wall_count = env.white_walls if env.turn == 0 else env.black_walls
    if wall_count == 0:
        return []
    for r in range(env.size-1):
        for c in range(env.size-1):
            if env.is_valid_wall('h', (r, c)):
                walls.append(('h_wall', r, c))
            if env.is_valid_wall('v', (r, c)):
                walls.append(('v_wall', r, c))
    return walls


def backup_env(env):
    return {
        "white_pos": env.white_pos,
        "black_pos": env.black_pos,
        "h_walls": set(env.h_walls),
        "v_walls": set(env.v_walls),
        "white_walls": env.white_walls,
        "black_walls": env.black_walls,
        "turn": env.turn
    }


def restore_env(env, backup):
    env.white_pos = backup["white_pos"]
    env.black_pos = backup["black_pos"]
    env.h_walls = set(backup["h_walls"])
    env.v_walls = set(backup["v_walls"])
    env.white_walls = backup["white_walls"]
    env.black_walls = backup["black_walls"]
    env.turn = backup["turn"]


def apply_action(env, action):
    if action[0] == 'move':
        env.move_pawn(action[1:])
    elif action[0] in ['h_wall', 'v_wall']:
        env.place_wall('h' if action[0] == 'h_wall' else 'v', action[1:])
