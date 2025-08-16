#!/usr/bin/env python3

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import re
import glob
import shutil
from collections import deque
import logging
from datetime import datetime
import argparse
import traceback
import multiprocessing as mp  # NEW: multiprocessing for parallel rollouts
import ray, numpy as np, torch, time, random

try:
    import yaml
except ImportError:
    print("PyYAML not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyYAML"])
    import yaml

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    import matplotlib.colors as mcolors
    
    # 配置中文字体支持
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Bitstream Vera Sans']
    plt.rcParams['axes.unicode_minus'] = False
    # 设置字体大小，避免中文显示问题
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("Matplotlib not found. Installing for visualization...")
    try:
        import subprocess
        subprocess.check_call(["uv", "add","matplotlib"])
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.animation import FuncAnimation
        import matplotlib.colors as mcolors
        VISUALIZATION_AVAILABLE = True
    except Exception:
        print("⚠️ 無法安裝matplotlib，將跳過可視化功能")
        VISUALIZATION_AVAILABLE = False

from kaggle_environments import make, evaluate

# Set up logging with proper format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('training.log')  # File output
    ]
)
logger = logging.getLogger(__name__)

# 確保logger訊息不被抑制
logger.setLevel(logging.INFO)

# ===== Shared Tactical Functions =====
# These functions are used by both worker processes and the main trainer

def flat_to_2d(board_flat):
    """Convert flat board to 2D grid format."""
    try:
        if isinstance(board_flat[0], list):
            return board_flat
    except Exception:
        pass
    return [list(board_flat[r*7:(r+1)*7]) for r in range(6)]

def find_drop_row(grid, col):
    """Find the lowest empty row in a column."""
    for r in range(5, -1, -1):
        if grid[r][col] == 0:
            return r
    return None

def is_win_from(grid, r, c, mark):
    """Check if placing a mark at (r,c) creates a winning line."""
    dirs = [(1,0), (0,1), (1,1), (-1,1)]
    for dr, dc in dirs:
        cnt = 1
        for s in (1, -1):
            rr, cc = r + dr*s, c + dc*s
            while 0 <= rr < 6 and 0 <= cc < 7 and grid[rr][cc] == mark:
                cnt += 1
                rr += dr*s
                cc += dc*s
        if cnt >= 4:
            return True
    return False

def is_winning_move(board_flat, col, mark):
    """Check if a move in col is a winning move for mark."""
    grid = flat_to_2d(board_flat)
    if col < 0 or col > 6:
        return False
    r = find_drop_row(grid, col)
    if r is None:
        return False
    grid[r][col] = mark
    return is_win_from(grid, r, col, mark)

def apply_move(board_flat, col, mark):
    """Apply a move and return the new board state."""
    grid = flat_to_2d(board_flat)
    if col < 0 or col > 6:
        return None
    r = find_drop_row(grid, col)
    if r is None:
        return None
    grid[r][col] = mark
    return [grid[i][j] for i in range(6) for j in range(7)]

def if_i_can_win(board_flat, mark, agent):
    """Find a winning move for mark, if any."""
    for c in agent.get_valid_actions(board_flat):
        if is_winning_move(board_flat, c, mark):
            return c
    return None

def if_i_will_lose(board_flat, mark, agent):
    """Find a move to block opponent's immediate win."""
    opp = 3 - mark
    for c in agent.get_valid_actions(board_flat):
        if is_winning_move(board_flat, c, opp):
            return c
    return None

def if_i_will_lose_at_next(board_flat, move_col, mark, agent):
    """Check if making move_col gives opponent an immediate winning reply."""
    grid = flat_to_2d(board_flat)
    r = find_drop_row(grid, move_col) if 0 <= move_col <= 6 else None
    if r is None:
        return True
    grid[r][move_col] = mark
    opp = 3 - mark
    # Check if opponent has immediate winning reply
    new_board = [grid[i][j] for i in range(6) for j in range(7)]
    for c in agent.get_valid_actions(new_board):
        if is_winning_move(new_board, c, opp):
            return True
    return False

def safe_moves(board_flat, mark, valid_actions, agent):
    """Return moves that don't give opponent immediate win."""
    return [a for a in valid_actions if not if_i_will_lose_at_next(board_flat, a, mark, agent)]

# ===== Shared Opponent Strategies =====

def random_opponent_strategy(board_flat, mark, valid_actions, agent):
    """Random opponent with basic tactics."""
    c = if_i_can_win(board_flat, mark, agent)
    if c is not None:
        return c
    c = if_i_will_lose(board_flat, mark, agent)
    if c is not None:
        return c
    safe = safe_moves(board_flat, mark, valid_actions, agent)
    if safe:
        return random.choice(safe)
    return random.choice(valid_actions)

def minimax_opponent_strategy(board_flat, mark, valid_actions, agent, depth=3):
    """Minimax opponent implementation."""
    def score_window(window, mark):
        opp = 3 - mark
        cnt_self = window.count(mark)
        cnt_opp = window.count(opp)
        cnt_empty = window.count(0)
        if cnt_self > 0 and cnt_opp > 0:
            return 0
        score = 0
        if cnt_self == 4:
            score += 10000
        elif cnt_self == 3 and cnt_empty == 1:
            score += 100
        elif cnt_self == 2 and cnt_empty == 2:
            score += 10
        if cnt_opp == 3 and cnt_empty == 1:
            score -= 120
        return score

    def evaluate_board(board_flat, mark):
        grid = flat_to_2d(board_flat)
        score = 0
        # center preference
        center_col = [grid[r][3] for r in range(6)]
        score += center_col.count(mark) * 3
        # Horizontal
        for r in range(6):
            row = grid[r]
            for c in range(4):
                window = row[c:c+4]
                score += score_window(window, mark)
        # Vertical
        for c in range(7):
            col = [grid[r][c] for r in range(6)]
            for r in range(3):
                window = col[r:r+4]
                score += score_window(window, mark)
        # Diagonals
        for r in range(3):
            for c in range(4):
                window = [grid[r+i][c+i] for i in range(4)]
                score += score_window(window, mark)
        for r in range(3, 6):
            for c in range(4):
                window = [grid[r-i][c+i] for i in range(4)]
                score += score_window(window, mark)
        return score

    def has_winner(board_flat, mark):
        grid = flat_to_2d(board_flat)
        for r in range(6):
            for c in range(7):
                if grid[r][c] != mark:
                    continue
                if is_win_from(grid, r, c, mark):
                    return True
        return False

    def minimax(board_flat, depth, alpha, beta, current_mark, maximizing_mark):
        valid_moves = agent.get_valid_actions(board_flat)
        if depth == 0 or not valid_moves:
            return evaluate_board(board_flat, maximizing_mark), None
        
        best_move = None
        if current_mark == maximizing_mark:
            value = -float('inf')
            for c in valid_moves:
                nb = apply_move(board_flat, c, current_mark)
                if nb is None:
                    continue
                if has_winner(nb, current_mark):
                    return 1e6 - (5 - depth), c
                child_val, _ = minimax(nb, depth-1, alpha, beta, 3-current_mark, maximizing_mark)
                if child_val > value:
                    value, best_move = child_val, c
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value, best_move
        else:
            value = float('inf')
            for c in valid_moves:
                nb = apply_move(board_flat, c, current_mark)
                if nb is None:
                    continue
                if has_winner(nb, current_mark):
                    return -1e6 + (5 - depth), c
                child_val, _ = minimax(nb, depth-1, alpha, beta, 3-current_mark, maximizing_mark)
                if child_val < value:
                    value, best_move = child_val, c
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value, best_move

    # Check tactical moves first
    c = if_i_can_win(board_flat, mark, agent)
    if c is not None:
        return c
    c = if_i_will_lose(board_flat, mark, agent)
    if c is not None:
        return c
    
    # Use minimax
    safe = safe_moves(board_flat, mark, valid_actions, agent)
    moves = safe if safe else valid_actions
    best_score = -float('inf')
    best_move = random.choice(moves)
    
    for a in moves:
        nb = apply_move(board_flat, a, mark)
        if nb is None:
            continue
        score, _ = minimax(nb, depth-1, -float('inf'), float('inf'), 3-mark, mark)
        if score > best_score:
            best_score, best_move = score, a
    return best_move
# ===== 漸進式對手策略 (從弱到強) =====

def pure_random_opponent_strategy(board_flat, mark, valid_actions, agent):
    """純隨機對手，不使用任何戰術"""
    return random.choice(valid_actions)

def basic_win_only_opponent_strategy(board_flat, mark, valid_actions, agent):
    """只會贏棋的基礎對手，不會防守"""
    # 只檢查能否獲勝
    c = if_i_can_win(board_flat, mark, agent)
    if c is not None:
        return c
    # 其他情況純隨機
    return random.choice(valid_actions)

def defensive_only_opponent_strategy(board_flat, mark, valid_actions, agent):
    """只會防守的對手，不會主動獲勝"""
    # 只檢查是否需要阻止對手獲勝
    c = if_i_will_lose(board_flat, mark, agent)
    if c is not None:
        return c
    # 其他情況純隨機
    return random.choice(valid_actions)

def center_bias_opponent_strategy(board_flat, mark, valid_actions, agent):
    """偏好中央列的弱對手"""
    # 基本戰術：贏和防守
    c = if_i_can_win(board_flat, mark, agent)
    if c is not None:
        return c
    c = if_i_will_lose(board_flat, mark, agent)
    if c is not None:
        return c
    
    # 偏好中央列 (3, 4, 2, 5, 1, 6, 0)
    center_preference = [3, 4, 2, 5, 1, 6, 0]
    for col in center_preference:
        if col in valid_actions:
            return col
    return random.choice(valid_actions)

def weak_tactical_opponent_strategy(board_flat, mark, valid_actions, agent):
    """弱化的戰術對手，有50%機率忽略戰術"""
    # 50%機率使用戰術
    if random.random() < 0.5:
        c = if_i_can_win(board_flat, mark, agent)
        if c is not None:
            return c
        c = if_i_will_lose(board_flat, mark, agent)
        if c is not None:
            return c
    
    # 其他情況隨機選擇
    return random.choice(valid_actions)

def mistake_prone_opponent_strategy(board_flat, mark, valid_actions, agent):
    """容易犯錯的對手，會做出危險移動"""
    # 基本戰術
    c = if_i_can_win(board_flat, mark, agent)
    if c is not None:
        return c
    c = if_i_will_lose(board_flat, mark, agent)
    if c is not None:
        return c
    
    # 30%機率故意選擇危險移動
    if random.random() < 0.3:
        dangerous_moves = []
        for action in valid_actions:
            if if_i_will_lose_at_next(board_flat, action, mark, agent):
                dangerous_moves.append(action)
        if dangerous_moves:
            return random.choice(dangerous_moves)
    
    # 其他情況隨機
    return random.choice(valid_actions)

def self_play_opponent_strategy(board_flat, mark, valid_actions, agent):
    """Self-play opponent using current policy network."""
    state = agent.encode_state(board_flat, mark)
    action, _, _ = agent.select_action(state, valid_actions, training=False)
    return int(action)

_WORKER = {
    "agent": None,
    "env": None,
    "policy_version": -1,  # 尚未載入任何版本
}

def count_open_threats(board_flat, mark, threat_length):
    """Count open threat patterns of given length for a player.
    
    An 'open' threat means the pattern can potentially be extended to 4-in-a-row.
    For example: [0, mark, mark, 0] is an open 2-threat.
    """
    grid = flat_to_2d(board_flat)
    count = 0
    directions = [(1,0), (0,1), (1,1), (-1,1)]  # horizontal, vertical, diagonal
    
    for r in range(6):
        for c in range(7):
            for dr, dc in directions:
                # Check if we can fit a 4-length pattern starting from (r,c)
                if not (0 <= r + 3*dr < 6 and 0 <= c + 3*dc < 7):
                    continue
                
                # Extract 4-cell window
                window = []
                for i in range(4):
                    rr, cc = r + i*dr, c + i*dc
                    window.append(grid[rr][cc])
                
                # Count marks in window
                mark_count = window.count(mark)
                opp_count = window.count(3 - mark)
                empty_count = window.count(0)
                
                # Check if it's an open threat of desired length
                if mark_count == threat_length and opp_count == 0 and empty_count == (4 - threat_length):
                    count += 1
    
    return count

def count_center_control(board_flat, mark):
    """Count pieces in center and adjacent columns with weighted scoring."""
    grid = flat_to_2d(board_flat)
    center_count = sum(1 for r in range(6) if grid[r][3] == mark)  # center column
    adjacent_count = 0
    for col in [2, 4]:  # adjacent to center
        adjacent_count += sum(1 for r in range(6) if grid[r][col] == mark)
    return center_count, adjacent_count

def has_immediate_win(board_flat, mark):
    """Check if player has an immediate winning move available."""
    mock_agent = type('MockAgent', (), {
        'get_valid_actions': lambda self, board: [c for c in range(7) if flat_to_2d(board)[0][c] == 0]
    })()
    return if_i_can_win(board_flat, mark, mock_agent) is not None

def compute_potential_function(board_flat, mark, gamma=0.99):
    """Compute potential function Φ(s) for potential-based reward shaping.
    
    Φ(s) = 0.02·center_diff + 0.05·(my_open2 - opp_open2) + 0.12·(my_open3 - opp_open3)
           + 0.40·I(my_immediate_win) - 0.40·I(opp_immediate_win)
    """
    opp_mark = 3 - mark
    
    # Center control difference
    my_center, my_adjacent = count_center_control(board_flat, mark)
    opp_center, opp_adjacent = count_center_control(board_flat, opp_mark)
    center_diff = (my_center + 0.5 * my_adjacent) - (opp_center + 0.5 * opp_adjacent)
    
    # Open threat differences
    my_open2 = count_open_threats(board_flat, mark, 2)
    opp_open2 = count_open_threats(board_flat, opp_mark, 2)
    open2_diff = my_open2 - opp_open2
    
    my_open3 = count_open_threats(board_flat, mark, 3)
    opp_open3 = count_open_threats(board_flat, opp_mark, 3)
    open3_diff = my_open3 - opp_open3
    
    # Immediate win indicators
    my_immediate_win = 1.0 if has_immediate_win(board_flat, mark) else 0.0
    opp_immediate_win = 1.0 if has_immediate_win(board_flat, opp_mark) else 0.0
    
    # Combine into potential function
    potential = (0.02 * center_diff + 
                0.05 * open2_diff + 
                0.12 * open3_diff + 
                0.40 * my_immediate_win - 
                0.40 * opp_immediate_win)
    
    return potential

def calculate_custom_reward_global(prev_board, action, new_board, mark, valid_actions, game_over=False, winner=None, move_count=0, debug=False, gamma=0.99):
    """Enhanced reward function with potential-based shaping for PPO training.
    
    Combines sparse terminal rewards with dense tactical shaping using potential-based
    reward shaping to maintain policy invariance.
    
    Terminal rewards:
    - Win: +1.0 - 0.01 * move_count (prefer faster wins)
    - Loss: -1.0 + 0.01 * move_count (prefer prolonging when losing)  
    - Draw: -0.05 (push for decisive results)
    - Illegal move: -1.0
    
    Dense shaping via potential function:
    - Center control, open threats, immediate wins/losses
    - Applied as: γ·Φ(s') - Φ(s) to maintain optimality
    
    Step penalties:
    - Small time penalty: -0.001 per move
    - Block opponent win: +0.20
    - Create blunder: -0.30
    
    Args:
        prev_board: Board state before action (42-length list)
        action: Action taken (0-6)
        new_board: Board state after action (42-length list)  
        mark: Current player mark (1 or 2)
        valid_actions: Legal actions before move
        game_over: Whether game ended this step
        winner: Winner (1, 2, or None for draw)
        move_count: Current move number
        debug: Whether to print debug info
        gamma: Discount factor for potential shaping
        
    Returns:
        float: Total shaped reward clipped to [-1, 1]
    """
    # Start with sparse reward
    sparse_reward = 0.0
    opp_mark = 3 - mark
    
    # 1. Illegal move penalty
    if action not in valid_actions:
        if debug:
            print(f"[DEBUG] Illegal move penalty: action={action}, valid={valid_actions}")
        return -1.0  # Terminal penalty, no need for other calculations
    
    # 2. Terminal rewards (game over)
    if game_over:
        if winner == mark:
            # Win bonus, with preference for faster wins
            sparse_reward = 1.0 - 0.01 * move_count
            if debug:
                print(f"[DEBUG] Win reward: +{sparse_reward:.3f} (move {move_count})")
        elif winner == opp_mark:
            # Loss penalty, slightly less harsh for longer games (reward prolonging)
            sparse_reward = -1.0 + 0.01 * move_count
            if debug:
                print(f"[DEBUG] Loss penalty: {sparse_reward:.3f} (move {move_count})")
        else:
            # Draw - slightly negative to encourage decisive play
            sparse_reward = -0.05
            if debug:
                print(f"[DEBUG] Draw penalty: -0.05")
    
    # 3. Step-based tactical rewards (non-terminal)
    step_reward = 0.0
    
    # Small time penalty to prevent dithering
    step_reward -= 0.001
    
    # Check if we blocked an opponent's immediate win
    try:
        prev_grid = flat_to_2d(prev_board)
        prev_valid_actions = [c for c in range(7) if prev_grid[0][c] == 0]
        
        blocked_win = False
        for opp_action in prev_valid_actions:
            if is_winning_move(prev_board, opp_action, opp_mark) and opp_action == action:
                step_reward += 0.20
                blocked_win = True
                if debug:
                    print(f"[DEBUG] Blocked opponent win: +0.20 (column {action})")
                break
    except Exception:
        pass
    
    # Check if we created a blunder (opponent can win immediately after our move)
    try:
        if not game_over:
            new_grid = flat_to_2d(new_board)
            new_valid_actions = [c for c in range(7) if new_grid[0][c] == 0]
            
            for opp_action in new_valid_actions:
                if is_winning_move(new_board, opp_action, opp_mark):
                    step_reward -= 0.30
                    if debug:
                        print(f"[DEBUG] Created blunder: -0.30 (opponent can win at {opp_action})")
                    break
    except Exception:
        pass
    
    # 4. Potential-based shaping (the main tactical component)
    shaping_reward = 0.0
    try:
        phi_prev = compute_potential_function(prev_board, mark, gamma)
        phi_curr = compute_potential_function(new_board, mark, gamma)
        shaping_reward = gamma * phi_curr - phi_prev
        
        if debug:
            print(f"[DEBUG] Potential shaping: {shaping_reward:.3f} (Φ_prev={phi_prev:.3f}, Φ_curr={phi_curr:.3f})")
    except Exception as e:
        if debug:
            print(f"[DEBUG] Potential function failed: {e}")
        shaping_reward = 0.0
    
    # 5. Combine all components
    total_reward = sparse_reward + step_reward + shaping_reward
    
    # 6. Clip to reasonable range to maintain PPO stability
    total_reward = max(-1.0, min(1.0, total_reward))
    
    if debug:
        print(f"[DEBUG] Final reward: {total_reward:.3f} = sparse({sparse_reward:.3f}) + step({step_reward:.3f}) + shaping({shaping_reward:.3f})")
    
    return total_reward

def _worker_init_persistent(agent_cfg):
    # 關 GPU、限執行緒，避免 CPU oversubscription
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    # 初始化一次 Agent（CPU）與 Env
    agent = PPOAgent(agent_cfg)            # 你的類別
    agent.device = torch.device('cpu')
    agent.policy_net.to(agent.device)
    agent.policy_net.eval()

    from kaggle_environments import make    # 放這裡避免主進程 import 影響
    env = make('connectx', debug=False)

    _WORKER["agent"] = agent
    _WORKER["env"] = env
    _WORKER["policy_version"] = -1

def _worker_play_one(args):
    """
    單局對戰；持久化 agent/env，不重建。
    僅在收到更高 policy_version 且夾帶權重時才 load_state_dict。
    """
    try:
        agent = _WORKER["agent"]
        env = _WORKER["env"]

        # 權重更新（必要時）
        pv = int(args.get("policy_version", -1))
        weights_np = args.get("policy_state", None)  # 只有版本剛升時才會帶
        if pv > _WORKER["policy_version"] and weights_np is not None:
            state_dict = {k: torch.from_numpy(v.copy()) if isinstance(v, np.ndarray) else v
                          for k, v in weights_np.items()}
            agent.policy_net.load_state_dict(state_dict, strict=True)
            agent.policy_net.eval()
            _WORKER["policy_version"] = pv

        # 每局的隨機性
        seed = args.get('seed', None)
        if seed is not None:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        # 開局
        env.reset()
        
        # 獲取當前訓練進度來選擇對手難度
        current_win_rate = args.get('current_win_rate', 0.0)
        
        # 根據勝率選擇對手策略
        if current_win_rate < 0.2:
            # 初期：純隨機和只會贏的對手
            opponent_type = random.choice(['pure_random', 'win_only'])
        elif current_win_rate < 0.4:
            # 進階：加入只會防守的對手和戰術開局
            opponent_type = random.choice(['pure_random', 'win_only', 'defensive_only', 'tactical_opening'])
        elif current_win_rate < 0.55:
            # 中期：加入偏好中央和弱戰術對手，增加戰術開局比例
            opponent_type = random.choice(['win_only', 'defensive_only', 'center_bias', 'weak_tactical', 'tactical_opening', 'tactical_opening'])
        elif current_win_rate < 0.7:
            # 後期：加入容易犯錯的對手，強化戰術開局
            opponent_type = random.choice(['center_bias', 'weak_tactical', 'mistake_prone', 'tactical_opening', 'tactical_opening'])
        else:
            # 高階：使用完整戰術對手，主要使用戰術開局
            opponent_type = random.choice(['random', 'minimax', 'self', 'tactical_opening', 'tactical_opening', 'tactical_opening'])
        
        player2_prob = float(args.get('player2_training_prob', 0.5))
        
        # 特殊處理：如果使用戰術開局對手，強制對手為 player 1，訓練玩家為 player 2
        if opponent_type == 'tactical_opening':
            training_player = 2  # 訓練玩家強制為後手
        else:
            training_player = int(np.random.choice([1, 2], p=[1.0 - player2_prob, player2_prob]))

        transitions = []
        move_count, max_moves = 0, 50

        with torch.no_grad():
            while not env.done and move_count < max_moves:
                actions = []
                # 記錄本回合開始時的棋盤狀態
                round_start_board = None
                for player_idx in range(2):
                    if env.state[player_idx]['status'] == 'ACTIVE':
                        board, current_player = agent.extract_board_and_mark(env.state, player_idx)
                        valid_actions = agent.get_valid_actions(board)
                        
                        # 記錄訓練玩家動作前的棋盤狀態
                        if current_player == training_player:
                            round_start_board = board.copy()
                        
                        if current_player == training_player:
                            state = agent.encode_state(board, current_player)
                            action, prob, value = agent.select_action(state, valid_actions, training=True)
                            
                            # 記錄完整信息以便計算自定義獎勵
                            transition_data = {
                                'state': state,
                                'action': int(action),
                                'prob': float(prob),
                                'value': float(value),
                                'board_before': board.copy(),  # 動作前棋盤
                                'valid_actions': valid_actions.copy(),  # 合法動作
                                'mark': current_player,  # 玩家標記
                                'training_player': training_player,
                                'opponent_type': opponent_type,
                                'is_dangerous': bool(if_i_will_lose_at_next(board, int(action), current_player, agent)),
                                'move_index': move_count,  # 回合索引
                            }
                            transitions.append(transition_data)
                        else:
                            # 根據選定的對手類型選擇策略
                            if opponent_type == 'pure_random':
                                action = pure_random_opponent_strategy(board, current_player, valid_actions, agent)
                            elif opponent_type == 'win_only':
                                action = basic_win_only_opponent_strategy(board, current_player, valid_actions, agent)
                            elif opponent_type == 'defensive_only':
                                action = defensive_only_opponent_strategy(board, current_player, valid_actions, agent)
                            elif opponent_type == 'center_bias':
                                action = center_bias_opponent_strategy(board, current_player, valid_actions, agent)
                            elif opponent_type == 'weak_tactical':
                                action = weak_tactical_opponent_strategy(board, current_player, valid_actions, agent)
                            elif opponent_type == 'mistake_prone':
                                action = mistake_prone_opponent_strategy(board, current_player, valid_actions, agent)
                            elif opponent_type == 'tactical_opening':
                                # 戰術開局對手：強制先手並使用 3->4->2 開局，然後戰術邏輯
                                # 只有當此對手是 player 1 (mark==1) 時才使用開局策略
                                if current_player == 1:
                                    action = if_i_can_win(board, current_player, agent)
                                    if action is None:
                                        action = if_i_will_lose(board, current_player, agent)
                                    if action is None:
                                        # 檢查開局序列 3->4->2
                                        grid = flat_to_2d(board)
                                        my_tokens = sum(1 for r in range(6) for c in range(7) if grid[r][c] == 1)
                                        if my_tokens == 0 and 3 in valid_actions:
                                            action = 3
                                        elif my_tokens == 1 and 4 in valid_actions:
                                            action = 4
                                        elif my_tokens == 2 and 2 in valid_actions:
                                            action = 2
                                        else:
                                            # 開局完成後使用安全策略
                                            safe = safe_moves(board, current_player, valid_actions, agent)
                                            action = random.choice(safe) if safe else random.choice(valid_actions)
                                else:
                                    # 如果不是先手，使用普通戰術策略
                                    action = random_opponent_strategy(board, current_player, valid_actions, agent)
                            elif opponent_type == 'random':
                                action = random_opponent_strategy(board, current_player, valid_actions, agent)
                            elif opponent_type == 'minimax':
                                action = minimax_opponent_strategy(board, current_player, valid_actions, agent, depth=3)
                            else:  # self
                                action = self_play_opponent_strategy(board, current_player, valid_actions, agent)
                        actions.append(int(action))
                    else:
                        actions.append(0)
                
                # 執行動作
                try:
                    env.step(actions)
                except Exception:
                    break
                
                # 記錄動作後的棋盤狀態（針對訓練玩家的最後一個 transition）
                if transitions and round_start_board is not None:
                    try:
                        # 獲取動作後的棋盤狀態
                        post_board, _ = agent.extract_board_and_mark(env.state, 0)  # 使用player 0視角獲取棋盤
                        transitions[-1]['board_after'] = post_board.copy()
                    except Exception:
                        transitions[-1]['board_after'] = round_start_board.copy()  # 退化到動作前狀態
                
                move_count += 1

        try:
            player_result = env.state[0]['reward'] if training_player == 1 else env.state[1]['reward']
        except Exception:
            player_result = 0

        try:
            player_result = env.state[0]['reward'] if training_player == 1 else env.state[1]['reward']
        except Exception:
            player_result = 0

        # 計算自定義獎勵並分配給所有 transitions
        final_transitions = []
        for i, transition in enumerate(transitions):
            # 計算基礎環境獎勵
            base_reward = float(player_result)
            
            # 計算自定義獎勵（使用新的全局函數）
            custom_reward = 0.0
            if 'board_before' in transition and 'board_after' in transition:
                try:
                    board_before = transition['board_before']
                    board_after = transition['board_after']
                    action = transition['action']
                    valid_actions = transition['valid_actions']
                    mark = transition['mark']
                    move_index = transition['move_index']
                    total_moves = len(transitions)
                    
                    # 判斷遊戲是否在此步結束
                    is_last_move = (i == len(transitions) - 1)
                    game_over = is_last_move
                    winner = None
                    if game_over:
                        if player_result == 1:
                            winner = mark  # 我方勝利
                        elif player_result == -1:
                            winner = 3 - mark  # 對方勝利
                        # player_result == 0 時 winner 保持 None (平局)
                    
                    # 使用新的全局自定義獎勵函數
                    custom_reward = calculate_custom_reward_global(
                        prev_board=board_before,
                        action=action,
                        new_board=board_after,
                        mark=mark,
                        valid_actions=valid_actions,
                        game_over=game_over,
                        winner=winner,
                        move_count=i+1,
                        gamma=getattr(agent, 'gamma', 0.99),  # Use agent's gamma if available
                        debug=False  # Set to True for detailed reward logging
                    )
                    
                except Exception as e:
                    # 如果自定義獎勵計算失敗，使用默認值
                    custom_reward = 0.0
            
            # 合併基礎獎勵和自定義獎勵
            final_reward = base_reward + custom_reward
            
            # 為每個transition添加最終獎勵
            transition['reward'] = float(final_reward)
            transition['custom_reward'] = float(custom_reward)
            transition['base_reward'] = float(base_reward)
            final_transitions.append(transition)

        return {
            'transitions': final_transitions,
            'training_player': training_player,
            'player_result': int(player_result),
            'game_length': len(final_transitions),
            'opponent_type': opponent_type,
            'policy_version_used': _WORKER["policy_version"],
            'custom_rewards_applied': len([t for t in final_transitions if 'custom_reward' in t])
        }
    except Exception as e:
        return {
            'transitions': [],
            'training_player': 1,
            'player_result': 0,
            'game_length': 0,
            'opponent_type': 'unknown',
            'error': str(e),
            'policy_version_used': _WORKER["policy_version"],
        }


class DropPath(nn.Module):
    """Stochastic Depth per sample (when training only)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(max(0.0, drop_prob))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # shape: (N, 1, 1, 1) for broadcasting
        mask = torch.empty((x.shape[0], 1, 1, 1), dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask


class ConnectXNet(nn.Module):
    """Advanced ConnectX policy-value network.
    Enhancements:
      - Coordinate embedding (row/col planes)
      - Wider channel width (128)
      - Bottleneck residual blocks with Squeeze-and-Excitation (SE)
      - Stochastic Depth (DropPath) per block
      - Periodic lightweight spatial self-attention (every attn_every blocks)
      - Shared trunk then dual heads (policy softmax, value tanh)
    Input remains flat 126 (3x6x7); coord planes added internally (->5 channels) for compatibility.
    """

    def __init__(self, input_size=126, hidden_size=192, num_layers=256, drop_path_rate: float = 0.08, attn_every: int = 4):
        super().__init__()

        print(f'input_size: {input_size}, hidden_size: {hidden_size}')
        # Clamp depth
        max_blocks = 32
        blocks = int(num_layers) if isinstance(num_layers, int) else max_blocks
        if blocks > max_blocks:
            try:
                logger.warning(f"num_layers={blocks} 過深，縮減為 {max_blocks}")
            except Exception:
                pass
            blocks = max_blocks
        self.blocks = blocks
        # 調整channels以配合attention heads (24的倍數)
        # 24的最近倍數: 120, 144, 168, 192
        self.channels = 144  # 144 = 24 * 6，確保heads能整除
        self.drop_path_rate = float(max(0.0, drop_path_rate))
        # 由於增加了attention heads，可以更頻繁地使用attention
        self.attn_every = max(1, min(int(attn_every), 3))  # 限制在1-3之間，更頻繁的attention

        # Coordinate embedding (2,6,7)
        row = torch.linspace(-1, 1, steps=6).unsqueeze(1).repeat(1, 7)
        col = torch.linspace(-1, 1, steps=7).unsqueeze(0).repeat(6, 1)
        coord = torch.stack([row, col], dim=0)  # (2,6,7)
        self.register_buffer('coord_embed', coord)

        in_ch = 3 + 2  # original planes + coord planes
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, self.channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, self.channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.05),
        )

        # --- Building blocks ---
        class BottleneckSE(nn.Module):
            def __init__(self, c, drop_path=0.0, reduction=4):
                super().__init__()
                mid = max(32, c // 2)
                self.conv1 = nn.Conv2d(c, mid, 1, bias=False)
                self.gn1 = nn.GroupNorm(8, mid)
                self.conv2 = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
                self.gn2 = nn.GroupNorm(8, mid)
                self.conv3 = nn.Conv2d(mid, c, 1, bias=False)
                self.gn3 = nn.GroupNorm(8, c)
                # SE
                se_hidden = max(8, c // reduction)
                self.se_fc1 = nn.Linear(c, se_hidden)
                self.se_fc2 = nn.Linear(se_hidden, c)
                self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
                self.act = nn.ReLU(inplace=True)
            def forward(self, x):
                identity = x
                out = self.act(self.gn1(self.conv1(x)))
                out = self.act(self.gn2(self.conv2(out)))
                out = self.gn3(self.conv3(out))
                # Squeeze Excitation
                w = out.mean(dim=[2,3])          # (B,C)
                w = self.act(self.se_fc1(w))
                w = torch.sigmoid(self.se_fc2(w)).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
                out = out * w
                out = self.drop_path(out)
                out = self.act(out + identity)
                return out

        class SpatialSelfAttention(nn.Module):
            def __init__(self, c, heads=24):
                super().__init__()
                # 確保heads能整除c，如果不能就調整到最接近的因數
                if c % heads != 0:
                    # 找到最接近的因數
                    possible_heads = [i for i in range(1, c + 1) if c % i == 0]
                    heads = min(possible_heads, key=lambda x: abs(x - heads))
                    print(f"調整attention heads: {heads} (channels={c})")
                
                self.heads = heads
                self.head_dim = c // heads
                self.scale = self.head_dim ** -0.5
                self.qkv = nn.Conv2d(c, c * 3, 1, bias=False)
                self.proj = nn.Conv2d(c, c, 1, bias=False)
                
            def forward(self, x):  # x: (B,C,H,W)
                B, C, H, W = x.shape
                qkv = self.qkv(x).reshape(B, 3, self.heads, self.head_dim, H * W)
                q, k, v = qkv[:,0], qkv[:,1], qkv[:,2]  # (B,heads,head_dim,HW)
                attn = (q.transpose(-2,-1) @ k) * self.scale  # (B,heads,HW,HW)
                attn = torch.softmax(attn, dim=-1)
                out = (attn @ v.transpose(-2,-1)).transpose(-2,-1)  # (B,heads,head_dim,HW)
                out = out.reshape(B, C, H, W)
                return self.proj(out)

        # Assemble trunk
        dprates = torch.linspace(0, self.drop_path_rate, steps=self.blocks).tolist() if self.drop_path_rate > 0 else [0.0]*self.blocks
        self.trunk = nn.ModuleList()
        for i in range(self.blocks):
            self.trunk.append(BottleneckSE(self.channels, drop_path=dprates[i]))
            if self.attn_every > 0 and (i+1) % self.attn_every == 0:
                self.trunk.append(SpatialSelfAttention(self.channels))

        # Head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.channels * 6 * 7, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, 7),
            nn.Softmax(dim=-1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh(),
        )

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, 3, 6, 7)
        # concat coord embeddings
        coord = self.coord_embed.unsqueeze(0).expand(B, -1, -1, -1)
        x = torch.cat([x, coord], dim=1)  # (B,5,6,7)
        y = self.stem(x)
        for block in self.trunk:
            y = block(y)
        feat = self.head(y)
        policy = self.policy_head(feat)
        value = self.value_head(feat)
        return policy, value

class PPOAgent:
    """PPO 強化學習智能體"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 神經網路
        self.policy_net = ConnectXNet(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers']
        ).to(self.device)

        # 優化器
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # 學習率調度器 - 根據性能動態調整
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # 監控勝率最大化
            factor=0.7,  # 學習率降低因子
            patience=1000,  # 等待回合數
            min_lr=1e-8
        )

        # 訓練參數
        self.gamma = config['gamma']
        self.eps_clip = config['eps_clip']
        self.k_epochs = config['k_epochs']
        self.entropy_coef = config['entropy_coef']
        self.value_coef = config['value_coef']

        # 經驗緩衝區
        self.memory = deque(maxlen=config['buffer_size'])

        # --- Stuck detection / partial reset settings ---
        reset_cfg = config.get('reset', {}) if isinstance(config, dict) else {}
        from collections import deque as _dq
        self.entropy_window = _dq(maxlen=int(reset_cfg.get('entropy_window', 30)))
        self.entropy_threshold = float(reset_cfg.get('entropy_threshold', 0.9))  # ln(7)≈1.95, 0.9 is fairly sharp
        self.low_entropy_patience = int(reset_cfg.get('low_entropy_patience', 30))
        self.reset_cooldown_updates = int(reset_cfg.get('cooldown_updates', 200))
        self.partial_reset_fraction = float(reset_cfg.get('fraction', 0.25))  # reset fraction of residual blocks
        self.update_step = 0
        self.last_partial_reset_update = -10**9
        # Ensure numpy alias
        self._np = np

    # === Kaggle env helpers used by trainer ===
    def extract_board_and_mark(self, env_state, player_idx):
        """Return (board_flat_list, mark) from kaggle_environments state list.
        board: list[int] length 42, row-major 6x7, 0 empty, 1/2 marks.
        mark: 1 or 2 for the given player index.
        """
        try:
            obs = env_state[player_idx]['observation']
            board = obs.get('board') or obs.get('board_state')
            mark = obs.get('mark') or (1 if player_idx == 0 else 2)
            return list(board), int(mark)
        except Exception:
            # Fallback: try top-level
            try:
                board = env_state['board']
                mark = env_state.get('mark', 1)
                return list(board), int(mark)
            except Exception:
                return [0] * 42, 1

    def encode_state(self, board, mark):
        """Encode board (len=42) and mark (1/2) into 126-d float state (3x6x7).
        Planes: [my pieces, opp pieces, ones].
        """
        try:
            grid = [board[r * 7:(r + 1) * 7] for r in range(6)]
            my = np.zeros((6, 7), dtype=np.float32)
            opp = np.zeros((6, 7), dtype=np.float32)
            for r in range(6):
                for c in range(7):
                    v = grid[r][c]
                    if v == mark:
                        my[r, c] = 1.0
                    elif v != 0:
                        opp[r, c] = 1.0
            ones = np.ones((6, 7), dtype=np.float32)
            stacked = np.stack([my, opp, ones], axis=0)
            return stacked.reshape(-1)
        except Exception:
            return np.zeros(126, dtype=np.float32)

    def get_valid_actions(self, board):
        """Valid columns where top cell is empty."""
        try:
            grid_top = [board[c] for c in range(7)]  # row 0
            return [c for c in range(7) if grid_top[c] == 0]
        except Exception:
            # Generic fallback
            valid = []
            try:
                for c in range(7):
                    col_full = True
                    for r in range(6):
                        if board[r * 7 + c] == 0:
                            col_full = False
                            break
                    if not col_full:
                        valid.append(c)
            except Exception:
                valid = list(range(7))
            return valid

    def select_action(self, state, valid_actions, training=True, temperature=1.0, exploration_bonus=0.0):
        """Return (action, prob, value). Masks invalid actions and samples during training.
        temperature <= 0 or not training -> greedy.
        """
        self.policy_net.eval()  # eval OK for GroupNorm with small batch; sampling unaffected
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
            probs, value = self.policy_net(s)
            probs = probs.squeeze(0).clamp_min(1e-8)
            mask = torch.zeros_like(probs)
            if not valid_actions:
                valid_actions = list(range(7))
            mask[valid_actions] = 1.0
            masked = probs * mask
            if masked.sum() <= 0:
                # fallback to uniform over valid
                masked = mask / mask.sum()
            else:
                masked = masked / masked.sum()
            if training and temperature is not None and temperature > 1e-6:
                logits = torch.log(masked + 1e-8) / float(temperature)
                dist = torch.distributions.Categorical(logits=logits)
            else:
                dist = torch.distributions.Categorical(probs=masked)
            action = int(dist.sample().item())
            prob = float(masked[action].item())
            return action, prob, float(value.item())

    def get_action_scores(self, state, valid_actions):
        """取得目前策略對每個動作的打分 / 概率與 value。

        Args:
            state: 126/flattened state (或 list/np.array)
            valid_actions: 可落子列表

        Returns dict:
            {
              'valid_actions': [...],
              'raw_policy': list[7],          # 原始 policy_net softmax 輸出
              'masked_policy': list[7],       # 只在 valid 上重新正規化後的分佈
              'logits': list[7],              # 對應 masked_policy 的 log(prob)
              'value': float,                 # 狀態價值 V(s)
              'entropy': float,               # masked distribution entropy
              'action_ranking': [(action, prob), ...]  # 依 masked prob 由高到低
            }
        """
        self.policy_net.eval()
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
            probs, value = self.policy_net(s)  # probs: (1,7)
            probs = probs.squeeze(0).clamp_min(1e-8)
            if not valid_actions:
                valid_actions = list(range(7))
            mask = torch.zeros_like(probs)
            mask[valid_actions] = 1.0
            masked = probs * mask
            if masked.sum() <= 0:
                masked = mask / mask.sum()  # uniform on valid
            else:
                masked = masked / masked.sum()
            entropy = float(-(masked * (masked.add(1e-8).log())).sum().item())
            ranking = sorted([(int(a), float(masked[a].item())) for a in range(7)], key=lambda x: x[1], reverse=True)
            return {
                'valid_actions': list(map(int, valid_actions)),
                'raw_policy': [float(p) for p in probs.tolist()],
                'masked_policy': [float(p) for p in masked.tolist()],
                'logits': [float(math.log(p + 1e-8)) for p in masked.tolist()],
                'value': float(value.item()),
                'entropy': entropy,
                'action_ranking': ranking,
            }

    def debug_print_action_scores(self, board, mark):
        """便利函式：直接輸出某盤面下每個動作概率/排名。"""
        try:
            state = self.encode_state(board, mark)
        except Exception:
            return
        valid = self.get_valid_actions(board)
        info = self.get_action_scores(state, valid)
        logger.info(
            "[ActionScores] mark=%s value=%.4f entropy=%.3f ranking=%s", 
            mark, info['value'], info['entropy'], info['action_ranking']
        )
        return info

    def store_transition(self, state, action, prob, reward, done):
        self.memory.append((state, action, prob, reward, done))

    def compute_gae(self, rewards, values, dones, next_value, gamma=0.99, lam=0.95):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            v_next = values[t + 1] if t + 1 < T else next_value
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + gamma * v_next * mask - values[t]
            last_gae = delta + gamma * lam * mask * last_gae
            adv[t] = last_gae
        returns = adv + np.array(values[:T], dtype=np.float32)
        return adv.tolist(), returns.tolist()

    def partial_reset(self, which: str = 'res_blocks_and_head', fraction: float | None = None):
        """Reinitialize a subset of the network to escape local minima.
        which: 'res_blocks', 'head', 'res_blocks_and_head'
        fraction: portion of residual blocks to reset (default from config)
        """
        fraction = self.partial_reset_fraction if fraction is None else float(fraction)
        net: ConnectXNet = self.policy_net  # type: ignore
        reset_count = 0
        
        # Reset a random subset of trunk blocks
        if which in ('res_blocks', 'res_blocks_and_head'):
            trunk_blocks = list(net.trunk)
            import random as _rnd
            k = max(1, int(len(trunk_blocks) * max(0.0, min(1.0, fraction))))
            idxs = _rnd.sample(range(len(trunk_blocks)), k)
            for i in idxs:
                block = trunk_blocks[i]
                # Reinitialize parameters for this block
                for m in block.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.Linear):
                        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.GroupNorm):
                        if m.weight is not None:
                            nn.init.ones_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                reset_count += 1
        
        # Optionally reset head and policy head
        if which in ('head', 'res_blocks_and_head'):
            for m in net.head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            for m in net.policy_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            # Value head left intact to preserve value estimates
        
        try:
            logger.info(f"🔄 Partial reset applied: which={which}, fraction={fraction:.2f}, reset_blocks={reset_count}")
        except Exception:
            pass
        self.last_partial_reset_update = self.update_step
   
    def update_policy_from_batch(self, states, actions, old_action_probs, rewards, dones, is_weights):
        """
        使用給定的批次數據進行PPO更新，支持重要性加權（用於PER）
        
        Args:
            states: list[tensor or np], 會在內部stack到device
            actions: np[int64] shape [B]
            old_action_probs: np[float32] shape [B]
            rewards: np[float32] shape [B]
            dones: np[bool] shape [B]
            is_weights: torch.float32 shape [B]  # 重要性加權

        Returns:
            dict(total_loss=..., td_errors_abs=np.ndarray[B])
        """
        batch_size = len(states)
        
        # 1) 將states轉換為張量並移到device
        try:
            # 處理不同類型的states輸入
            if isinstance(states[0], np.ndarray):
                states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
            elif isinstance(states[0], torch.Tensor):
                states_tensor = torch.stack(states).to(self.device)
            else:
                # 假設是list of float lists
                states_tensor = torch.FloatTensor(states).to(self.device)
        except Exception:
            # 後備方案
            states_tensor = torch.FloatTensor(np.array(states)).to(self.device)

        # 2) 計算當前策略網路的值
        with torch.no_grad():
            _, values_tensor = self.policy_net(states_tensor)
            values = values_tensor.cpu().numpy().flatten()

        # 3) 計算優勢和回報使用GAE
        advantages, returns = self.compute_gae(
            rewards.tolist(), values.tolist(), dones.tolist(), 0,
            self.gamma, self.config['gae_lambda']
        )

        # 4) 正規化優勢
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 5) 轉換為張量
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_probs_tensor = torch.FloatTensor(old_action_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # 6) PPO更新多個epochs
        total_loss = 0.0
        entropy_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        
        for epoch in range(self.k_epochs):
            # 前向傳播
            new_probs, values = self.policy_net(states_tensor)

            # 計算動作概率比率
            new_action_probs = new_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
            ratio = new_action_probs / (old_probs_tensor + 1e-8)

            # PPO clip損失
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2)
            
            # 價值損失
            value_loss = nn.MSELoss(reduction='none')(values.squeeze(), returns_tensor)

            # 熵損失（鼓勵探索）
            entropy = -(new_probs * torch.log(new_probs + 1e-8)).sum(dim=1)
            
            # 應用重要性加權（PER的修正）
            weighted_policy_loss = (policy_loss * is_weights).mean()
            weighted_value_loss = (value_loss * is_weights).mean()
            weighted_entropy = (entropy * is_weights).mean()

            # 總損失
            loss = weighted_policy_loss + self.value_coef * weighted_value_loss - self.entropy_coef * weighted_entropy
            
            # 反向傳播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer.step()

            # 累積統計
            total_loss += loss.item()
            policy_loss_sum += weighted_policy_loss.item()
            value_loss_sum += weighted_value_loss.item()
            entropy_sum += weighted_entropy.item()

        # 7) 計算TD error絕對值作為新的priority
        with torch.no_grad():
            _, current_values = self.policy_net(states_tensor)
            current_values = current_values.squeeze().cpu().numpy()
            
            # 計算TD error: |r + γV(s') - V(s)|
            next_values = np.zeros_like(current_values)
            for i in range(batch_size):
                if not dones[i] and i < batch_size - 1:
                    next_values[i] = current_values[i + 1] if i + 1 < len(current_values) else 0
                else:
                    next_values[i] = 0
            
            td_errors = rewards + self.gamma * next_values * (1 - dones.astype(float)) - current_values
            td_errors_abs = np.abs(td_errors) + 1e-3

        # 8) 更新熵檢測
        self.update_step += 1
        avg_entropy = entropy_sum / max(1, self.k_epochs)
        self.entropy_window.append(avg_entropy)
        
        if len(self.entropy_window) >= max(5, self.entropy_window.maxlen or 5):
            try:
                mean_ent = float(np.mean(self.entropy_window))
                if mean_ent < self.entropy_threshold and (self.update_step - self.last_partial_reset_update) >= self.reset_cooldown_updates:
                    logger.info(f"🔁 低熵觸發部分重置: mean_entropy={mean_ent:.3f} < thr={self.entropy_threshold:.3f}")
                    self.partial_reset('res_blocks_and_head')
            except Exception as e:
                try:
                    logger.debug(f"entropy reset check failed: {e}")
                except Exception:
                    pass

        return {
            "total_loss": total_loss / self.k_epochs,
            "policy_loss": policy_loss_sum / self.k_epochs,
            "value_loss": value_loss_sum / self.k_epochs,
            "entropy": avg_entropy,
            "td_errors_abs": td_errors_abs
        }

    def update_policy(self, use_batch_method=False):
        """使用 PPO 更新策略
        
        Args:
            use_batch_method: 如果為True，使用update_policy_from_batch方法
        """
        # 釋放未使用的 CUDA 快取（減少碎片化）
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if len(self.memory) < self.config['min_batch_size']:
            return None

        # 準備訓練數據
        states = []
        actions = []
        old_probs = []
        rewards = []
        dones = []

        for transition in self.memory:
            state, action, prob, reward, done = transition
            states.append(state)
            actions.append(action)
            old_probs.append(prob)
            rewards.append(reward)
            dones.append(done)

        if use_batch_method:
            # 使用新的批次更新方法
            states_array = np.array(states)
            actions_array = np.array(actions, dtype=np.int64)
            old_probs_array = np.array(old_probs, dtype=np.float32)
            rewards_array = np.array(rewards, dtype=np.float32)
            dones_array = np.array(dones, dtype=bool)
            
            # 創建統一的重要性權重（傳統方法中都是1.0）
            is_weights = torch.ones(len(states), dtype=torch.float32, device=self.device)
            
            # 調用新的批次更新方法
            result = self.update_policy_from_batch(
                states=states_array,
                actions=actions_array,
                old_action_probs=old_probs_array,
                rewards=rewards_array,
                dones=dones_array,
                is_weights=is_weights
            )
            
            # 清空記憶體
            self.memory.clear()
            return result
        
        # 原始的更新方法
        # 計算所有狀態的價值
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        with torch.no_grad():
            _, values_tensor = self.policy_net(states_tensor)
            values = values_tensor.cpu().numpy().flatten()

        # 計算優勢和回報
        advantages, returns = self.compute_gae(
            rewards, values, dones, 0,
            self.gamma, self.config['gae_lambda']
        )

        # 正規化優勢
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 轉換為張量
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_probs_tensor = torch.FloatTensor(old_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # PPO 更新
        total_loss = 0
        entropy_sum = 0.0
        for _ in range(self.k_epochs):
            # 前向傳播
            new_probs, values = self.policy_net(states_tensor)

            # 計算比率
            new_action_probs = new_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
            ratio = new_action_probs / (old_probs_tensor + 1e-8)

            # PPO 損失
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()

            # 價值損失
            value_loss = nn.MSELoss()(values.squeeze(), returns_tensor)

            # 熵損失（探索）
            entropy = -(new_probs * torch.log(new_probs + 1e-8)).sum(dim=1).mean()
            entropy_sum += float(entropy.item())

            # 總損失
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            total_loss += loss.item()

            # 反向傳播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer.step()

        # 清空記憶體
        self.memory.clear()
        # 再次嘗試釋放快取，為下次更新準備
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # --- Stuck detection on entropy ---
        self.update_step += 1
        avg_entropy = entropy_sum / max(1, self.k_epochs)
        self.entropy_window.append(avg_entropy)
        if len(self.entropy_window) >= max(5, self.entropy_window.maxlen or 5):
            try:
                mean_ent = float(np.mean(self.entropy_window))
                if mean_ent < self.entropy_threshold and (self.update_step - self.last_partial_reset_update) >= self.reset_cooldown_updates:
                    logger.info(f"🔁 低熵觸發部分重置: mean_entropy={mean_ent:.3f} < thr={self.entropy_threshold:.3f}")
                    self.partial_reset('res_blocks_and_head')
            except Exception:
                pass
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': avg_entropy,
            'total_loss': total_loss / self.k_epochs
        }







@ray.remote(num_cpus=1)
class RolloutActor:
    def __init__(self, agent_cfg):
        # 隔離 CUDA + 限制 BLAS 執行緒
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

        # 初始化一次 Agent（CPU）與 Env
        self.agent = PPOAgent(agent_cfg)
        self.agent.device = torch.device('cpu')
        self.agent.policy_net.to(self.agent.device)
        self.agent.policy_net.eval()
        from kaggle_environments import make
        self.env = make('connectx', debug=False)
        self.policy_version = -1

    def set_weights(self, weights_np, version: int):
        # 只有版本提升才載入
        if version > self.policy_version and weights_np is not None:
            state_dict = {k: torch.from_numpy(v.copy()) if isinstance(v, np.ndarray) else v
                          for k, v in weights_np.items()}
            self.agent.policy_net.load_state_dict(state_dict, strict=True)
            self.agent.policy_net.eval()
            self.policy_version = version
        return self.policy_version

    def rollout(self, n_episodes: int, player2_prob: float, seed: int = None):
        if seed is not None:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        all_transitions = []
        meta = {"episodes": 0, "policy_version_used": self.policy_version, "wins":0,"losses":0,"draws":0}
        max_moves = 50

        with torch.no_grad():
            for _ in range(n_episodes):
                self.env.reset()
                opponent_type = random.choice(['random', 'minimax', 'self'])
                training_player = int(np.random.choice([1,2], p=[1.0 - player2_prob, player2_prob]))
                ep_transitions = []
                move_count = 0

                while not self.env.done and move_count < max_moves:
                    actions = []
                    for player_idx in range(2):
                        if self.env.state[player_idx]['status'] == 'ACTIVE':
                            board, current_player = self.agent.extract_board_and_mark(self.env.state, player_idx)
                            valid_actions = self.agent.get_valid_actions(board)
                            if current_player == training_player:
                                state = self.agent.encode_state(board, current_player)
                                action, prob, value = self.agent.select_action(state, valid_actions, training=True)
                                ep_transitions.append({
                                    'state': state,
                                    'action': int(action),
                                    'prob': float(prob),
                                    'value': float(value),
                                    'is_dangerous': bool(if_i_will_lose_at_next(board, int(action), current_player, self.agent)),
                                })
                            else:
                                if opponent_type == 'random':
                                    action = random_opponent_strategy(board, current_player, valid_actions, self.agent)
                                elif opponent_type == 'minimax':
                                    action = minimax_opponent_strategy(board, current_player, valid_actions, self.agent, depth=3)
                                else:
                                    action = self_play_opponent_strategy(board, current_player, valid_actions, self.agent)
                            actions.append(int(action))
                        else:
                            actions.append(0)
                    try:
                        self.env.step(actions)
                    except Exception:
                        break
                    move_count += 1

                # 回合結束：決定結果
                try:
                    r = self.env.state[0]['reward'] if training_player == 1 else self.env.state[1]['reward']
                except Exception:
                    r = 0
                if r == 1: meta["wins"] += 1
                elif r == -1: meta["losses"] += 1
                else: meta["draws"] += 1

                all_transitions.append((ep_transitions, int(r)))
                meta["episodes"] += 1

        return all_transitions, meta

class PERBuffer:
    def __init__(self, capacity=200_000, alpha=0.6, beta_start=0.4, beta_frames=200_000, danger_boost=2.0, eps=1e-3):
        """
        alpha: 決定優先度對抽樣機率的影響（0=均勻, 1=完全依賴優先度）
        beta: 重要性加權修正，從 beta_start 緩升到 1.0
        danger_boost: 危險步優先度乘子
        """
        self.capacity = capacity
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_frames = int(beta_frames)
        self.frame = 1
        self.eps = float(eps)
        self.danger_boost = float(danger_boost)

        self.data = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.is_danger = deque(maxlen=capacity)
        self.max_priority = 1.0

    def __len__(self):
        return len(self.data)

    def beta_by_frame(self):
        # 緩升到 1.0，避免一開始權重太重
        t = min(1.0, self.frame / max(1, self.beta_frames))
        return self.beta_start + t * (1.0 - self.beta_start)

    def add(self, transition, td_error_abs=None, is_dangerous=False):
        # 初始優先度：max_priority；若有 td_error_abs 用它
        p = float(td_error_abs) if td_error_abs is not None else self.max_priority
        if is_dangerous:
            p *= self.danger_boost
        self.data.append(transition)
        self.priorities.append(p)
        self.is_danger.append(bool(is_dangerous))
        self.max_priority = max(self.max_priority, p)

    def sample(self, batch_size, danger_fraction=0.25):
        n = len(self.data)
        if n == 0:
            return [], [], []

        # 機率分佈
        prios = np.array(self.priorities, dtype=np.float64)
        probs = prios ** self.alpha
        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= probs_sum

        # 危險子集
        danger_mask = np.array(self.is_danger, dtype=bool)
        k_danger = int(round(batch_size * danger_fraction))
        k_rest = batch_size - k_danger

        idxs = []
        if k_danger > 0 and danger_mask.any():
            probs_d = probs.copy()
            probs_d[~danger_mask] = 0.0
            s = probs_d.sum()
            probs_d = probs_d / s if s > 0 else np.ones_like(probs_d) / probs_d.size
            idxs_danger = np.random.choice(np.arange(n), size=min(k_danger, danger_mask.sum()), replace=False, p=probs_d)
            idxs.extend(idxs_danger.tolist())

        # 其餘從全體按機率抽（避免重覆）
        if k_rest > 0:
            probs_rest = probs.copy()
            if idxs:
                probs_rest[idxs] = 0.0
                s = probs_rest.sum()
                probs_rest = probs_rest / s if s > 0 else np.ones_like(probs_rest) / probs_rest.size
            idxs_rest = np.random.choice(np.arange(n), size=min(k_rest, n - len(idxs)), replace=False, p=probs_rest)
            idxs.extend(idxs_rest.tolist())

        # 重要性加權（IS weights）
        probs_used = probs[idxs]
        beta = self.beta_by_frame()
        weights = (n * probs_used) ** (-beta)
        weights = weights / (weights.max() + 1e-8)
        self.frame += 1

        batch = [self.data[i] for i in idxs]
        return batch, idxs, weights

    def update_priorities(self, idxs, td_errors_abs, is_dangerous_flags=None):
        for j, i in enumerate(idxs):
            p = float(td_errors_abs[j]) + self.eps
            if is_dangerous_flags is not None and is_dangerous_flags[j]:
                p *= self.danger_boost
            self.priorities[i] = p
            self.max_priority = max(self.max_priority, p)

class ConnectXTrainer:
    """ConnectX 訓練器"""

    def __init__(self, config_path_or_dict="config.yaml"):
        # 載入配置
        if isinstance(config_path_or_dict, dict):
            self.config = config_path_or_dict
        else:
            with open(config_path_or_dict, 'r') as f:
                self.config = yaml.safe_load(f)

        # 初始化智能體
        self.agent = PPOAgent(self.config['agent'])

        # 訓練統計
        self.episode_rewards = []
        self.win_rates = []
        self.training_losses = []
        
        # 新增：歷史模型保存（用於課程化學習）
        self.historical_models = []
        self.model_save_frequency = 1000  # 每1000回合保存一個歷史模型
        
        # 探索增強參數
        self.exploration_decay = 0.995
        self.min_exploration = 0.1
        self.current_exploration = 1.0

        # 訓練玩家選擇機率設定
        training_config = self.config.get('training', {})
        self.player2_training_prob = training_config.get('player2_training_probability', 0.7)  # 默認70%機率選擇player2
        # 強化後手訓練比例，提升後手勝率
        # boosted_p2 = max(self.player2_training_prob, 0.85)
        # if boosted_p2 != self.player2_training_prob:
        #     logger.info(f"提高後手訓練比例: {self.player2_training_prob:.2f} -> {boosted_p2:.2f}")
        #     self.player2_training_prob = boosted_p2
        
        # 持續學習數據
        self.continuous_learning_data = None
        self.continuous_learning_targets = None

        # 創建目錄
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        # --- 新增：停滯偵測狀態 ---
        from collections import deque as _dq
        self._stagnation_scores = _dq(maxlen=50)  # 最近評估分數緩衝
        self._best_score_seen = -1.0
        self._best_score_episode = 0
        self._last_stagnation_handle_ep = -10**9

    # --- Board helpers (6x7) - using shared functions ---
    def _flat_to_2d(self, board_flat):
        return flat_to_2d(board_flat)

    def _find_drop_row(self, grid, col):
        return find_drop_row(grid, col)

    def _apply_move(self, board_flat, col, mark):
        return apply_move(board_flat, col, mark)

    def _is_win_from(self, grid, r, c, mark):
        return is_win_from(grid, r, c, mark)

    def _is_winning_move(self, board_flat, col, mark):
        return is_winning_move(board_flat, col, mark)

    def _any_winning_moves(self, board_flat, mark):
        wins = []
        for c in self.agent.get_valid_actions(board_flat):
            if self._is_winning_move(board_flat, c, mark):
                wins.append(c)
        return wins

    # --- Tactical utilities - using shared functions ---
    def if_i_can_win(self, board_flat, mark):
        return if_i_can_win(board_flat, mark, self.agent)

    def if_i_will_lose(self, board_flat, mark):
        return if_i_will_lose(board_flat, mark, self.agent)

    def if_i_will_lose_at_next(self, board_flat, move_col, mark):
        return if_i_will_lose_at_next(board_flat, move_col, mark, self.agent)

    # --- Debug / Analysis ---
    def get_action_score_report(self, board_flat, mark):
        """回傳目前 PPO 對此盤面 (board_flat) 在 mark 視角下各動作分布與價值。
        Returns dict (同 PPOAgent.get_action_scores)。"""
        try:
            return self.agent.debug_print_action_scores(board_flat, mark)
        except Exception as e:
            logger.warning(f"取得動作分數報告失敗: {e}")
            return None

    # --- Custom Reward System ---
    def calculate_custom_reward(self, prev_board, action, new_board, mark, valid_actions, game_over=False, winner=None, move_count=0):
        """計算自定義獎勵/懲罰系統 - 統一調用全域函數
        
        這個方法現在是全域 calculate_custom_reward_global 函數的包裝器，
        確保所有獎勵計算邏輯統一，避免重複代碼。
        
        獎勵規則：
        1. 非法動作懲罰: -1.0
        2. 勝利獎勵: +2.0  
        3. 防守獎勵 (擋住對手即將勝利): +1.0
        4. 威脅懲罰 (忽略對手即將勝利): -0.3
        5. 錯過勝利機會懲罰: -0.5
        6. 拖時間獎勵: +0.01 * (回合數-20), 最多+0.2
        
        Args:
            prev_board: 動作前的棋盤狀態 (42-length list)
            action: 執行的動作 (0-6)
            new_board: 動作後的棋盤狀態 (42-length list)  
            mark: 當前玩家標記 (1 or 2)
            valid_actions: 動作前的合法動作列表
            game_over: 遊戲是否結束
            winner: 獲勝者 (1, 2, or None for draw)
            move_count: 當前回合數
            
        Returns:
            float: 總獎勵/懲罰值
        """
        # 調用全域函數進行統一計算
        total_reward = calculate_custom_reward_global(
            prev_board=prev_board,
            action=action,
            new_board=new_board,
            mark=mark,
            valid_actions=valid_actions,
            game_over=game_over,
            winner=winner,
            move_count=move_count,
            gamma=self.gamma
        )
        
        # 添加詳細的調試日誌（只在類方法中提供）
        if total_reward != 0.0:
            opp_mark = 3 - mark
            debug_info = []
            
            # 檢查各種獎勵/懲罰的具體原因
            if action not in valid_actions:
                debug_info.append(f"非法動作懲罰: action={action}, valid={valid_actions}")
            elif game_over and winner == mark:
                debug_info.append(f"勝利獎勵: +2.0")
            else:
                # 檢查防守獎勵
                try:
                    prev_grid = flat_to_2d(prev_board)
                    prev_valid_actions = [c for c in range(7) if prev_grid[0][c] == 0]
                    for opp_action in prev_valid_actions:
                        if is_winning_move(prev_board, opp_action, opp_mark) and opp_action == action:
                            debug_info.append(f"防守成功獎勵: +1.0 (擋住列{action})")
                            break
                except Exception:
                    pass
                
                # 檢查威脅懲罰
                try:
                    opponent_winning_moves = []
                    for opp_action in prev_valid_actions:
                        if is_winning_move(prev_board, opp_action, opp_mark):
                            opponent_winning_moves.append(opp_action)
                    if opponent_winning_moves and action not in opponent_winning_moves:
                        debug_info.append(f"忽略對手威脅懲罰: -0.3 (對手可勝利於{opponent_winning_moves}, 我選擇{action})")
                except Exception:
                    pass
                
                # 檢查錯過勝利機會懲罰
                try:
                    winning_move = self.if_i_can_win(prev_board, mark)
                    if winning_move is not None and action != winning_move:
                        debug_info.append(f"錯過勝利機會懲罰: -0.5 (可勝利於列{winning_move}, 卻選擇{action})")
                except Exception:
                    pass
                
                # 檢查拖時間獎勵
                if move_count > 20:
                    drag_reward = min(0.2, (move_count - 20) * 0.01)
                    debug_info.append(f"拖時間獎勵: +{drag_reward:.3f} (回合{move_count})")
            
            # 輸出調試信息
            for info in debug_info:
                logger.debug(info)
        
        return total_reward

    def apply_custom_rewards_to_transitions(self, transitions, game_result, move_count):
        """將自定義獎勵應用到 transition 序列中
        
        Args:
            transitions: list of dict, 每個包含 {'state', 'action', 'prob', 'board_before', 'board_after', 'mark', 'valid_actions'}
            game_result: 遊戲結果 (1=我方勝, -1=我方敗, 0=平局)
            move_count: 總回合數
            
        Returns:
            list: 更新後的 transitions，每個添加了 'custom_reward' 字段
        """
        if not transitions:
            return transitions
            
        updated_transitions = []
        for i, tr in enumerate(transitions):
            # 基本資訊
            prev_board = tr.get('board_before', [0]*42)
            action = tr.get('action', 0)
            new_board = tr.get('board_after', [0]*42)
            mark = tr.get('mark', 1)
            valid_actions = tr.get('valid_actions', list(range(7)))
            
            # 判斷遊戲是否在此步結束
            is_last_move = (i == len(transitions) - 1)
            game_over = is_last_move
            winner = None
            if game_over:
                if game_result == 1:
                    winner = mark  # 我方勝利
                elif game_result == -1:
                    winner = 3 - mark  # 對方勝利
                # game_result == 0 時 winner 保持 None (平局)
            
            # 計算自定義獎勵 - 使用實例方法版本
            custom_reward = self.calculate_custom_reward(
                prev_board=prev_board,
                action=action, 
                new_board=new_board,
                mark=mark,
                valid_actions=valid_actions,
                game_over=game_over,
                winner=winner,
                move_count=i+1
            )
            
            # 複製 transition 並添加自定義獎勵
            updated_tr = tr.copy()
            updated_tr['custom_reward'] = custom_reward
            updated_transitions.append(updated_tr)
            
        return updated_transitions

    # NEW: list all moves that do NOT give opponent an immediate winning reply
    def _safe_moves(self, board_flat, mark, valid_actions):
        return safe_moves(board_flat, mark, valid_actions, self.agent)

    # Tactical-aware random opponent move - using shared function
    def _random_with_tactics(self, board_flat, mark, valid_actions):
        return random_opponent_strategy(board_flat, mark, valid_actions, self.agent)

    # --- Random opening tactic for first player (mark==1) ---
    def _random_opening_move(self, board_flat, mark, valid_actions):
        """If this random player is the starter (mark==1), follow opening: 3 -> 4 -> 2 -> then 5 or 1 depending on top row empty.
        Returns a move or None if not applicable."""
        try:
            if int(mark) != 1:
                return None
            grid = self._flat_to_2d(board_flat)
            # Count own tokens on board to determine which turn for this player
            my_tokens = 0
            for r in range(6):
                for c in range(7):
                    if grid[r][c] == 1:
                        my_tokens += 1
            # Opening sequence by own move index (before placing this move)
            if my_tokens == 0 and 3 in valid_actions:
                return 3
            if my_tokens == 1 and 4 in valid_actions:
                return 4
            if my_tokens == 2 and 2 in valid_actions:
                return 2
            if my_tokens == 3:
                top5_empty = (grid[0][5] == 0)
                top1_empty = (grid[0][1] == 0)
                if top5_empty and 5 in valid_actions:
                    return 5
                if top1_empty and 1 in valid_actions:
                    return 1
            return None
        except Exception:
            return None

    def _random_with_opening(self, board_flat, mark, valid_actions):
        """Opening when starting; after choosing, ensure it is safe, else pick a safe move, else random."""
        move = self._random_opening_move(board_flat, mark, valid_actions)
        if move is not None:
            # If opening move would immediately allow opponent to win, try to pick a safe alternative
            if self.if_i_will_lose_at_next(board_flat, move, mark):
                safe = self._safe_moves(board_flat, mark, valid_actions)
                if safe:
                    return random.choice(safe)
                # no safe moves, keep original (or random)
                return random.choice(valid_actions)
            return move
        # No opening move; prefer safe moves
        safe = self._safe_moves(board_flat, mark, valid_actions)
        if safe:
            return random.choice(safe)
        return random.choice(valid_actions)

    # --- Unified tactical + opening random opponent - using shared functions ---
    def _tactical_random_opening_agent(self, board_flat, mark, valid_actions):
        """Priority: win -> block -> (if starter opening) -> safe move -> random.
        Adds safe-move layer to avoid handing immediate wins to opponent."""
        # Win if possible
        c = if_i_can_win(board_flat, mark, self.agent)
        if c is not None:
            return c
        # Block if necessary
        c = if_i_will_lose(board_flat, mark, self.agent)
        if c is not None:
            return c
        # If starter, try opening move (but reject if unsafe)
        move = self._random_opening_move(board_flat, mark, valid_actions)
        if move is not None and not if_i_will_lose_at_next(board_flat, move, mark, self.agent):
            return move
        # Safe moves filter
        safe = safe_moves(board_flat, mark, valid_actions, self.agent)
        if safe:
            return random.choice(safe)
        # Fallback to random
        return random.choice(valid_actions)

    def _play_one_game_vs_random(self):
        env = make('connectx', debug=False)
        env.reset()
        move_count = 0
        max_moves = 50
        with torch.no_grad():
            while not env.done and move_count < max_moves:
                actions = []
                for p in range(2):
                    if env.state[p]['status'] == 'ACTIVE':
                        board, mark = self.agent.extract_board_and_mark(env.state, p)
                        state = self.agent.encode_state(board, mark)
                        valid = self.agent.get_valid_actions(board)
                        # Force tactical random opponent to be player 0 (first/red player)
                        # Agent is always player 1 (second/yellow player)
                        if p == 0:
                            a = self._tactical_random_opening_agent(board, mark, valid)
                        else:
                            a, _, _ = self.agent.select_action(state, valid, training=False)
                        actions.append(int(a))
                    else:
                        actions.append(0)
                try:
                    env.step(actions)
                except Exception:
                    break
                move_count += 1
        # 回傳玩家2結果（我方，現在是後手）
        try:
            return 1 if env.state[1]['reward'] == 1 else (0 if env.state[1]['reward'] == 0 else -1)
        except Exception:
            return 0

    # --- Minimax opponent for evaluation ---
    def _choose_minimax_move(self, board_flat, mark, max_depth):
        # Use shared minimax strategy
        valid = self.agent.get_valid_actions(board_flat)
        return minimax_opponent_strategy(board_flat, mark, valid, self.agent, depth=max_depth)

    def evaluate_against_random(self, games: int = 50):
        wins = 0
        draws = 0
        total_games = max(1, int(games))
        
        for game_idx in range(total_games):
            try:
                r = self._play_one_game_vs_random()
                if r == 1:
                    wins += 1
                elif r == 0:
                    draws += 1
            except Exception as e:
                logger.warning(f"Game {game_idx} vs random failed: {e}")
                continue
                
        win_rate = wins / total_games
        logger.info(f"evaluate_against_random: {wins}/{total_games} wins, win_rate={win_rate:.3f}")
        return win_rate

    def evaluate_against_minimax(self, games: int = 20):
        wins = 0
        draws = 0
        total_games = max(1, int(games))
        
        for game_idx in range(total_games):
            try:
                env = make('connectx', debug=False)
                env.reset()
                moves = 0
                with torch.no_grad():
                    while not env.done and moves < 50:
                        actions = []
                        for p in range(2):
                            if env.state[p]['status'] == 'ACTIVE':
                                board, mark = self.agent.extract_board_and_mark(env.state, p)
                                state = self.agent.encode_state(board, mark)
                                valid = self.agent.get_valid_actions(board)
                                if p == 0:
                                    a, _, _ = self.agent.select_action(state, valid, training=False)
                                else:
                                    a = self._choose_minimax_move(board, mark, max_depth=int(self.config.get('evaluation', {}).get('minimax_depth', 3)))
                                actions.append(int(a))
                            else:
                                actions.append(0)
                        try:
                            env.step(actions)
                        except Exception:
                            break
                        moves += 1
                try:
                    res = env.state[0]['reward']
                    if res == 1:
                        wins += 1
                    elif res == 0:
                        draws += 1
                except Exception:
                    draws += 1
            except Exception as e:
                logger.warning(f"Game {game_idx} vs minimax failed: {e}")
                continue
                
        win_rate = wins / total_games
        logger.info(f"evaluate_against_minimax: {wins}/{total_games} wins, win_rate={win_rate:.3f}")
        return win_rate

    def evaluate_with_metrics(self, games: int = 50):
        wins = 0
        losses = 0
        draws = 0
        lengths = []
        for _ in range(max(1, int(games))):
            env = make('connectx', debug=False)
            env.reset()
            moves = 0
            with torch.no_grad():
                while not env.done and moves < 50:
                    actions = []
                    for p in range(2):
                        if env.state[p]['status'] == 'ACTIVE':
                            board, mark = self.agent.extract_board_and_mark(env.state, p)
                            state = self.agent.encode_state(board, mark)
                            valid = self.agent.get_valid_actions(board)
                            # Force tactical random opponent to be player 0 (first/red player)
                            # Agent is always player 1 (second/yellow player)
                            if p == 0:
                                a = self._tactical_random_opening_agent(board, mark, valid)
                            else:
                                a, _, _ = self.agent.select_action(state, valid, training=False)
                            actions.append(int(a))
                        else:
                            actions.append(0)
                    try:
                        env.step(actions)
                    except Exception:
                        break
                    moves += 1
            try:
                res = env.state[1]['reward']  # Check player 1 (agent) result
                if res == 1:
                    wins += 1
                elif res == -1:
                    losses += 1
                else:
                    draws += 1
            except Exception:
                draws += 1
            lengths.append(moves)
        total = max(1, int(games))
        win_rate = wins / total
        avg_len = float(np.mean(lengths)) if lengths else 0.0
        return {
            'win_rate': win_rate,
            'avg_game_length': avg_len,
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'quick_wins': 0,
            'comeback_wins': 0,
        }

    def _choose_policy_with_tactics(self, board_flat, mark, valid_actions, training=False):
        # 先看是否能直接贏
        c = if_i_can_win(board_flat, mark, self.agent)
        if c is not None:
            return c
        # 再看是否必須擋下對方
        c = if_i_will_lose(board_flat, mark, self.agent)
        if c is not None:
            return c
        # 用策略網路選擇
        state = self.agent.encode_state(board_flat, mark)
        a, _, _ = self.agent.select_action(state, valid_actions, training=training)
        a = int(a)
        # 避免立即送給對手致勝
        if if_i_will_lose_at_next(board_flat, a, mark, self.agent):
            safe = safe_moves(board_flat, mark, valid_actions, self.agent)
            if safe:
                return random.choice(safe)
        return a

    def _play_one_game_self_tactical(self, main_as_player1=True):
        """自我對戰：對手使用戰術規則(if_i_can_win/if_i_will_lose)，主控方用策略網路。
        main_as_player1 決定我方是先手還是後手，回傳我方勝/和/負 (1/0/-1)。"""
        env = make('connectx', debug=False)
        env.reset()
        moves = 0
        with torch.no_grad():
            while not env.done and moves < 50:
                actions = []
                for p in range(2):
                    if env.state[p]['status'] != 'ACTIVE':
                        actions.append(0)
                        continue
                    board, mark = self.agent.extract_board_and_mark(env.state, p)
                    valid = self.agent.get_valid_actions(board)
                    # 決定哪一方是我方
                    is_main = (p == 0 and main_as_player1) or (p == 1 and not main_as_player1)
                    if is_main:
                        # 我方用策略 + 安全檢查
                        a = self._choose_policy_with_tactics(board, mark, valid, training=False)
                    else:
                        # 對手使用合併版：戰術優先 + 先手開局 + 隨機
                        a = self._tactical_random_opening_agent(board, mark, valid)
                    actions.append(int(a))
                try:
                    env.step(actions)
                except Exception:
                    break
                moves += 1
        try:
            if main_as_player1:
                res = env.state[0]['reward']
            else:
                res = env.state[1]['reward']
            return 1 if res == 1 else (0 if res == 0 else -1)
        except Exception:
            return 0

    def evaluate_self_play_tactical(self, games: int = 50):
        wins = 0
        draws = 0
        for _ in range(max(1, int(games))):
            # Force agent to be player 1 (second player), tactical opponent always player 0
            r = self._play_one_game_self_tactical(main_as_player1=False)
            if r == 1:
                wins += 1
            elif r == 0:
                draws += 1
        return wins / max(1, int(games))
    def train_with_ray(self):
        tcfg = self.config.get('training', {})
        num_actors = int(tcfg.get('num_workers',  max(1, (os.cpu_count() or 2) - 1)))
        episodes_per_task = int(tcfg.get('episodes_per_update', 8))  # 每個 actor 一次打幾局
        max_episodes = int(tcfg.get('max_episodes', 100_000))
        eval_frequency = int(self.config.get('evaluation', {}).get('frequency', 200))
        eval_games = int(self.config.get('evaluation', {}).get('games', 30))
        win_scale = float(tcfg.get('win_reward_scaling', 1.0))
        loss_scale = float(tcfg.get('loss_penalty_scaling', 1.0))
        danger_scale = float(self.config.get('agent', {}).get('tactical_bonus', 0.1))
        visualize_every = int(tcfg.get('visualize_every', 100))

        # PER 超參數
        capacity = int(tcfg.get('replay_capacity', 300_000))
        alpha = float(tcfg.get('per_alpha', 0.6))
        beta_start = float(tcfg.get('per_beta_start', 0.4))
        beta_frames = int(tcfg.get('per_beta_frames', 200_000))
        danger_boost = float(tcfg.get('danger_priority_boost', 2.0))
        danger_fraction = float(tcfg.get('danger_oversample_fraction', 0.25))
        min_batch_size = int(self.agent.config.get('min_batch_size', 512))
        sgd_batch_size = int(self.agent.config.get('sgd_batch_size', 256))

        rng = random.Random()

        # 啟動 Ray
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                runtime_env={
                    'excludes': [
                        '.git/',
                        '/pycache/',
                        'checkpoints/',
                        'videos/',
                        'game_videos/',
                        'game_visualizations/**'
                    ]
                }
            )


        # 建 actors
        actors = [RolloutActor.remote(self.config['agent']) for _ in range(num_actors)]

        # 初始權重 -> 廣播
        def export_weights_numpy():
            return {k: v.detach().cpu().numpy() for k, v in self.agent.policy_net.state_dict().items()}

        policy_version = 0
        weights_np = export_weights_numpy()
        ray.get([a.set_weights.remote(weights_np, policy_version) for a in actors])

        # PER 緩衝
        per = PERBuffer(capacity=capacity, alpha=alpha, beta_start=beta_start, beta_frames=beta_frames, danger_boost=danger_boost)

        # 建立初始 in-flight 任務
        in_flight = []
        def submit(actor):
            return actor.rollout.remote(n_episodes=episodes_per_task, player2_prob=self.player2_training_prob, seed=rng.randrange(2**31-1))
        in_flight = [submit(a) for a in actors]

        episodes_done_total = len(self.episode_rewards)
        best_score = -1.0
        # Track recent win rate for difficulty reporting
        recent_results = []  # keep last ~1000 results (1=win, 0=non-win)
        current_win_rate = 0.0
        # Next scheduled checkpoints for evaluation/visualization (align to next multiple)
        if eval_frequency > 0:
            if episodes_done_total > 0:
                next_eval_at = ((episodes_done_total // eval_frequency) + 1) * eval_frequency
            else:
                next_eval_at = eval_frequency
        else:
            next_eval_at = None
        if visualize_every > 0:
            if episodes_done_total > 0:
                next_viz_at = ((episodes_done_total // visualize_every) + 1) * visualize_every
            else:
                next_viz_at = visualize_every
        else:
            next_viz_at = None

        logger.info(f"🚀 Ray Actor 平行訓練啟動：actors={num_actors}, episodes_per_task={episodes_per_task}")

        try:
            while episodes_done_total < max_episodes:
                # 等待任一 actor 完成
                done_ids, in_flight = ray.wait(in_flight, num_returns=1, timeout=1.0)
                if not done_ids:
                    continue

                # 立刻補件
                finished_id = done_ids[0]
                # 僅補一個新的任務，維持與 actors 數量一致，避免過量提交
                if len(in_flight) < len(actors):
                    try:
                        in_flight.append(submit(rng.choice(actors)))
                    except Exception:
                        in_flight.append(submit(actors[0]))

                # 取結果
                all_transitions, meta = ray.get(finished_id)
                # 逐回合寫入 PER
                for ep_transitions, result in all_transitions:
                    ep_reward_sum = 0.0
                    last_idx = len(ep_transitions) - 1
                    if last_idx < 0:
                        self.episode_rewards.append(0.0)
                        episodes_done_total += 1
                        continue

                    for idx, tr in enumerate(ep_transitions):
                        reward = 0.0
                        if idx == last_idx:
                            if result == 1:
                                reward += 1.0 * win_scale
                            elif result == -1:
                                reward += -1.0 * loss_scale
                        if tr.get('is_dangerous', False):
                            reward += -10.0 * danger_scale
                        ep_reward_sum += reward

                        transition = {
                            'state': tr['state'],
                            'action': tr['action'],
                            'prob': tr['prob'],
                            'reward': reward,
                            'done': (idx == last_idx),
                            'is_dangerous': tr.get('is_dangerous', False),
                        }

                        # 初始 priority：用 |reward| 當近似（等會兒更新後再覆蓋）
                        init_prio = abs(reward) + 1e-3
                        if transition['is_dangerous']:
                            init_prio *= danger_boost
                        per.add(transition, td_error_abs=init_prio, is_dangerous=transition['is_dangerous'])

                    self.episode_rewards.append(ep_reward_sum)
                    episodes_done_total += 1

                    # Update moving win-rate (requires at least minimal stability window)
                    try:
                        recent_results.append(1 if int(result) == 1 else 0)
                        if len(recent_results) > 1000:
                            recent_results.pop(0)
                        if len(recent_results) >= 100:
                            current_win_rate = sum(recent_results) / float(len(recent_results))
                    except Exception:
                        pass

                # 只要資料夠，就多次 SGD 更新（讓 Actor 不停打局）
                while len(per) >= min_batch_size:
                    batch, idxs, is_weights = per.sample(sgd_batch_size, danger_fraction=danger_fraction)
                    if not batch:
                        break

                    # 封裝給 agent（兩種選擇：1) 丟回 self.agent.memory 然後用你原有的 update；
                    # 2) 直接寫一個 update_policy_from_batch）
                    # 這裡示範 2)，用一個你可以在 agent 內實作的介面：
                    states = [b['state'] for b in batch]
                    actions = np.array([b['action'] for b in batch], dtype=np.int64)
                    old_probs = np.array([b['prob'] for b in batch], dtype=np.float32)
                    rewards = np.array([b['reward'] for b in batch], dtype=np.float32)
                    dones = np.array([b['done'] for b in batch], dtype=bool)
                    is_danger_flags = np.array([b['is_dangerous'] for b in batch], dtype=bool)
                    isw = torch.as_tensor(is_weights, dtype=torch.float32, device=self.agent.device)

                    # 你可以在 agent 內部把 states -> tensor、計算新 logprob / value、GAE、PPO loss 等
                    # 並回傳每筆樣本的 |TD-error| 或 |Advantage| 當 priority 更新依據
                    info = self.agent.update_policy_from_batch(
                        states=states,
                        actions=actions,
                        old_action_probs=old_probs,
                        rewards=rewards,
                        dones=dones,
                        is_weights=isw
                    )
                    # 期望 info 回來有 'td_errors_abs'（len==batch），沒有就用 |rewards| 代替
                    if info is not None:
                        self.training_losses.append(float(info.get('total_loss', 0.0)))

                    td_err = info.get('td_errors_abs') if info is not None else np.abs(rewards) + 1e-3
                    per.update_priorities(idxs, td_err, is_dangerous_flags=is_danger_flags)

                    # 成功更新後→廣播新權重（只在有實質更新時）
                    policy_version += 1
                    weights_np = {k: v.detach().cpu().numpy() for k, v in self.agent.policy_net.state_dict().items()}
                    # 只在版本升級點廣播一次（Ray 會並行 set_weights）
                    ray.get([a.set_weights.remote(weights_np, policy_version) for a in actors])

                # 週期性評估（以門檻觸發，避免模數對不齊而永不觸發）
                if next_eval_at is not None and episodes_done_total >= next_eval_at:
                    metrics = self.evaluate_comprehensive(games=eval_games)
                    score = float(metrics.get('comprehensive_score', 0.0))
                    score_incl = float(metrics.get('comprehensive_score_incl_self', score))
                    score_excl = float(metrics.get('comprehensive_score_excl_self', score))
                    use_self = bool(metrics.get('comprehensive_uses_self_play', False))
                    self.win_rates.append(score)
                    try:
                        self.agent.scheduler.step(score)
                    except Exception:
                        pass

                    # Difficulty label from moving win-rate
                    if current_win_rate < 0.2:
                        difficulty = "初級 (純隨機+只會贏)"
                    elif current_win_rate < 0.4:
                        difficulty = "初階 (加入防守)"
                    elif current_win_rate < 0.55:
                        difficulty = "中階 (偏好中央+弱戰術)"
                    elif current_win_rate < 0.7:
                        difficulty = "進階 (容易犯錯)"
                    else:
                        difficulty = "高階 (完整戰術)"

                    # Optional PPO diagnostics
                    ppo_diag = {}
                    try:
                        ppo_diag = self.evaluate_ppo_diagnostics(samples=max(3, eval_games // 10)) or {}
                    except Exception:
                        ppo_diag = {}

                    logger.info(
                        f"📈 Eps={episodes_done_total} | Score={score:.3f} (incl={score_incl:.3f}, excl={score_excl:.3f}, use_self={use_self}) | "
                        f"win_rate={current_win_rate:.3f} | diff={difficulty} | "
                        f"self={metrics.get('self_play', 0):.3f} minimax={metrics.get('vs_minimax', 0):.3f} rand={metrics.get('vs_random', 0):.3f} | "
                        f"ppo(entropy={ppo_diag.get('avg_entropy','n/a')}, value={ppo_diag.get('avg_value','n/a')})"
                    )

                    # Stagnation detection and handling (parity with train_parallel)
                    try:
                        if self._detect_convergence_stagnation(episodes_done_total, score):
                            self._handle_convergence_stagnation(episodes_done_total)
                    except Exception:
                        pass
                    if score > best_score:
                        best_score = score
                        send_telegram(f"🎉 新最佳分數 Eps={episodes_done_total} | Score={best_score:.3f} | ")
                        self.save_checkpoint(f"best_model_wr_{best_score:.3f}.pt")
                    # 更新下一次評估門檻
                    next_eval_at += eval_frequency
                # 週期性視覺化（同樣使用門檻觸發）
                if next_viz_at is not None and episodes_done_total >= next_viz_at:
                    try:
                        quick_games = max(5, int(eval_games // 2))
                        metrics_v = self.evaluate_comprehensive(games=quick_games)
                        score_v = float(metrics_v.get('comprehensive_score', 0.0))
                        logger.info(
                            f"🎯 視覺化評估 Eps={episodes_done_total} | Score={score_v:.3f} (incl={metrics_v.get('comprehensive_score_incl_self', score_v):.3f}, excl={metrics_v.get('comprehensive_score_excl_self', score_v):.3f}, use_self={metrics_v.get('comprehensive_uses_self_play', False)}) | "
                            f"self={metrics_v.get('self_play', 0):.3f} minimax={metrics_v.get('vs_minimax', 0):.3f} rand={metrics_v.get('vs_random', 0):.3f}"
                        )
                    except Exception as ee:
                        logger.warning(f"視覺化前評估失敗：{ee}")
                    try:
                        self.visualize_training_game(episodes_done_total, save_dir='videos', opponent='tactical', fps=2)
                    except Exception as ve:
                        logger.warning(f"可視覺化失敗：{ve}")
                    # 更新下一次視覺化門檻
                    next_viz_at += visualize_every

        finally:
            try:
                ray.shutdown()
            except Exception:
                pass

        logger.info("✅ Ray Actor 平行訓練完成")
        return self.agent

    def evaluate_self_play_player2_focus(self, games: int = 50):
        """評估我方作為後手(玩家2)時的勝率，對手採戰術規則。"""
        wins = 0
        draws = 0
        total_games = max(1, int(games))
        
        for game_idx in range(total_games):
            try:
                r = self._play_one_game_self_tactical(main_as_player1=False)
                if r == 1:
                    wins += 1
                elif r == 0:
                    draws += 1
            except Exception as e:
                logger.warning(f"Game {game_idx} self_play_player2 failed: {e}")
                continue
                
        win_rate = wins / total_games
        logger.info(f"evaluate_self_play_player2_focus: {wins}/{total_games} wins, win_rate={win_rate:.3f}")
        return win_rate

    def evaluate_ppo_diagnostics(self, samples: int = 5):
        """Quick PPO diagnostics: average entropy and value on random legal states.
        Returns dict with avg_entropy and avg_value. Best-effort only; safe on CPU.
        """
        try:
            import numpy as np
            ent_list, val_list = [], []
            self.agent.policy_net.eval()
            # Create a few empty or near-empty states to probe policy/value
            for _ in range(max(1, int(samples))):
                # start from empty, optionally add a couple random moves
                board = [0] * 42
                moves = np.random.randint(0, 4)
                mark = 1
                for _m in range(moves):
                    valid = [c for c in range(7) if board[c] == 0]
                    if not valid:
                        break
                    c = int(np.random.choice(valid))
                    # drop piece
                    grid = flat_to_2d(board)
                    r = find_drop_row(grid, c)
                    if r is None:
                        continue
                    grid[r][c] = mark
                    board = [grid[i][j] for i in range(6) for j in range(7)]
                    mark = 3 - mark

                # Evaluate on current mark's perspective
                state = self.agent.encode_state(board, mark)
                valid = self.agent.get_valid_actions(board)
                info = self.agent.get_action_scores(state, valid)
                if info is None:
                    continue
                ent_list.append(float(info.get('entropy', 0.0)))
                val_list.append(float(info.get('value', 0.0)))
            if not ent_list:
                return {'avg_entropy': None, 'avg_value': None}
            return {
                'avg_entropy': round(float(np.mean(ent_list)), 4),
                'avg_value': round(float(np.mean(val_list)), 4),
            }
        except Exception:
            return {'avg_entropy': None, 'avg_value': None}

    def evaluate_comprehensive(self, games: int = 50):
        try:
            vs_random = self.evaluate_against_random(games)
            logger.info(f"vs_random result: {vs_random}")
        except Exception as e:
            logger.warning(f"evaluate_against_random failed: {e}")
            vs_random = 0.0
            
        try:
            vs_minimax = self.evaluate_against_minimax(max(1, games // 2))
            logger.info(f"vs_minimax result: {vs_minimax}")
        except Exception as e:
            logger.warning(f"evaluate_against_minimax failed: {e}")
            vs_minimax = 0.0
            
        try:
            # 自我對戰分兩種：先手與後手焦點，這裡使用後手焦點以鼓勵後手能力
            self_play = self.evaluate_self_play_player2_focus(max(1, games // 2))
            logger.info(f"self_play result: {self_play}")
        except Exception as e:
            logger.warning(f"evaluate_self_play_player2_focus failed: {e}")
            self_play = 0.0
            
        # 權重（若配置有提供）
        eval_cfg = self.config.get('evaluation', {}) if isinstance(self.config, dict) else {}
        weights = eval_cfg.get('weights', {}) if isinstance(eval_cfg, dict) else {}
        w_self = float(weights.get('self_play', 0.4))
        w_minimax = float(weights.get('vs_minimax', 0.4))
        w_random = float(weights.get('vs_random', 0.2))

        # 是否在綜合評分中計入自對弈（預設：不計入，避免被 ≈50% 拉高或掩蓋）
        include_self_in_score = bool(eval_cfg.get('count_self_play_in_score', False))

        # 分別計算「包含自對弈」與「排除自對弈」兩種分數
        total_w_incl = max(1e-6, w_self + w_minimax + w_random)
        score_incl_self = (w_self * self_play + w_minimax * vs_minimax + w_random * vs_random) / total_w_incl

        # 排除自對弈：自對弈權重視為 0，並重新正規化
        total_w_excl = max(1e-6, (0.0) + w_minimax + w_random)
        score_excl_self = (w_minimax * vs_minimax + w_random * vs_random) / total_w_excl

        # 最終輸出遵循配置
        comprehensive = score_incl_self if include_self_in_score else score_excl_self
        
        return {
            'vs_random': vs_random,
            'vs_minimax': vs_minimax,
            'self_play': self_play,
            'comprehensive_score': comprehensive,
            'comprehensive_score_incl_self': score_incl_self,
            'comprehensive_score_excl_self': score_excl_self,
            'comprehensive_uses_self_play': include_self_in_score,
        }

    # --- Stagnation detection / handling ---
    def _detect_convergence_stagnation(self, current_episode: int, current_score: float) -> bool:
        """偵測是否進入收斂停滯狀態。
        條件(滿足任一核心條件 + 基本條件)：
          1. 最近窗口(score_window)的分數波動範圍過小 (range < range_threshold)
          2. 已經超過 patience_episodes 仍沒有刷新最佳分數
          3. 分數長期低於 low_score_threshold (平均值 < 該值 且 episode >= min_episode_start)
        觸發後由 _handle_convergence_stagnation 執行部分權重重置與探索增強。
        """
        cfg = self.config.get('stagnation', {}) if isinstance(self.config, dict) else {}
        window_size = int(cfg.get('window', 10))
        range_threshold = float(cfg.get('range_threshold', 0.02))
        patience_eps = int(cfg.get('patience_episodes', 800))
        low_score_threshold = float(cfg.get('low_score_threshold', 0.35))
        min_episode_start = int(cfg.get('min_episode_start', 500))
        cooldown_eps = int(cfg.get('cooldown_episodes', 800))

        # 更新狀態
        self._stagnation_scores.append(float(current_score))
        if current_score > self._best_score_seen + 1e-6:
            self._best_score_seen = float(current_score)
            self._best_score_episode = int(current_episode)

        # 冷卻未過不再觸發
        if current_episode - self._last_stagnation_handle_ep < cooldown_eps:
            return False

        if len(self._stagnation_scores) < max(5, window_size):
            return False

        recent = list(self._stagnation_scores)[-window_size:]
        r_min, r_max = min(recent), max(recent)
        r_range = r_max - r_min
        avg_recent = sum(recent) / len(recent)

        no_improve_eps = current_episode - self._best_score_episode

        cond_small_range = (r_range < range_threshold)
        cond_patience = (no_improve_eps >= patience_eps and current_episode >= min_episode_start)
        cond_low = (avg_recent < low_score_threshold and current_episode >= min_episode_start)

        triggered = (cond_small_range or cond_patience or cond_low)
        if triggered:
            try:
                logger.info(
                    f"⚠️ 偵測到可能停滯: eps={current_episode} score={current_score:.3f} "
                    f"range={r_range:.4f} avg={avg_recent:.3f} no_improve={no_improve_eps} "
                    f"(range<{range_threshold}? {cond_small_range}, patience? {cond_patience}, low? {cond_low})"
                )
            except Exception:
                pass
        return triggered

    def _handle_convergence_stagnation(self, current_episode: int):
        """處理收斂停滯：部分重置模型權重、提升探索、適度調整學習率/清緩衝。
        策略：
          1. 使用 agent.partial_reset 重置部分殘差區塊與 head (依 reset_fraction)
          2. 提升 entropy_coef（上限保護）鼓勵探索
          3. 可選：輕微調整學習率(降低或回彈)；這裡採用 *0.9 讓重新搜索更穩定
          4. 清空 PPO 記憶避免舊策略分佈干擾
          5. 重置對應 optimizer state 以防遺留動量
        """
        cfg = self.config.get('stagnation', {}) if isinstance(self.config, dict) else {}
        cooldown_eps = int(cfg.get('cooldown_episodes', 800))
        reset_fraction = float(cfg.get('reset_fraction', 0.30))
        entropy_boost = float(cfg.get('entropy_boost', 1.35))
        entropy_max = float(cfg.get('entropy_max', 0.02))  # entropy_coef 通常很小，設定上限
        lr_decay = float(cfg.get('lr_decay_factor', 0.9))
        
        min_gap_ok = (current_episode - self._last_stagnation_handle_ep) >= cooldown_eps
        if not min_gap_ok:
            return
        
        try:
            logger.info(
                f"🛠️ 執行停滯處理: episode={current_episode} reset_fraction={reset_fraction} entropy_boost={entropy_boost}"
            )
        except Exception:
            pass
        
        # 1) 部分重置（若 agent 提供 partial_reset）
        try:
            if hasattr(self.agent, 'partial_reset'):
                self.agent.partial_reset('res_blocks_and_head', fraction=reset_fraction)
            else:
                # 後備：手動挑選部分層重新初始化
                import random
                import math
                modules = []
                for m in self.agent.policy_net.modules():  # type: ignore
                    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)) and m.weight.requires_grad:
                        modules.append(m)
                random.shuffle(modules)
                k = max(1, int(len(modules) * reset_fraction))
                for m in modules[:k]:
                    if isinstance(m, torch.nn.Conv2d):
                        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    else:
                        torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
        except Exception as e:
            try:
                logger.warning(f"部分重置過程出錯: {e}")
            except Exception:
                pass
        
        # 2) 增加探索: entropy_coef 乘以 boost (有上限)
        try:
            if hasattr(self.agent, 'entropy_coef'):
                new_entropy = min(entropy_max, getattr(self.agent, 'entropy_coef', 0.0) * entropy_boost + 1e-12)
                self.agent.entropy_coef = new_entropy
                logger.info(f"🔄 entropy_coef -> {new_entropy:.6f}")
        except Exception:
            pass
        
        # 3) 調整學習率（全部 param group）
        try:
            for pg in self.agent.optimizer.param_groups:  # type: ignore
                old_lr = pg.get('lr', 0.0)
                pg['lr'] = max(1e-6, old_lr * lr_decay)
            logger.info("📉 已調整學習率 (乘以 lr_decay_factor)")
        except Exception:
            pass
        
        # 4) 清空記憶 / 5) 清 optimizer state (只保留必要結構)
        try:
            if hasattr(self.agent, 'memory'):
                self.agent.memory.clear()
            # 重建 optimizer 來清動量
            opt_cls = type(self.agent.optimizer)
            self.agent.optimizer = opt_cls(self.agent.policy_net.parameters(), **self.agent.optimizer.defaults)  # type: ignore
        except Exception:
            pass
        
        # 6) 更新冷卻標記
        self._last_stagnation_handle_ep = int(current_episode)
        try:
            logger.info("✅ 停滯處理完成，進入冷卻階段。")
        except Exception:
            pass
        return

    # ------------------------------
    # Checkpoint I/O utilities
    # ------------------------------
    def save_checkpoint(self, filename: str):
        """保存檢查點到 checkpoints/ 目錄"""
        try:
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint = {
                'model_state_dict': self.agent.policy_net.state_dict(),
                'optimizer_state_dict': self.agent.optimizer.state_dict(),
                'episode_rewards': self.episode_rewards,
                'win_rates': self.win_rates,
                'config': self.config,
                'model_architecture': {
                    'input_size': self.config['agent']['input_size'],
                    'hidden_size': self.config['agent']['hidden_size'],
                    'num_layers': self.config['agent']['num_layers'],
                },
                'save_timestamp': datetime.now().isoformat(),
                'pytorch_version': str(torch.__version__),
            }
            path = f"checkpoints/{filename}"
            tmp_path = path + ".tmp"
            # 先寫到臨時檔再原子替換，避免產生半寫入的壞檔
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, path)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            logger.info(f"✅ 已保存檢查點: {path} ({size_mb:.2f} MB)")
        except Exception as e:
            logger.error(f"保存檢查點失敗: {e}")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """載入檢查點並恢復訓練狀態"""
        try:
            if not os.path.exists(checkpoint_path):
                logger.error(f"檢查點不存在: {checkpoint_path}")
                return False

            logger.info(f"載入檢查點: {checkpoint_path}")
            # 嘗試標準載入
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.agent.device)
            except Exception as e:
                logger.warning(f"標準載入失敗，嘗試只載入權重: {e}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # 解析權重
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # 兼容直接保存state_dict的情況
                state_dict = checkpoint if isinstance(checkpoint, dict) else {}

            missing, unexpected = self.agent.policy_net.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning(f"載入時缺少鍵(前若干): {missing[:5]}")
            if unexpected:
                logger.warning(f"載入時未使用鍵(前若干): {unexpected[:5]}")
            logger.info("✅ 模型權重載入成功")

            # 優化器
            if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
                try:
                    self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.info("✅ 優化器狀態載入成功")
                except Exception as e:
                    logger.warning(f"優化器狀態載入失敗: {e}")

            # 訓練歷史
            if isinstance(checkpoint, dict):
                self.episode_rewards = checkpoint.get('episode_rewards', self.episode_rewards)
                self.win_rates = checkpoint.get('win_rates', self.win_rates)

            logger.info("🎉 檢查點載入完成！")
            return True
        except Exception as e:
            logger.error(f"載入檢查點時出錯: {e}")
            return False

    # =========================
    # [CHANGED] 主訓練流程
    # =========================

    def train_parallel(self):
        cfg_t = self.config.get('training', {})
        num_workers = int(cfg_t.get('num_workers', max(1, (mp.cpu_count() or 2) - 1)))
        episodes_per_update = int(cfg_t.get('episodes_per_update', 16))
        max_episodes = int(cfg_t.get('max_episodes', 100000))
        eval_frequency = int(self.config.get('evaluation', {}).get('frequency', 200))
        eval_games = int(self.config.get('evaluation', {}).get('games', 30))
        win_scale = float(cfg_t.get('win_reward_scaling', 1.0))
        loss_scale = float(cfg_t.get('loss_penalty_scaling', 1.0))
        danger_scale = float(self.config.get('agent', {}).get('tactical_bonus', 0.1))
        min_batch_size = int(self.agent.config.get('min_batch_size', 512))
        visualize_every = int(cfg_t.get('visualize_every', 100))

        inflight_multiplier = int(cfg_t.get('inflight_multiplier', 2))
        target_inflight = max(1, num_workers * inflight_multiplier)

        try:
            mp_ctx = mp.get_context('spawn')
        except ValueError:
            mp_ctx = mp

        pool = mp_ctx.Pool(
            processes=num_workers,
            initializer=_worker_init_persistent,
            initargs=(self.config['agent'],)
        )

        rng = random.Random()
        best_score = -1.0
        best_win_rate = -1.0
        episodes_done_total = len(self.episode_rewards)
        current_win_rate = 0.0  # 追蹤當前勝率

        # 版本控制
        policy_version = 0
        pending_weight_tasks = 0

        def _make_args(send_weights: bool):
            weights_np = None
            if send_weights:
                weights_np = {k: v.detach().cpu().numpy()
                            for k, v in self.agent.policy_net.state_dict().items()}
            return {
                'policy_version': policy_version,
                'policy_state': weights_np if send_weights else None,
                'player2_training_prob': self.player2_training_prob,
                'current_win_rate': current_win_rate,  # 傳遞當前勝率
                'seed': rng.randrange(2**31 - 1),
            }

        # 建立初始 in-flight 任務
        in_flight = []
        pending_weight_tasks = num_workers
        for _ in range(target_inflight):
            send_w = pending_weight_tasks > 0
            if send_w:
                pending_weight_tasks -= 1
            ar = pool.apply_async(_worker_play_one, (_make_args(send_w),))
            in_flight.append(ar)

        logger.info(f"🚀 漸進式難度訓練開始：workers={num_workers}, target_inflight={target_inflight}")

        steps_since_update = 0
        recent_results = []  # 追蹤最近的對戰結果
        
        try:
            while episodes_done_total < max_episodes:
                # 輕量輪詢完成的任務
                i = 0
                while i < len(in_flight):
                    ar = in_flight[i]
                    if ar.ready():
                        # 取結果並處理
                        res = ar.get()
                        # 從 in_flight 移除
                        in_flight[i] = in_flight[-1]
                        in_flight.pop()
                        # 立刻補上一個新任務
                        send_w = pending_weight_tasks > 0
                        if send_w:
                            pending_weight_tasks -= 1
                        in_flight.append(pool.apply_async(_worker_play_one, (_make_args(send_w),)))

                        # === consume result ===
                        err_msg = res.get('error') if isinstance(res, dict) else None
                        transitions = res.get('transitions', []) if isinstance(res, dict) else []
                        if not transitions:
                            self.episode_rewards.append(0.0)
                            recent_results.append(0)  # 記錄平局
                            if err_msg:
                                logger.debug(f"worker episode returned empty transitions: {err_msg}")
                        else:
                            player_result = int(res.get('player_result', 0))
                            recent_results.append(1 if player_result == 1 else 0)  # 記錄勝敗
                            
                            # 只保留最近1000局的結果來計算勝率
                            if len(recent_results) > 1000:
                                recent_results.pop(0)
                            
                            # 更新當前勝率
                            if len(recent_results) >= 100:  # 至少100局後才開始計算
                                current_win_rate = sum(recent_results) / len(recent_results)
                            
                            ep_reward_sum = 0.0
                            for idx, tr in enumerate(transitions):
                                state = tr['state']
                                action = int(tr['action'])
                                prob = float(tr['prob'])
                                # 使用計算好的最終獎勵（包含自定義獎勵）
                                reward = tr.get('reward', 0.0)
                                
                                # 如果需要額外的危險行為懲罰，可以添加
                                if tr.get('is_dangerous', False):
                                    reward += -10.0 * danger_scale
                                    
                                ep_reward_sum += reward
                                done = (idx == len(transitions) - 1)
                                self.agent.store_transition(state, action, prob, reward, done)
                                steps_since_update += 1
                            self.episode_rewards.append(ep_reward_sum)

                            # 更新策略
                            if steps_since_update >= min_batch_size:
                                info = self.agent.update_policy()
                                if info is not None:
                                    self.training_losses.append(info.get('total_loss', 0.0))
                                    policy_version += 1
                                    pending_weight_tasks += num_workers
                                steps_since_update = 0

                        episodes_done_total += 1

                        # 週期性評估
                        if eval_frequency > 0 and episodes_done_total % eval_frequency == 0:
                            metrics = self.evaluate_comprehensive(games=eval_games)
                            score = float(metrics.get('comprehensive_score'))

                            self.win_rates.append(score)
                            try:
                                self.agent.scheduler.step(score)
                            except Exception:
                                pass
                            
                            # 顯示當前對手難度等級
                            if current_win_rate < 0.2:
                                difficulty = "初級 (純隨機+只會贏)"
                            elif current_win_rate < 0.4:
                                difficulty = "初階 (加入防守)"
                            elif current_win_rate < 0.55:
                                difficulty = "中階 (偏好中央+弱戰術)"
                            elif current_win_rate < 0.7:
                                difficulty = "進階 (容易犯錯)"
                            else:
                                difficulty = "高階 (完整戰術)"
                            
                            logger.info(
                                f"📈 Eps={episodes_done_total} | Score={score:.3f} | 當前勝率={current_win_rate:.3f} | "
                                f"對手難度={difficulty} | self={metrics.get('self_play', 0):.3f} minimax={metrics.get('vs_minimax', 0):.3f} rand={metrics.get('vs_random', 0):.3f}"
                            )
                            
                            try:
                                if self._detect_convergence_stagnation(episodes_done_total, score):
                                    self._handle_convergence_stagnation(episodes_done_total)
                            except Exception:
                                pass
                        
                            if score > best_score:
                                best_score = score
                                send_telegram(f"🎉 新最佳分數 Eps={episodes_done_total} | Score={best_score:.3f} | ")
                                self.save_checkpoint(f"best_model_wr_{best_score:.3f}.pt")
                            elif current_win_rate > best_win_rate:
                                best_win_rate = current_win_rate
                                send_telegram(f"🎉 新最佳we Eps={episodes_done_total} | Score={current_win_rate:.3f} | ")
                                self.save_checkpoint(f"best_model_we_{current_win_rate:.3f}.pt")
                            # 視覺化 (定期) ：快速評估 + 產生影片
                            if visualize_every > 0 and episodes_done_total > 0 and episodes_done_total % visualize_every == 0:
                                try:
                                    quick_games = max(5, int(eval_games // 2))
                                    metrics_v = self.evaluate_comprehensive(games=quick_games)
                                    score_v = float(metrics_v.get('comprehensive_score', 0.0))
                                    logger.info(
                                        f"🎯 視覺化評估 Eps={episodes_done_total} | Score={score_v:.3f} | "
                                        f"self={metrics_v.get('self_play', 0):.3f} minimax={metrics_v.get('vs_minimax', 0):.3f} rand={metrics_v.get('vs_random', 0):.3f}"
                                    )
                                except Exception as ee:
                                    logger.warning(f"視覺化前快速評估失敗：{ee}")
                                # 生成訓練對局影片
                                try:
                                    self.visualize_training_game(
                                        episodes_done_total,
                                        save_dir='videos',
                                        opponent='tactical',
                                        fps=2
                                    )
                                except Exception as ve:
                                    logger.warning(f"可視覺化失敗：{ve}")
                
                try:
                    quick_games = max(5, int(eval_games // 2))
                    metrics_v = self.evaluate_comprehensive(games=quick_games)
                    score_v = float(metrics_v.get('comprehensive_score', 0.0))
                    logger.info(
                        f"🎯 視覺化評估 Eps={episodes_done_total} | Score={score_v:.3f} | "
                        f"self={metrics_v.get('self_play', 0):.3f} minimax={metrics_v.get('vs_minimax', 0):.3f} rand={metrics_v.get('vs_random', 0):.3f}"
                    )
                except Exception as ee:
                    logger.warning(f"視覺化前評估失敗：{ee}")
                try:
                    self.visualize_training_game(episodes_done_total, save_dir='videos', opponent='tactical', fps=2)
                except Exception as ve:
                    logger.warning(f"可視覺化失敗：{ve}")

                # 補足 in_flight
                while len(in_flight) < target_inflight:
                    send_w = pending_weight_tasks > 0
                    if send_w:
                        pending_weight_tasks -= 1
                    in_flight.append(pool.apply_async(_worker_play_one, (_make_args(send_w),)))
                
                import time
                time.sleep(0.002)

        finally:
            try:
                pool.close(); pool.join()
            except Exception:
                pass

        logger.info("✅ 漸進式難度訓練完成")
        return self.agent


    def _record_game_frames(self, opponent: str = 'tactical', max_moves: int = 50):
        """遊玩一局並記錄每步的棋盤影格（6x7數組）。"""
        frames = []
        try:
            env = make('connectx', debug=False)
            env.reset()
            moves = 0
            with torch.no_grad():
                # 記錄初始棋盤（若可）
                try:
                    board, _ = self.agent.extract_board_and_mark(env.state, 0)
                    frames.append(self._flat_to_2d(board))
                except Exception:
                    pass
                while not env.done and moves < max_moves:
                    actions = []
                    for p in range(2):
                        if env.state[p]['status'] != 'ACTIVE':
                            actions.append(0)
                            continue
                        board, mark = self.agent.extract_board_and_mark(env.state, p)
                        valid = self.agent.get_valid_actions(board)
                        if p == 0:
                            # 我方用策略 + 安全檢查
                            a = self._choose_policy_with_tactics(board, mark, valid, training=False)
                        else:
                            if opponent == 'tactical':
                                a = self._tactical_random_opening_agent(board, mark, valid)
                            elif opponent == 'minimax':
                                a = self._choose_minimax_move(board, mark, max_depth=int(self.config.get('evaluation', {}).get('minimax_depth', 3)))
                            else:
                                # random family uses the same unified logic
                                a = self._tactical_random_opening_agent(board, mark, valid)
                        actions.append(int(a))
                    try:
                        env.step(actions)
                    except Exception:
                        break
                    moves += 1
                    try:
                        board, _ = self.agent.extract_board_and_mark(env.state, 0)
                        frames.append(self._flat_to_2d(board))
                    except Exception:
                        pass
        except Exception:
            pass
        return frames

    def visualize_training_game(self, episode_idx: int, save_dir: str = 'videos', opponent: str = 'tactical', fps: int = 2):
        """將一局可視化並輸出為影片（強制使用FFmpeg MP4）。返回輸出路徑或None。"""
        if not globals().get('VISUALIZATION_AVAILABLE', False):
            logger.info("未安裝可視化依賴，略過影片輸出。")
            return None
        try:
            import shutil
            import matplotlib
            import matplotlib.pyplot as plt
            from matplotlib import animation
            from matplotlib.patches import Patch
            # Configure ffmpeg path from system PATH. Force MP4 via FFmpeg.
            ffmpeg_path = shutil.which('ffmpeg')
            if not ffmpeg_path:
                raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg to export MP4.")
            matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg_path
        except Exception as e:
            logger.warning(f"載入matplotlib/ffmpeg失敗，略過影片輸出: {e}")
            return None

        frames = self._record_game_frames(opponent=opponent)
        if not frames:
            logger.info("無可視化影格，略過影片輸出。")
            return None

        # 讓最後一幀多停留一會（約2秒）
        try:
            hold_frames = max(1, int(fps * 2))  # 停留 2 秒
            frames = frames + [frames[-1]] * hold_frames
        except Exception:
            pass

        # 準備輸出目錄與檔名
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        mp4_path = os.path.join(save_dir, f"episode_{episode_idx}_{ts}.mp4")
        # gif_path 保留但不使用，若需回退可啟用
        gif_path = os.path.join(save_dir, f"episode_{episode_idx}_{ts}.gif")

        # 建立畫布與座標軸
        fig, ax = plt.subplots(figsize=(5.6, 4.8))
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(-0.5, 5.5)
        ax.set_xticks(range(7))
        ax.set_yticks(range(6))
        ax.grid(True)
        ax.set_title(f"Episode {episode_idx} vs {opponent}")

        # 新增圖例說明顏色: 紅=Agent(P1), 金=Opponent(P2)
        handles = [
            Patch(color='red', label='Agent (P1)'),
            Patch(color='gold', label='Opponent (P2)')
        ]
        ax.legend(handles=handles, loc='upper right', framealpha=0.9)

        # 初始化圓片集合
        discs = []
        def draw_board(grid):
            # 清除先前圓片
            for d in discs:
                d.remove()
            discs.clear()
            # 繪製棋子：grid[r][c] 1->紅, 2->黃
            for r in range(6):
                for c in range(7):
                    v = grid[r][c]
                    if v == 0:
                        continue
                    color = 'red' if v == 1 else 'gold'
                    circle = plt.Circle((c, 5 - r), 0.4, color=color)
                    ax.add_patch(circle)
                    discs.append(circle)

        def init():
            draw_board(frames[0])
            return discs

        def update(i):
            draw_board(frames[i])
            return discs

        anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(frames), interval=int(1000 / max(1, fps)), blit=False)

        out_path = None
        # 強制使用 FFmpeg 輸出 MP4（如果失敗則不回退 GIF）
        try:
            writer = animation.FFMpegWriter(fps=fps, codec='mpeg4', bitrate=1800, extra_args=['-pix_fmt', 'yuv420p'])
            anim.save(mp4_path, writer=writer)
            out_path = mp4_path
        except Exception as e:
            logger.warning(f"使用FFmpeg輸出MP4失敗: {e}")
            out_path = None
        finally:
            plt.close(fig)

        if out_path:
            logger.info(f"🎬 已輸出訓練對局影片: {out_path}")
        return out_path

def send_telegram(msg: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID") or "6166024220"
    if not token or not chat_id:
        logger.info("未設置 TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID，略過訊息通知。")
        return
    try:
        import requests
        base = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": msg}
        r = requests.post(base, data=payload, timeout=3.0)
        logger.info("已發送 Telegram 通知。")
    except Exception as e:
        logger.warning(f"Telegram 發送失敗: {e}")



def main():
    """Main entry: load config, resume from latest checkpoint, train, notify."""
    # Local imports to keep global scope clean
    from datetime import datetime

    # --- helpers ---
    def parse_ts_from_name(fname: str):
        FINAL_RE = re.compile(r"^final_(\d{8})_(\d{6})\.pt$")
        TS_TAIL_RE = re.compile(r"_(\d{8})_(\d{6})\.pt$")
        m = FINAL_RE.match(fname)
        if m:
            try:
                return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            except Exception:
                return None
        m = TS_TAIL_RE.search(fname)
        if m:
            try:
                return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            except Exception:
                return None
        return None

    def choose_latest_checkpoint_by_name(files):
        latest = None
        latest_time = None
        for p in files:
            fname = os.path.basename(p)
            ts = parse_ts_from_name(fname)
            try:
                t = ts.timestamp() if ts is not None else os.path.getmtime(p)
            except OSError:
                t = 0
            if latest is None or t > latest_time:
                latest = p
                latest_time = t
        return latest

    # --- 新增：挑選 best_model_wr_X.pt 中 win rate 最高者 ---
    def choose_best_wr_checkpoint(files):
        """在檔名中尋找 best_model_wr_*.pt，取 win rate 最大者，並驗證可讀取。
        例如: best_model_wr_0.393.pt > best_model_wr_0.380.pt
        若讀取失敗則跳過，全部失敗回傳 None。"""
        import re as _re
        pattern = _re.compile(r"best_model_wr_(\d+(?:\.\d+)?)\.pt$")
        # 收集 (wr, path)
        candidates = []
        for p in files:
            m = pattern.search(os.path.basename(p))
            if m:
                try:
                    wr = float(m.group(1))
                    candidates.append((wr, p))
                except ValueError:
                    continue
        if not candidates:
            return None
        # 依 win rate 由大到小排序
        candidates.sort(key=lambda x: x[0], reverse=True)
        for wr, path in candidates:
            try:
                # 快速驗證檔案是否可被 torch.load（不必套用）
                _ = torch.load(path, map_location='cpu')
                logger.info(f"選擇最高勝率檢查點: {os.path.basename(path)} (wr={wr:.3f})")
                return path
            except Exception as e:
                logger.warning(f"略過不可用勝率檢查點 {path}: {e}")
                continue
        return None

    def find_working_checkpoint(files):
        """回傳第一個可成功 torch.load 的檢查點（依時間新→舊）。若沒有則回傳 None。"""
        def _mtime(p):
            try:
                ts = parse_ts_from_name(os.path.basename(p))
                return ts.timestamp() if ts else os.path.getmtime(p)
            except Exception:
                return 0
        for p in sorted(files, key=_mtime, reverse=True):
            try:
                _ = torch.load(p, map_location='cpu')
                logger.info(f"檢查點可用：{p}")
                return p
            except Exception as e:
                logger.warning(f"略過不可用檢查點 {p}: {e}")
                continue
        return None
    # --- choose config ---
    cfg_path = "config.yaml"
    if not os.path.exists(cfg_path):
        alt_path = "connectx_config.yaml"
        if os.path.exists(alt_path):
            cfg_path = alt_path
        else:
            logger.error("找不到 config.yaml 或 connectx_config.yaml，無法啟動訓練。")
            send_telegram("⚠️ 找不到設定檔，訓練未啟動。")
            return

    trainer = ConnectXTrainer(cfg_path)

    # --- resume from checkpoint ---
    resume_from_env = os.getenv("CHECKPOINT_PATH") or os.getenv("RESUME_FROM")
    ckpt_to_load = None
    if resume_from_env and os.path.exists(resume_from_env):
        ckpt_to_load = resume_from_env
    else:
        files = glob.glob(os.path.join('checkpoints', '*.pt'))
        # 1) 先找最高勝率 best_model_wr_*.pt
        ckpt_to_load = choose_best_wr_checkpoint(files)
        if ckpt_to_load is None:
            # 2) 再找可用 (時間新→舊) 檢查點
            ckpt_to_load = find_working_checkpoint(files)
            if ckpt_to_load is None:
                # 3) 退回原本依名稱/mtime 的最新者
                ckpt_to_load = choose_latest_checkpoint_by_name(files)

    logger.info(f"ckpt_to_load: {ckpt_to_load}")
    send_telegram("使用: "+str(ckpt_to_load)+" model來繼續")
    if ckpt_to_load:
        loaded = trainer.load_checkpoint(ckpt_to_load)
        if not loaded:
            logger.warning(f"無法載入檢查點: {ckpt_to_load}，將以隨機初始化開始。")
    else:
        logger.info("未找到可用檢查點，將以隨機初始化開始訓練。")

    # --- train ---
    start_ts = datetime.now()
    err = None
    try:
        training_cfg = trainer.config.get('training', {})
        use_parallel = bool(training_cfg.get('parallel_rollout', True))
        if use_parallel:
            logger.info("使用平行蒐集進行訓練 train_parallel()")
            trainer.train_parallel()
            # trainer.train_with_ray()
        else:
            logger.info("使用單執行緒訓練 train()")
           
            trainer.train()
    except Exception as e:
        err = e
        logger.error(f"訓練過程發生錯誤: {e}\n{traceback.format_exc()}")
    finally:
        pass
    # --- notify ---

    elapsed = datetime.now() - start_ts
    episodes_done = len(trainer.episode_rewards)
    last_wr = trainer.win_rates[-1] if trainer.win_rates else None
    if err is None:
        msg = f"🎉 訓練完畢！\n回合數: {episodes_done}\n最後勝率: {last_wr if last_wr is not None else 'N/A'}\n耗時: {str(elapsed).split('.')[0]}"
    else:
        msg = f"⚠️ 訓練失敗: {err}\n回合數: {episodes_done}\n耗時: {str(elapsed).split('.')[0]}"
    send_telegram(msg)

if __name__ == "__main__":
    main()
