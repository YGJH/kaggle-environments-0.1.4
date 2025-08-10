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
        opponent_type = random.choice(['random', 'minimax', 'self'])
        player2_prob = float(args.get('player2_training_prob', 0.5))
        training_player = int(np.random.choice([1, 2], p=[1.0 - player2_prob, player2_prob]))

        transitions = []
        move_count, max_moves = 0, 50

        with torch.no_grad():
            while not env.done and move_count < max_moves:
                actions = []
                for player_idx in range(2):
                    if env.state[player_idx]['status'] == 'ACTIVE':
                        board, current_player = agent.extract_board_and_mark(env.state, player_idx)
                        valid_actions = agent.get_valid_actions(board)
                        if current_player == training_player:
                            state = agent.encode_state(board, current_player)
                            action, prob, value = agent.select_action(state, valid_actions, training=True)
                            transitions.append({
                                'state': state,
                                'action': int(action),
                                'prob': float(prob),
                                'value': float(value),
                                'training_player': training_player,
                                'opponent_type': opponent_type,
                                'is_dangerous': bool(if_i_will_lose_at_next(board, int(action), current_player, agent)),
                            })
                        else:
                            if opponent_type == 'random':
                                action = random_opponent_strategy(board, current_player, valid_actions, agent)
                            elif opponent_type == 'minimax':
                                action = minimax_opponent_strategy(board, current_player, valid_actions, agent, depth=3)
                            else:
                                action = self_play_opponent_strategy(board, current_player, valid_actions, agent)
                        actions.append(int(action))
                    else:
                        actions.append(0)
                try:
                    env.step(actions)
                except Exception:
                    break
                move_count += 1

        try:
            player_result = env.state[0]['reward'] if training_player == 1 else env.state[1]['reward']
        except Exception:
            player_result = 0

        return {
            'transitions': transitions,
            'training_player': training_player,
            'player_result': int(player_result),
            'game_length': len(transitions),
            'opponent_type': opponent_type,
            'policy_version_used': _WORKER["policy_version"],
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
        self.channels = 128
        self.drop_path_rate = float(max(0.0, drop_path_rate))
        self.attn_every = max(0, int(attn_every))

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
            def __init__(self, c, heads=4):
                super().__init__()
                self.heads = heads
                self.scale = (c // heads) ** -0.5
                self.qkv = nn.Conv2d(c, c * 3, 1, bias=False)
                self.proj = nn.Conv2d(c, c, 1, bias=False)
            def forward(self, x):  # x: (B,C,H,W)
                B, C, H, W = x.shape
                qkv = self.qkv(x).reshape(B, 3, self.heads, C // self.heads, H * W)
                q, k, v = qkv[:,0], qkv[:,1], qkv[:,2]  # (B,heads,dim,HW)
                attn = (q.transpose(-2,-1) @ k) * self.scale  # (B,heads,HW,HW)
                attn = torch.softmax(attn, dim=-1)
                out = (attn @ v.transpose(-2,-1)).transpose(-2,-1)  # (B,heads,dim,HW)
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

    def update_policy(self):
        """使用 PPO 更新策略"""
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
            except Exception as e:
                try:
                    logger.debug(f"entropy reset check failed: {e}")
                except Exception:
                    pass
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': avg_entropy,
            'total_loss': total_loss / self.k_epochs
        }

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

    # NEW: list all moves that do NOT give opponent an immediate winning reply
    def _safe_moves(self, board_flat, mark, valid_actions):
        return safe_moves(board_flat, mark, valid_actions, self.agent)

    # Tactical-aware random opponent move - using shared function
    def _random_with_tactics(self, board_flat, mark, valid_actions):
        return random_opponent_strategy(board_flat, mark, valid_actions, self.agent)

    # --- Random opening tactic for first player (mark==1) ---
    def _random_opening_move(self, board_flat, mark, valid_actions):
        """If this random player is the starter (mark==1), follow opening: 3 -> 2 -> 4 -> then 5 or 1 depending on top row empty.
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
            if my_tokens == 1 and 2 in valid_actions:
                return 2
            if my_tokens == 2 and 4 in valid_actions:
                return 4
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
                        if p == 0:
                            a, _, _ = self.agent.select_action(state, valid, training=False)
                        else:
                            a = self._tactical_random_opening_agent(board, mark, valid)
                        actions.append(int(a))
                    else:
                        actions.append(0)
                try:
                    env.step(actions)
                except Exception:
                    break
                move_count += 1
        # 回傳玩家1結果（我方）
        try:
            return 1 if env.state[0]['reward'] == 1 else (0 if env.state[0]['reward'] == 0 else -1)
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
                            if p == 0:
                                a, _, _ = self.agent.select_action(state, valid, training=False)
                            else:
                                a = self._tactical_random_opening_agent(board, mark, valid)
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
            r = self._play_one_game_self_tactical(main_as_player1=True)
            if r == 1:
                wins += 1
            elif r == 0:
                draws += 1
        return wins / max(1, int(games))

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
        weights = self.config.get('evaluation', {}).get('weights', {})
        w_self = float(weights.get('self_play', 0.4))
        w_minimax = float(weights.get('vs_minimax', 0.4))
        w_random = float(weights.get('vs_random', 0.2))
        total_w = max(1e-6, w_self + w_minimax + w_random)
        comprehensive = (w_self * self_play + w_minimax * vs_minimax + w_random * vs_random) / total_w
        
        return {
            'vs_random': vs_random,
            'vs_minimax': vs_minimax,
            'self_play': self_play,
            'comprehensive_score': comprehensive,
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
        episodes_per_update = int(cfg_t.get('episodes_per_update', 16))  # 每次想處理多少局（僅做節奏控制）
        max_episodes = int(cfg_t.get('max_episodes', 100000))
        eval_frequency = int(cfg_t.get('eval_frequency', 200))
        eval_games = int(cfg_t.get('eval_games', 30))
        win_scale = float(cfg_t.get('win_reward_scaling', 1.0))
        loss_scale = float(cfg_t.get('loss_penalty_scaling', 1.0))
        danger_scale = float(self.config.get('agent', {}).get('tactical_bonus', 0.1))
        min_batch_size = int(self.agent.config.get('min_batch_size', 512))
        visualize_every = int(cfg_t.get('visualize_every', 100))

        inflight_multiplier = int(cfg_t.get('inflight_multiplier', 2))  # 建議 1~3
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
        episodes_done_total = len(self.episode_rewards)

        # 版本控制
        policy_version = 0
        pending_weight_tasks = 0  # 本版本還需要送出幾個帶權重的任務（初始 0，由首次更新後觸發）

        def _make_args(send_weights: bool):
            weights_np = None
            if send_weights:
                # 僅當需要廣播新版本時才序列化一次
                weights_np = {k: v.detach().cpu().numpy()
                            for k, v in self.agent.policy_net.state_dict().items()}
            return {
                'policy_version': policy_version,
                'policy_state': weights_np if send_weights else None,
                'player2_training_prob': self.player2_training_prob,
                'seed': rng.randrange(2**31 - 1),
            }

        # 建立初始 in-flight 任務
        in_flight = []
        # 第一次也需要把版本 0 的權重送給所有 worker
        pending_weight_tasks = num_workers
        for _ in range(target_inflight):
            send_w = pending_weight_tasks > 0
            if send_w:
                pending_weight_tasks -= 1
            ar = pool.apply_async(_worker_play_one, (_make_args(send_w),))
            in_flight.append(ar)

        logger.info(f"🚀 完全流水線訓練開始：workers={num_workers}, target_inflight={target_inflight}")

        steps_since_update = 0
        try:
            while episodes_done_total < max_episodes:
                # 輕量輪詢完成的任務（避免 busy-wait）
                i = 0
                while i < len(in_flight):
                    ar = in_flight[i]
                    if ar.ready():
                        # 取結果並處理
                        res = ar.get()
                        # 從 in_flight 移除（交換刪除避免 O(n)）
                        in_flight[i] = in_flight[-1]
                        in_flight.pop()
                        # 立刻補上一個新任務（保持管線滿）
                        send_w = pending_weight_tasks > 0
                        if send_w:
                            pending_weight_tasks -= 1
                        in_flight.append(pool.apply_async(_worker_play_one, (_make_args(send_w),)))

                        # === consume result ===
                        err_msg = res.get('error') if isinstance(res, dict) else None
                        transitions = res.get('transitions', []) if isinstance(res, dict) else []
                        if not transitions:
                            self.episode_rewards.append(0.0)
                            if err_msg:
                                logger.debug(f"worker episode returned empty transitions: {err_msg}")
                        else:
                            player_result = int(res.get('player_result', 0))
                            ep_reward_sum = 0.0
                            last_idx = len(transitions) - 1
                            for idx, tr in enumerate(transitions):
                                state = tr['state']
                                action = int(tr['action'])
                                prob = float(tr['prob'])
                                reward = 0.0
                                if idx == last_idx:
                                    reward += (1.0 * win_scale) if player_result == 1 else (
                                            -1.0 * loss_scale if player_result == -1 else 0.0)
                                if tr.get('is_dangerous', False):
                                    reward += -10.0 * danger_scale
                                ep_reward_sum += reward
                                done = (idx == last_idx)
                                self.agent.store_transition(state, action, prob, reward, done)
                                steps_since_update += 1
                            self.episode_rewards.append(ep_reward_sum)

                            # 依你喜歡的節奏：步數達到就即刻更新
                            if steps_since_update >= min_batch_size:
                                info = self.agent.update_policy()
                                if info is not None:
                                    self.training_losses.append(info.get('total_loss', 0.0))
                                    policy_version += 1
                                    # 新版本出爐 → 至少廣播 num_workers 份帶權重任務
                                    pending_weight_tasks += num_workers
                                steps_since_update = 0

                        episodes_done_total += 1

                        # 週期性評估 / 視覺化（不要卡太久—保持快速）
                        if eval_frequency > 0 and episodes_done_total % eval_frequency == 0:
                            metrics = self.evaluate_comprehensive(games=eval_games)
                            score = float(metrics.get('comprehensive_score', 0.0))
                            self.win_rates.append(score)
                            try:
                                self.agent.scheduler.step(score)
                            except Exception:
                                pass
                            logger.info(
                                f"📈 Eps={episodes_done_total} | Score={score:.3f} | "
                                f"self={metrics.get('self_play', 0):.3f} minimax={metrics.get('vs_minimax', 0):.3f} rand={metrics.get('vs_random', 0):.3f}"
                            )
                            try:
                                if self._detect_convergence_stagnation(episodes_done_total, score):
                                    self._handle_convergence_stagnation(episodes_done_total)
                            except Exception:
                                pass
                            if score > best_score:
                                best_score = score
                                self.save_checkpoint(f"best_model_wr_{best_score:.3f}.pt")

                        if visualize_every > 0 and episodes_done_total % visualize_every == 0:
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

                        # 不遞增 i，因為我們把末尾元素搬到 i 了；繼續檢查新的 in_flight[i]
                        continue

                    else:
                        i += 1

                # 若 in_flight 因故少於目標，補足（理論上不會發生，但保險）
                while len(in_flight) < target_inflight:
                    send_w = pending_weight_tasks > 0
                    if send_w:
                        pending_weight_tasks -= 1
                    in_flight.append(pool.apply_async(_worker_play_one, (_make_args(send_w),)))
                import time
                # 小睡一下，降低 busy-wait（不影響吞吐）
                time.sleep(0.002)

        finally:
            try:
                pool.close(); pool.join()
            except Exception:
                pass

        logger.info("✅ 完全流水線訓練完成")
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
                # 先快速嘗試讀取，僅為驗證檔案結構，不需真正套用
                _ = torch.load(p, map_location='cpu')
                logger.info(f"檢查點可用：{p}")
                return p
            except Exception as e:
                logger.warning(f"略過不可用檢查點 {p}: {e}")
                continue
       
        return None

    def send_telegram(msg: str):
        token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID") or "6166024220"
        if not token or not chat_id:
            logger.info("未設置 TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID，略過訊息通知。")
            return
        try:
            base = f"https://api.telegram.org/bot{token}/sendMessage"

            logger.info("已發送 Telegram 通知。")
        except Exception as e:
            logger.warning(f"Telegram 發送失敗: {e}")

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
    import glob
    resume_from_env = os.getenv("CHECKPOINT_PATH") or os.getenv("RESUME_FROM")
    ckpt_to_load = None
    if resume_from_env and os.path.exists(resume_from_env):
        ckpt_to_load = resume_from_env
    else:
        files = glob.glob(os.path.join('checkpoints', '*.pt'))
       
        # 優先選擇可成功被 torch.load 的最新檔，避免讀到半寫入壞檔
        ckpt_to_load = find_working_checkpoint(files)
        if ckpt_to_load is None:
            # 退而求其次選名稱/mtime 最新者
            ckpt_to_load = choose_latest_checkpoint_by_name(files)

    logger.info(f"ckpt_to_load: {ckpt_to_load}")

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
        else:
            logger.info("使用單執行緒訓練 train()")
           
            trainer.train()
    except Exception as e:
        err = e
        logger.error(f"訓練過程發生錯誤: {e}\n{traceback.format_exc()}")
    finally:
        # 儲存最後檢查點
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            trainer.save_checkpoint(f"final_{ts_tag}.pt")
        except Exception as se:
            logger.warning(f"保存最終檢查點失敗: {se}")

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
