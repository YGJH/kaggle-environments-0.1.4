#!/usr/bin/env python3

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
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

# NEW: Worker function for collecting one episode in a separate process
# It runs a lightweight copy of the PPOAgent on CPU to avoid GPU contention.
def _worker_collect_episode(args):
    try:
        # Force CPU in worker to avoid CUDA context sharing issues
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
        cfg = args['config']
        policy_state = args['policy_state']
        player2_prob = args.get('player2_training_prob', 0.7)
        use_tactical_opp = bool(args.get('use_tactical_opponent', False))
        seed = args.get('seed', None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Create a minimal agent on CPU and load weights
        agent = PPOAgent(cfg['agent'])
        agent.device = torch.device('cpu')
        agent.policy_net.to(agent.device)
        # state_dict tensors already on CPU
        agent.policy_net.load_state_dict(policy_state)
        agent.policy_net.eval()

        # --- Tactical helpers (local to worker) ---
        def flat_to_2d(board_flat):
            try:
                if isinstance(board_flat[0], list):
                    return board_flat
            except Exception:
                pass
            return [list(board_flat[r*7:(r+1)*7]) for r in range(6)]
        def find_drop_row(grid, col):
            for r in range(5, -1, -1):
                if grid[r][col] == 0:
                    return r
            return None
        def is_win_from(grid, r, c, mark):
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
            grid = flat_to_2d(board_flat)
            if col < 0 or col > 6:
                return False
            r = find_drop_row(grid, col)
            if r is None:
                return False
            grid[r][col] = mark
            return is_win_from(grid, r, col, mark)
        def if_i_can_win(board_flat, mark):
            for c in agent.get_valid_actions(board_flat):
                if is_winning_move(board_flat, c, mark):
                    return c
            return None
        def if_i_will_lose(board_flat, mark):
            opp = 3 - mark
            for c in agent.get_valid_actions(board_flat):
                if is_winning_move(board_flat, c, opp):
                    return c
            return None
        def if_i_will_lose_at_next(board_flat, move_col, mark):
            grid = flat_to_2d(board_flat)
            r = find_drop_row(grid, move_col) if 0 <= move_col <= 6 else None
            if r is None:
                return True
            grid[r][move_col] = mark
            opp = 3 - mark
            # opponent immediate winning reply?
            for c in agent.get_valid_actions([grid[i][j] for i in range(6) for j in range(7)]):
                if is_winning_move([grid[i][j] for i in range(6) for j in range(7)], c, opp):
                    return True
            return False
        def random_with_tactics(board_flat, mark, valid_actions):
            c = if_i_can_win(board_flat, mark)
            if c is not None:
                return c
            c = if_i_will_lose(board_flat, mark)
            if c is not None:
                return c
            safe = [a for a in valid_actions if not if_i_will_lose_at_next(board_flat, a, mark)]
            if safe:
                return random.choice(safe)
            return random.choice(valid_actions)

        env = make('connectx', debug=False)
        env.reset()

        # Choose which player to collect training transitions for (skewed towards player2 by config)
        p1_prob = 1.0 - float(player2_prob)
        training_player = int(np.random.choice([1, 2], p=[p1_prob, player2_prob]))

        transitions = []
        move_count = 0
        max_moves = 50

        with torch.no_grad():
            while not env.done and move_count < max_moves:
                actions = []
                for player_idx in range(2):
                    if env.state[player_idx]['status'] == 'ACTIVE':
                        board, current_player = agent.extract_board_and_mark(env.state, player_idx)
                        valid_actions = agent.get_valid_actions(board)
                        if use_tactical_opp and current_player != training_player:
                            action = random_with_tactics(board, current_player, valid_actions)
                            prob = 1.0 / max(1, len(valid_actions))
                            value = 0.0
                        else:
                            state = agent.encode_state(board, current_player)
                            action, prob, value = agent.select_action(state, valid_actions, training=True)
                        if current_player == training_player:
                            transitions.append({
                                'state': agent.encode_state(board, current_player),
                                'action': int(action),
                                'prob': float(prob),
                                'value': float(value),
                                'training_player': training_player,
                                'is_dangerous': bool(if_i_will_lose_at_next(board, int(action), current_player)) if use_tactical_opp else False
                            })
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
            'player_result': player_result,
            'game_length': len(transitions)
        }
    except Exception as e:
        # In workers, avoid heavy logging; return empty result on failure
        return {
            'transitions': [],
            'training_player': 1,
            'player_result': 0,
            'game_length': 0,
            'error': str(e)
        }

class ConnectXNet(nn.Module):
    """強化學習用的 ConnectX 深度神經網路"""

    def __init__(self, input_size=126, hidden_size=156, num_layers=256):
        super(ConnectXNet, self).__init__()

        # 輸入層
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 隱藏層（殘差連接 + 層正規化代替批量正規化）
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),  # 使用 LayerNorm 代替 BatchNorm1d
                nn.ReLU(),
                nn.Dropout(0.15),  # 稍微增加dropout
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size)   # 使用 LayerNorm 代替 BatchNorm1d
            ) for _ in range(num_layers)
        ])

        # 策略頭（動作概率）
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),  # 使用 LayerNorm 代替 BatchNorm1d
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 7),  # 7 列
            nn.Softmax(dim=-1)
        )

        # 價值頭（狀態價值）
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),  # 使用 LayerNorm 代替 BatchNorm1d
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # 輸入處理
        x = self.input_layer(x)

        # 殘差連接
        for hidden_layer in self.hidden_layers:
            residual = x
            x = hidden_layer(x)
            x = torch.relu(x + residual)

        # 輸出頭
        policy = self.policy_head(x)
        value = self.value_head(x)

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

    def extract_board_and_mark(self, env_state, player_idx):
        """從環境狀態中提取棋盤和玩家標記"""
        try:
            # 檢查環境狀態基本結構
            if not env_state or len(env_state) <= player_idx:
                logger.warning(f"環境狀態無效或玩家索引超出範圍: len={len(env_state) if env_state else 0}, player_idx={player_idx}")
                return [0] * 42, player_idx + 1

            # 獲取玩家狀態
            player_state = env_state[player_idx]
            if 'observation' not in player_state:
                logger.warning(f"玩家 {player_idx} 狀態中沒有 observation")
                return [0] * 42, player_idx + 1

            obs = player_state['observation']

            # 首先嘗試獲取標記
            mark = None
            if 'mark' in obs:
                mark = obs['mark']
            elif hasattr(obs, 'mark'):
                mark = obs.mark
            else:
                mark = player_idx + 1  # 默認標記
                logger.warning(f"無法獲取玩家標記，使用默認值: {mark}")

            # 嘗試獲取棋盤
            board = None

            # 方法1: 直接從當前玩家觀察獲取
            if 'board' in obs:
                board = obs['board']
                logger.debug(f"從玩家 {player_idx} 字典方式獲取棋盤")
            elif hasattr(obs, 'board'):
                board = obs.board
                logger.debug(f"從玩家 {player_idx} 屬性方式獲取棋盤")

            # 方法2: 如果當前玩家沒有棋盤，從其他玩家獲取
            if board is None:
                logger.warning(f"玩家 {player_idx} 觀察中沒有棋盤數據，可用鍵: {list(obs.keys()) if hasattr(obs, 'keys') else 'N/A'}")
                logger.warning(f"玩家 {player_idx} 狀態: {player_state.get('status', 'Unknown')}")

                # 嘗試從其他玩家獲取
                for other_idx in range(len(env_state)):
                    if other_idx != player_idx:
                        try:
                            other_state = env_state[other_idx]
                            if 'observation' in other_state:
                                other_obs = other_state['observation']
                                if 'board' in other_obs:
                                    board = other_obs['board']
                                    logger.info(f"從玩家 {other_idx} 獲取棋盤 (字典方式)")
                                    break
                                elif hasattr(other_obs, 'board'):
                                    board = other_obs.board
                                    logger.info(f"從玩家 {other_idx} 獲取棋盤 (屬性方式)")
                                    break
                        except Exception as e:
                            logger.debug(f"從玩家 {other_idx} 獲取棋盤失敗: {e}")
                            continue

            # 方法3: 最後備用方案
            if board is None:
                logger.warning("所有方法都無法獲取棋盤，使用空棋盤")
                board = [0] * 42

            # 驗證棋盤數據
            if not isinstance(board, (list, tuple)) or len(board) != 42:
                logger.warning(f"棋盤數據格式不正確: type={type(board)}, len={len(board) if hasattr(board, '__len__') else 'N/A'}")
                board = [0] * 42

            logger.debug(f"成功提取玩家 {player_idx} 的狀態: 棋盤長度={len(board)}, 標記={mark}")
            return list(board), mark

        except Exception as e:
            logger.error(f"提取棋盤狀態時出錯: {e}")
            logger.error(f"環境狀態類型: {type(env_state)}")

            # 詳細調試信息
            try:
                if env_state and len(env_state) > player_idx:
                    player_state = env_state[player_idx]
                    logger.error(f"玩家 {player_idx} 狀態鍵: {list(player_state.keys()) if hasattr(player_state, 'keys') else 'N/A'}")
                    logger.error(f"玩家 {player_idx} 狀態: {player_state.get('status', 'Unknown')}")

                    if 'observation' in player_state:
                        obs = player_state['observation']
                        obs_keys = list(obs.keys()) if hasattr(obs, 'keys') else [attr for attr in dir(obs) if not attr.startswith('_')]
                        logger.error(f"觀察鍵: {obs_keys}")

                        # 檢查觀察的內容
                        if hasattr(obs, 'keys'):
                            for key in obs.keys():
                                try:
                                    value = obs[key]
                                    logger.error(f"  {key}: {type(value)} = {value}")
                                except:
                                    logger.error(f"  {key}: 無法訪問")
            except Exception as debug_e:
                logger.error(f"調試信息收集失敗: {debug_e}")

            # 返回默認值
            return [0] * 42, player_idx + 1

    def encode_state(self, board, mark):
        """編碼棋盤狀態"""
        # 確保 board 是有效的
        if board is None or (isinstance(board, (list, np.ndarray)) and len(board) == 0):
            board = [0] * 42
        else:
            # 處理嵌套列表或確保是一維列表
            if isinstance(board, np.ndarray):
                board = board.flatten().tolist()
            elif isinstance(board, list) and len(board) > 0:
                # 檢查是否為嵌套列表
                if isinstance(board[0], (list, np.ndarray)):
                    board = [item for sublist in board for item in sublist]
                    
            # 確保長度為 42
            if len(board) != 42:
                if len(board) < 42:
                    board = list(board) + [0] * (42 - len(board))
                else:
                    board = list(board)[:42]

        # 轉換為 6x7 矩陣
        state = np.array(board, dtype=np.int32).reshape(6, 7)

        # 創建三個特徵通道
        # 通道 1: 當前玩家的棋子
        player_pieces = (state == mark).astype(np.float32)
        # 通道 2: 對手的棋子
        opponent_pieces = (state == (3 - mark)).astype(np.float32)
        # 通道 3: 空位
        empty_spaces = (state == 0).astype(np.float32)

        # 拉平並連接
        encoded = np.concatenate([
            player_pieces.flatten(),
            opponent_pieces.flatten(),
            empty_spaces.flatten()
        ])

        return encoded

    def decode_state(self, encoded_state):
        """從編碼狀態重建棋盤"""
        try:
            # 編碼狀態有126個元素，分為三個42元素的通道
            if len(encoded_state) != 126:
                raise ValueError(f"編碼狀態長度錯誤: {len(encoded_state)}")
            
            # 提取三個通道
            player_pieces = encoded_state[:42].reshape(6, 7)
            opponent_pieces = encoded_state[42:84].reshape(6, 7)
            empty_spaces = encoded_state[84:126].reshape(6, 7)
            
            # 重建棋盤
            board = np.zeros((6, 7), dtype=int)
            board[player_pieces == 1] = 1  # 假設當前玩家是1
            board[opponent_pieces == 1] = 2  # 對手是2
            
            return board.tolist()
        except Exception as e:
            logger.warning(f"狀態解碼失敗: {e}，返回空棋盤")
            return [[0 for _ in range(7)] for _ in range(6)]

    def get_valid_actions(self, board):
        """獲取有效動作"""
        # 確保 board 是有效的
        if not board or len(board) != 42:
            board = [0] * 42

        # 檢查每一列的頂部是否為空
        valid_actions = []
        for col in range(7):
            if board[col] == 0:  # 檢查每列的頂部
                valid_actions.append(col)

        # 如果沒有有效動作，返回所有列（備用方案）
        if not valid_actions:
            valid_actions = list(range(7))

        return valid_actions

    def select_action(self, state, valid_actions, training=True, temperature=1.0, exploration_bonus=0.0):
        """選擇動作（支持溫度採樣和探索獎勵）"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # 根據 batch size 決定是否使用 BatchNorm
        use_eval_mode = state_tensor.size(0) == 1  # 單個樣本時使用eval模式

        if use_eval_mode:
            # 單個樣本時使用評估模式避免 BatchNorm 問題
            self.policy_net.eval()

        with torch.no_grad():
            action_probs, state_value = self.policy_net(state_tensor)

        # 恢復訓練模式
        if training and use_eval_mode:
            self.policy_net.train()

        # 遮罩無效動作
        action_probs = action_probs.cpu().numpy()[0]
        masked_probs = np.zeros_like(action_probs)
        masked_probs[valid_actions] = action_probs[valid_actions]

        # 正規化概率
        if masked_probs.sum() > 0:
            masked_probs /= masked_probs.sum()
        else:
            # 後備方案：均勻分佈
            masked_probs[valid_actions] = 1.0 / len(valid_actions)

        # 應用溫度和探索獎勵
        if training and (temperature != 1.0 or exploration_bonus > 0.0):
            # 溫度採樣
            if temperature != 1.0:
                masked_probs = np.power(masked_probs + 1e-8, 1/temperature)
                masked_probs /= masked_probs.sum()

            # 探索獎勵（為不常選擇的動作增加概率）
            if exploration_bonus > 0.0:
                uniform_dist = np.zeros_like(masked_probs)
                uniform_dist[valid_actions] = 1.0 / len(valid_actions)
                masked_probs = (1 - exploration_bonus) * masked_probs + exploration_bonus * uniform_dist

        if training:
            # 訓練時採樣動作
            action = np.random.choice(7, p=masked_probs)
        else:
            # 評估時選擇最佳動作
            action = valid_actions[np.argmax(masked_probs[valid_actions])]

        # 確保返回的動作是 Python int 類型，避免 numpy 類型問題
        action = int(action)

        return action, action_probs[action], state_value.item()

    def store_transition(self, state, action, prob, reward, done):
        """儲存轉換"""
        self.memory.append((state, action, prob, reward, done))

    def compute_gae(self, rewards, values, dones, next_value, gamma=0.99, lam=0.95):
        """計算廣義優勢估計（GAE）"""
        advantages = []
        gae = 0

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[i + 1]

            delta = rewards[i] + gamma * next_val * (1 - dones[i]) - values[i]
            gae = delta + gamma * lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

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

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
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
        boosted_p2 = max(self.player2_training_prob, 0.85)
        if boosted_p2 != self.player2_training_prob:
            logger.info(f"提高後手訓練比例: {self.player2_training_prob:.2f} -> {boosted_p2:.2f}")
            self.player2_training_prob = boosted_p2
        
        # 持續學習數據
        self.continuous_learning_data = None
        self.continuous_learning_targets = None

        # 創建目錄
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

    # --- Board helpers (6x7) ---
    def _flat_to_2d(self, board_flat):
        try:
            if isinstance(board_flat[0], list):
                # already 2d
                return board_flat
        except Exception:
            pass
        return [list(board_flat[r*7:(r+1)*7]) for r in range(6)]

    def _find_drop_row(self, grid, col):
        for r in range(5, -1, -1):
            if grid[r][col] == 0:
                return r
        return None

    def _apply_move(self, board_flat, col, mark):
        grid = self._flat_to_2d(board_flat)
        if col < 0 or col > 6:
            return None
        r = self._find_drop_row(grid, col)
        if r is None:
            return None
        grid[r][col] = mark
        return [grid[i][j] for i in range(6) for j in range(7)]

    def _is_win_from(self, grid, r, c, mark):
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

    def _is_winning_move(self, board_flat, col, mark):
        grid = self._flat_to_2d(board_flat)
        if col < 0 or col > 6:
            return False
        r = self._find_drop_row(grid, col)
        if r is None:
            return False
        grid[r][col] = mark
        return self._is_win_from(grid, r, col, mark)

    def _any_winning_moves(self, board_flat, mark):
        wins = []
        for c in self.agent.get_valid_actions(board_flat):
            if self._is_winning_move(board_flat, c, mark):
                wins.append(c)
        return wins

    # --- Tactical utilities requested ---
    def if_i_can_win(self, board_flat, mark):
        for c in self.agent.get_valid_actions(board_flat):
            if self._is_winning_move(board_flat, c, mark):
                return c
        return None

    def if_i_will_lose(self, board_flat, mark):
        opp = 3 - mark
        for c in self.agent.get_valid_actions(board_flat):
            if self._is_winning_move(board_flat, c, opp):
                return c
        return None

    def if_i_will_lose_at_next(self, board_flat, move_col, mark):
        opp = 3 - mark
        next_board = self._apply_move(board_flat, move_col, mark)
        if next_board is None:
            return True
        for c in self.agent.get_valid_actions(next_board):
            if self._is_winning_move(next_board, c, opp):
                return True
        return False

    # Tactical-aware random opponent move
    def _random_with_tactics(self, board_flat, mark, valid_actions):
        # 1) Immediate win
        c = self.if_i_can_win(board_flat, mark)
        if c is not None:
            return c
        # 2) Must block opponent win
        c = self.if_i_will_lose(board_flat, mark)
        if c is not None:
            return c
        # 3) Avoid blunders if possible
        safe = [a for a in valid_actions if not self.if_i_will_lose_at_next(board_flat, a, mark)]
        if safe:
            return random.choice(safe)
        # 4) Fallback to any
        return random.choice(valid_actions)

    # Reward shaping based on tactical patterns
    def analyze_tactical_patterns(self, board, action, player_id):
        try:
            # Normalize board to flat 42 list
            if isinstance(board, list) and len(board) == 6 and isinstance(board[0], list):
                flat = [board[r][c] for r in range(6) for c in range(7)]
            else:
                flat = list(board)
            bonus = 0.0
            # Immediate win
            if self._is_winning_move(flat, action, player_id):
                bonus += 12.0
            # Immediate block
            block_col = self.if_i_will_lose(flat, player_id)
            if block_col is not None and block_col == int(action):
                bonus += 8.0
            # Double-threat creation
            next_board = self._apply_move(flat, action, player_id)
            if next_board is not None:
                next_valid = self.agent.get_valid_actions(next_board)
                threats = sum(1 for c in next_valid if self._is_winning_move(next_board, c, player_id))
                if threats >= 2:
                    bonus += 6.0
            # Center control
            if int(action) == 3:
                bonus += 2.0
            # Danger: allow opponent immediate win
            if self.if_i_will_lose_at_next(flat, action, player_id):
                bonus -= 10.0
            # Scale by config tactical_bonus if present
            scale = float(self.config.get('agent', {}).get('tactical_bonus', 0.1))
            return bonus * scale
        except Exception:
            return 0.0

    def _detect_convergence_stagnation(self, episodes_done: int, win_rate: float):
        # Detect no improvement over last 5 evals and low variance
        if len(self.win_rates) < 6:
            return False
        recent = self.win_rates[-6:]
        best_recent = max(recent[:-1])
        improved = win_rate > best_recent + 1e-3
        var = np.var(recent)
        no_progress = not improved and var < 1e-4
        return no_progress

    def _handle_convergence_stagnation(self, episodes_done: int):
        # Simple actions: increase entropy a bit and reset scheduler cooldown via slight LR bump
        logger.info("(策略) 偵測到停滯：臨時提高探索與重置學習率冷卻")
        self.agent.entropy_coef = min(self.agent.entropy_coef * 1.2, 0.05)
        for g in self.agent.optimizer.param_groups:
            g['lr'] = max(g['lr'] * 1.05, g['lr'])

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
                            a = self._random_with_tactics(board, mark, valid)
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
    def _score_window(self, window, mark):
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

    def _evaluate_board(self, board_flat, mark):
        grid = self._flat_to_2d(board_flat)
        score = 0
        # center preference
        center_col = [grid[r][3] for r in range(6)]
        score += center_col.count(mark) * 3
        # Horizontal
        for r in range(6):
            row = grid[r]
            for c in range(4):
                window = row[c:c+4]
                score += self._score_window(window, mark)
        # Vertical
        for c in range(7):
            col = [grid[r][c] for r in range(6)]
            for r in range(3):
                window = col[r:r+4]
                score += self._score_window(window, mark)
        # Diagonals
        for r in range(3):
            for c in range(4):
                window = [grid[r+i][c+i] for i in range(4)]
                score += self._score_window(window, mark)
        for r in range(3, 6):
            for c in range(4):
                window = [grid[r-i][c+i] for i in range(4)]
                score += self._score_window(window, mark)
        return score

    def _has_winner(self, board_flat, mark):
        grid = self._flat_to_2d(board_flat)
        # check all
        for r in range(6):
            for c in range(7):
                if grid[r][c] != mark:
                    continue
                if self._is_win_from(grid, r, c, mark):
                    return True
        return False

    def _minimax(self, board_flat, depth, alpha, beta, current_mark, maximizing_mark):
        valid_moves = self.agent.get_valid_actions(board_flat)
        if depth == 0 or not valid_moves:
            return self._evaluate_board(board_flat, maximizing_mark), None
        # Terminal strong checks: if previous player already has a winning move next, skip handled at callers
        best_move = None
        if current_mark == maximizing_mark:
            value = -float('inf')
            for c in valid_moves:
                nb = self._apply_move(board_flat, c, current_mark)
                if nb is None:
                    continue
                # immediate win
                if self._has_winner(nb, current_mark):
                    return 1e6 - (5 - depth), c
                child_val, _ = self._minimax(nb, depth-1, alpha, beta, 3-current_mark, maximizing_mark)
                if child_val > value:
                    value, best_move = child_val, c
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value, best_move
        else:
            value = float('inf')
            for c in valid_moves:
                nb = self._apply_move(board_flat, c, current_mark)
                if nb is None:
                    continue
                if self._has_winner(nb, current_mark):
                    return -1e6 + (5 - depth), c
                child_val, _ = self._minimax(nb, depth-1, alpha, beta, 3-current_mark, maximizing_mark)
                if child_val < value:
                    value, best_move = child_val, c
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value, best_move

    def _choose_minimax_move(self, board_flat, mark, max_depth):
        # Tactical overrides
        c = self.if_i_can_win(board_flat, mark)
        if c is not None:
            return c
        c = self.if_i_will_lose(board_flat, mark)
        if c is not None:
            return c
        valid = self.agent.get_valid_actions(board_flat)
        safe = [a for a in valid if not self.if_i_will_lose_at_next(board_flat, a, mark)]
        moves = safe if safe else valid
        # Try each candidate with shallow minimax to break ties
        depth = max(1, min(4, int(self.config.get('evaluation', {}).get('minimax_depth', 3))))
        best_score = -float('inf')
        best_move = random.choice(moves)
        for a in moves:
            nb = self._apply_move(board_flat, a, mark)
            if nb is None:
                continue
            score, _ = self._minimax(nb, depth-1, -float('inf'), float('inf'), 3-mark, mark)
            if score > best_score:
                best_score, best_move = score, a
        return best_move

    def evaluate_against_random(self, games: int = 50):
        wins = 0
        draws = 0
        for _ in range(max(1, int(games))):
            r = self._play_one_game_vs_random()
            if r == 1:
                wins += 1
            elif r == 0:
                draws += 1
        total = max(1, int(games))
        return wins / total

    def evaluate_against_minimax(self, games: int = 20):
        wins = 0
        draws = 0
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
        total = max(1, int(games))
        return wins / total

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
                                a = self._random_with_tactics(board, mark, valid)
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
        c = self.if_i_can_win(board_flat, mark)
        if c is not None:
            return c
        # 再看是否必須擋下對方
        c = self.if_i_will_lose(board_flat, mark)
        if c is not None:
            return c
        # 用策略網路選擇
        state = self.agent.encode_state(board_flat, mark)
        a, _, _ = self.agent.select_action(state, valid_actions, training=training)
        a = int(a)
        # 避免立即送給對手致勝
        if self.if_i_will_lose_at_next(board_flat, a, mark):
            safe = [x for x in valid_actions if not self.if_i_will_lose_at_next(board_flat, x, mark)]
            if safe:
                return random.choice(safe)
        return a

    def _play_one_game_self_tactical(self, main_as_player1=True):
        """自我對戰：對手使用戰術规则(if_i_can_win/if_i_will_lose)，主控方用策略網路。
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
                        # 對手使用純戰術規則的隨機對手
                        a = self._random_with_tactics(board, mark, valid)
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
        for _ in range(max(1, int(games))):
            r = self._play_one_game_self_tactical(main_as_player1=False)
            if r == 1:
                wins += 1
            elif r == 0:
                draws += 1
        return wins / max(1, int(games))

    def evaluate_comprehensive(self, games: int = 50):
        vs_random = self.evaluate_against_random(games)
        vs_minimax = self.evaluate_against_minimax(max(1, games // 2))
        # 自我對戰分兩種：先手與後手焦點，這裡使用後手焦點以鼓勵後手能力
        self_play = self.evaluate_self_play_player2_focus(max(1, games // 2))
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
            torch.save(checkpoint, path)
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

    def fix_checkpoint_compatibility(self, old_checkpoint_path: str, new_checkpoint_path: str | None = None):
        """修復舊檢查點的兼容性問題，返回新檔案路徑或None"""
        try:
            if new_checkpoint_path is None:
                new_checkpoint_path = old_checkpoint_path.replace('.pt', '_fixed.pt')
            logger.info(f"嘗試修復檢查點: {old_checkpoint_path}")
            ckpt = torch.load(old_checkpoint_path, map_location='cpu')
            # 常見修復：版本資訊轉字串
            if isinstance(ckpt, dict) and 'pytorch_version' in ckpt and not isinstance(ckpt['pytorch_version'], str):
                ckpt['pytorch_version'] = str(ckpt['pytorch_version'])
            torch.save(ckpt, new_checkpoint_path)
            logger.info(f"已保存修復後的檢查點: {new_checkpoint_path}")
            return new_checkpoint_path
        except Exception as e:
            logger.error(f"修復檢查點失敗: {e}")
            return None

    # ------------------------------
    # Minimal training stubs (prevent AttributeError)
    # ------------------------------
    def train(self):
        """簡易訓練占位：目前僅回傳 agent 以避免流程中斷"""
        logger.warning("train() 尚未實作完整訓練流程，暫時跳過並回傳當前模型。")
        return self.agent

    def train_parallel(self):
        """平行蒐集 episode → 主行程更新 PPO → 週期評估/保存。"""
        training_cfg = self.config.get('training', {})
        num_workers = int(training_cfg.get('num_workers', max(1, (mp.cpu_count() or 2) - 1)))
        episodes_per_update = int(training_cfg.get('episodes_per_update', 16))
        max_episodes = int(training_cfg.get('max_episodes', 100000))
        eval_frequency = int(training_cfg.get('eval_frequency', 200))
        eval_games = int(training_cfg.get('eval_games', 30))
        checkpoint_frequency = int(training_cfg.get('checkpoint_frequency', 1000))
        use_tactical_opp = bool(training_cfg.get('use_tactical_opponent_in_rollout', False))
        tactical_ratio = float(training_cfg.get('tactical_rollout_ratio', 0.0))
        win_scale = float(training_cfg.get('win_reward_scaling', 1.0))
        loss_scale = float(training_cfg.get('loss_penalty_scaling', 1.0))
        danger_scale = float(self.config.get('agent', {}).get('tactical_bonus', 0.1))
        min_batch_size = int(self.agent.config.get('min_batch_size', 512))

        rng = random.Random()
        best_score = -1.0
        # 盡量沿用已存在的 episode 計數（若從檢查點載入）
        episodes_done_total = len(self.episode_rewards)

        logger.info(f"🚀 啟動平行訓練：workers={num_workers}, episodes_per_update={episodes_per_update}")

        def _collect_batch(policy_state_cpu, n_episodes: int):
            args = []
            for i in range(n_episodes):
                use_tac = use_tactical_opp and (rng.random() < tactical_ratio)
                args.append({
                    'config': self.config,
                    'policy_state': policy_state_cpu,
                    'player2_training_prob': self.player2_training_prob,
                    'use_tactical_opponent': use_tac,
                    'seed': rng.randrange(2**31 - 1),
                })
            with mp.Pool(processes=num_workers) as pool:
                results = pool.map(_worker_collect_episode, args)
            return results

        while episodes_done_total < max_episodes:
            # 將模型權重搬到 CPU 供工作者複製
            policy_state_cpu = {k: v.detach().cpu() for k, v in self.agent.policy_net.state_dict().items()}

            results = _collect_batch(policy_state_cpu, episodes_per_update)
            collected_eps = 0

            for res in results:
                transitions = res.get('transitions', [])
                if not transitions:
                    continue
                player_result = int(res.get('player_result', 0))  # 1/0/-1
                ep_reward_sum = 0.0
                last_idx = len(transitions) - 1
                for idx, tr in enumerate(transitions):
                    state = tr['state']
                    action = int(tr['action'])
                    prob = float(tr['prob'])
                    reward = 0.0
                    # 結束時給最終勝負回饋
                    if idx == last_idx:
                        if player_result == 1:
                            reward += 1.0 * win_scale
                        elif player_result == -1:
                            reward -= 1.0 * loss_scale
                        else:
                            reward += 0.0
                    # 危險步懲罰（來自 worker 的快速戰術檢查）
                    if tr.get('is_dangerous', False):
                        reward += -10.0 * danger_scale
                    ep_reward_sum += reward
                    done = (idx == last_idx)
                    self.agent.store_transition(state, action, prob, reward, done)
                self.episode_rewards.append(ep_reward_sum)
                collected_eps += 1

            episodes_done_total += collected_eps

            # 當緩衝足夠大時才更新 PPO
            if len(self.agent.memory) >= min_batch_size:
                info = self.agent.update_policy()
                if info is not None:
                    self.training_losses.append(info.get('total_loss', 0.0))

            # 週期性評估
            if eval_frequency > 0 and episodes_done_total % eval_frequency == 0:
                metrics = self.evaluate_comprehensive(games=eval_games)
                score = float(metrics.get('comprehensive_score', 0.0))
                self.win_rates.append(score)
                # 調度學習率（以評估分數為目標）
                try:
                    self.agent.scheduler.step(score)
                except Exception:
                    pass
                logger.info(
                    f"📈 Eps={episodes_done_total} | Score={score:.3f} | "
                    f"self={metrics.get('self_play', 0):.3f} minimax={metrics.get('vs_minimax', 0):.3f} rand={metrics.get('vs_random', 0):.3f}"
                )
                # 偵測停滯
                try:
                    if self._detect_convergence_stagnation(episodes_done_total, score):
                        self._handle_convergence_stagnation(episodes_done_total)
                except Exception:
                    pass
                # 更新最佳模型
                if score > best_score:
                    best_score = score
                    self.save_checkpoint(f"best_model_wr_{best_score:.3f}.pt")

            # 週期性檢查點
            if checkpoint_frequency > 0 and episodes_done_total % checkpoint_frequency == 0:
                self.save_checkpoint(f"checkpoint_episode_{episodes_done_total}.pt")

        logger.info("✅ 平行訓練完成")
        return self.agent
def main():
    # 1) 準備 Trainer 與載入模型
    import os
    import glob
    import traceback
    from datetime import datetime
    import urllib.parse
    import urllib.request

    def find_latest_checkpoint() -> str | None:
        try:
            candidates = glob.glob(os.path.join("checkpoints", "*.pt"))
            if not candidates:
                return None
            return max(candidates, key=os.path.getmtime)
        except Exception:
            return None

    def send_telegram(msg: str):
        token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID") or "6166024220"  # 預設為先前使用的ID
        if not token or not chat_id:
            logger.info("未設置 TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID，略過訊息通知。")
            return
        try:
            base = f"https://api.telegram.org/bot{token}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": chat_id,
                "text": msg
            }).encode("utf-8")
            with urllib.request.urlopen(base, data=data, timeout=15) as resp:
                _ = resp.read()
            logger.info("已發送 Telegram 通知。")
        except Exception as e:
            logger.warning(f"Telegram 發送失敗: {e}")

    # 選擇設定檔
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

    # 允許透過環境變數指定要載入的 checkpoint
    resume_from_env = os.getenv("CHECKPOINT_PATH") or os.getenv("RESUME_FROM")
    resume_from_cfg = trainer.config.get('training', {}).get('resume_from')
    ckpt_to_load = resume_from_env or resume_from_cfg or find_latest_checkpoint()

    if ckpt_to_load:
        loaded = trainer.load_checkpoint(ckpt_to_load)
        if not loaded:
            logger.warning(f"無法載入檢查點: {ckpt_to_load}，將以隨機初始化開始。")
    else:
        logger.info("未找到可用檢查點，將以隨機初始化開始訓練。")

    # 2) 開始訓練
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

    # 3) 傳送訊息
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
