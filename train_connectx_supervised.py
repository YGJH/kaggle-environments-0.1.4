#!/usr/bin/env python3
"""
ConnectX 監督學習訓練腳本 - 簡化版本
使用 connectx-state-action-value.txt 資料集進行訓練
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import time
from datetime import datetime
from collections import deque
from kaggle_environments import make
from tqdm import tqdm

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConnectXNet(nn.Module):
    """強化學習用的 ConnectX 深度神經網路"""

    def __init__(self, input_size=126, hidden_size=200, num_layers=256):
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
        if not board:
            board = [0] * 42
        elif len(board) != 42:
            # 如果長度不對，調整或填充
            if len(board) < 42:
                board = list(board) + [0] * (42 - len(board))
            else:
                board = list(board)[:42]

        # 轉換為 6x7 矩陣
        state = np.array(board).reshape(6, 7)

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

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 神經網路
        self.policy_net = ConnectXNet(
            input_size=config['agent']['input_size'],
            hidden_size=config['agent']['hidden_size'],
            num_layers=config['agent']['num_layers']
        ).to(self.device)

        # 優化器
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config['agent']['learning_rate'],
            weight_decay=config['agent']['weight_decay']
        )

        # 學習率調度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,
            patience=10,
            min_lr=1e-8
        )

        # 創建目錄
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

    def encode_state(self, board, mark):
        """編碼棋盤狀態"""
        # 確保 board 是有效的
        if not board:
            board = [0] * 42
        elif len(board) != 42:
            if len(board) < 42:
                board = list(board) + [0] * (42 - len(board))
            else:
                board = list(board)[:42]

        # 轉換為 6x7 矩陣
        state = np.array(board).reshape(6, 7)

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

    def load_dataset(self, file_path="connectx-state-action-value.txt", max_lines=10000):
        """載入訓練數據集 - 記憶體優化版本"""
        skipped_lines = 0
        valid_samples = 0

        try:
            logger.info(f"載入訓練數據集: {file_path}")
            logger.info(f"限制載入行數: {max_lines}")

            # 第一次掃描：計算有效樣本數量
            logger.info("🔍 第一次掃描：計算有效樣本數量...")
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        parts = line.split(',')
                        if len(parts) < 8:
                            continue
                            
                        board_str = parts[0]
                        if len(board_str) != 42 or not all(c in '012' for c in board_str):
                            continue
                            
                        # 快速檢查動作價值
                        action_vals = []
                        for j in range(1, 8):
                            val_str = parts[j].strip()
                            if val_str == '':
                                action_vals.append(-999.0)
                            else:
                                try:
                                    action_vals.append(float(val_str))
                                except ValueError:
                                    action_vals.append(0.0)
                        
                        # 檢查有效動作
                        if any(val > -900 for val in action_vals):
                            valid_samples += 1
                            
                    except Exception:
                        continue

            logger.info(f"📊 掃描結果：預計 {valid_samples} 個有效樣本")
            
            if valid_samples == 0:
                logger.error("沒有找到任何有效樣本！")
                return None, None

            # 預分配記憶體
            states = np.zeros((valid_samples, 126), dtype=np.float32)
            action_values = np.zeros((valid_samples, 7), dtype=np.float32)
            
            # 第二次掃描：載入數據
            logger.info("📥 第二次掃描：載入數據...")
            sample_idx = 0
            
            with open(file_path, 'r') as f:
                with tqdm(total=min(max_lines, valid_samples), desc="載入數據") as pbar:
                    for i, line in enumerate(f):
                        if i >= max_lines or sample_idx >= valid_samples:
                            break
                        
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            # 解析一行數據
                            parts = line.split(',')
                            if len(parts) < 8:
                                skipped_lines += 1
                                continue

                            # 解析棋盤狀態
                            board_str = parts[0]
                            if len(board_str) != 42:
                                skipped_lines += 1
                                continue

                            # 直接轉換棋盤狀態（避免中間列表）
                            board_state = [int(c) for c in board_str if c in '012']
                            if len(board_state) != 42:
                                skipped_lines += 1
                                continue

                            # 解析動作價值（直接寫入數組）
                            action_vals = np.full(7, -999.0, dtype=np.float32)
                            for j in range(1, 8):
                                val_str = parts[j].strip()
                                if val_str != '':
                                    try:
                                        action_vals[j-1] = float(val_str)
                                    except ValueError:
                                        action_vals[j-1] = 0.0

                            # 檢查有效動作
                            if not np.any(action_vals > -900):
                                skipped_lines += 1
                                continue

                            # 編碼狀態（直接寫入預分配的數組）
                            encoded_state = self.encode_state(board_state, 1)
                            states[sample_idx] = encoded_state
                            action_values[sample_idx] = action_vals
                            
                            sample_idx += 1
                            pbar.update(1)

                        except Exception as e:
                            logger.debug(f"第 {i + 1} 行解析錯誤: {e}")
                            skipped_lines += 1
                            continue

            # 裁剪到實際使用的大小
            if sample_idx < valid_samples:
                states = states[:sample_idx]
                action_values = action_values[:sample_idx]

            logger.info(f"數據載入完成:")
            logger.info(f"  成功解析: {sample_idx} 個樣本")
            logger.info(f"  跳過行數: {skipped_lines}")
            logger.info(f"  記憶體使用: {states.nbytes / 1024 / 1024:.1f} MB (狀態) + {action_values.nbytes / 1024 / 1024:.1f} MB (動作值)")

            if sample_idx == 0:
                logger.error("沒有成功解析任何數據！")
                return None, None

            return states, action_values

        except FileNotFoundError:
            logger.error(f"找不到數據集文件: {file_path}")
            return None, None
        except Exception as e:
            logger.error(f"載入數據集時出錯: {e}")
            return None, None

    def train(self, epochs=100, batch_size=128, max_lines=10000, memory_efficient=True):
        """監督學習訓練 - 支援記憶體優化模式"""
        logger.info("🚀 開始監督學習訓練")

        if memory_efficient and max_lines > 20000:
            # 記憶體優化模式：分批載入訓練
            return self.train_memory_efficient(epochs, batch_size, max_lines)
        else:
            # 標準模式：一次載入所有數據
            return self.train_standard(epochs, batch_size, max_lines)

    def train_memory_efficient(self, epochs=100, batch_size=128, max_lines=10000):
        """記憶體優化的訓練模式"""
        logger.info("💾 使用記憶體優化模式訓練")
        
        # 分批載入參數
        chunk_size = min(10000, max_lines // 4)  # 每次載入1/4數據
        
        self.policy_net.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            num_batches = 0
            
            # 分塊處理數據
            for chunk_start in range(0, max_lines, chunk_size):
                chunk_end = min(chunk_start + chunk_size, max_lines)
                chunk_max_lines = chunk_end - chunk_start
                
                # 載入數據塊
                states, action_values = self.load_dataset_chunk(
                    file_path="connectx-state-action-value.txt",
                    start_line=chunk_start,
                    max_lines=chunk_max_lines
                )
                
                if states is None or len(states) == 0:
                    continue
                
                # 隨機打亂當前塊的數據
                indices = np.random.permutation(len(states))
                
                # 批次訓練當前數據塊
                for batch_start in range(0, len(states), batch_size):
                    batch_end = min(batch_start + batch_size, len(states))
                    batch_indices = indices[batch_start:batch_end]
                    
                    # 準備批次數據
                    batch_states = torch.FloatTensor(states[batch_indices]).to(self.device)
                    batch_action_values = torch.FloatTensor(action_values[batch_indices]).to(self.device)
                    
                    # 執行一個訓練步驟
                    loss_info = self.train_step(batch_states, batch_action_values)
                    
                    total_loss += loss_info['total_loss']
                    total_policy_loss += loss_info['policy_loss']
                    total_value_loss += loss_info['value_loss']
                    num_batches += 1
                
                # 清理記憶體
                del states, action_values
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 計算平均損失並記錄
            self.log_epoch_results(epoch, epochs, total_loss, total_policy_loss, 
                                 total_value_loss, num_batches, epoch_start_time, best_loss)
            
            # 更新最佳損失
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint("best_supervised_model.pt")
        
        logger.info("✅ 記憶體優化訓練完成")
        return self.policy_net

    def train_standard(self, epochs=100, batch_size=128, max_lines=10000):
        """標準訓練模式"""
        logger.info("🔄 使用標準模式訓練")
        
        # 載入數據集
        states, action_values = self.load_dataset(max_lines=max_lines)
        if states is None or action_values is None:
            logger.error("❌ 數據集載入失敗")
            return None

        logger.info(f"📊 數據集載入成功: {len(states)} 個樣本")

        # 訓練循環
        self.policy_net.train()
        best_loss = float('inf')

        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            num_batches = 0

            # 隨機打亂數據
            indices = np.random.permutation(len(states))

            # 批次訓練
            for batch_start in range(0, len(states), batch_size):
                batch_end = min(batch_start + batch_size, len(states))
                batch_indices = indices[batch_start:batch_end]

                # 準備批次數據
                batch_states = torch.FloatTensor(states[batch_indices]).to(self.device)
                batch_action_values = torch.FloatTensor(action_values[batch_indices]).to(self.device)

                # 執行一個訓練步驟
                loss_info = self.train_step(batch_states, batch_action_values)
                
                total_loss += loss_info['total_loss']
                total_policy_loss += loss_info['policy_loss']
                total_value_loss += loss_info['value_loss']
                num_batches += 1

            # 記錄結果
            self.log_epoch_results(epoch, epochs, total_loss, total_policy_loss, 
                                 total_value_loss, num_batches, epoch_start_time, best_loss)
            
            # 更新最佳損失
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint("best_supervised_model.pt")

        logger.info("✅ 標準訓練完成")
        return self.policy_net

    def train_step(self, batch_states, batch_action_values):
        """執行單個訓練步驟"""
        # 前向傳播
        predicted_probs, predicted_values = self.policy_net(batch_states)

        # 計算目標
        # 處理無效動作
        valid_mask = (batch_action_values > -900).float()
        masked_action_values = batch_action_values * valid_mask + (-1000) * (1 - valid_mask)

        # 轉換為目標概率分佈
        target_probs = F.softmax(masked_action_values / 0.5, dim=1)  # 溫度參數

        # 價值目標：最大動作價值
        target_values = torch.max(batch_action_values * valid_mask + (-1000) * (1 - valid_mask), dim=1)[0].unsqueeze(1)
        target_values = torch.tanh(target_values / 10.0)  # 正規化到[-1,1]

        # 計算損失
        policy_loss = F.kl_div(torch.log(predicted_probs + 1e-8), target_probs, reduction='batchmean')
        value_loss = F.mse_loss(predicted_values, target_values)
        total_loss_batch = policy_loss + 0.5 * value_loss

        # 反向傳播
        self.optimizer.zero_grad()
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
        self.optimizer.step()

        return {
            'total_loss': total_loss_batch.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }

    def log_epoch_results(self, epoch, epochs, total_loss, total_policy_loss, 
                         total_value_loss, num_batches, epoch_start_time, best_loss):
        """記錄epoch結果"""
        # 計算平均損失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_policy_loss = total_policy_loss / num_batches if num_batches > 0 else 0
        avg_value_loss = total_value_loss / num_batches if num_batches > 0 else 0
        
        epoch_time = time.time() - epoch_start_time

        # 學習率調度
        self.scheduler.step(avg_loss)

        # 每10個epoch報告一次
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}] "
                      f"Loss: {avg_loss:.6f} "
                      f"Policy: {avg_policy_loss:.6f} "
                      f"Value: {avg_value_loss:.6f} "
                      f"Time: {epoch_time:.2f}s")

        # 保存定期檢查點
        if (epoch + 1) % 50 == 0:
            checkpoint_name = f"supervised_epoch_{epoch+1}.pt"
            self.save_checkpoint(checkpoint_name)

    def load_dataset_chunk(self, file_path="connectx-state-action-value.txt", start_line=0, max_lines=10000):
        """載入數據集的指定塊"""
        states = []
        action_values = []
        skipped_lines = 0
        current_line = 0

        try:
            with open(file_path, 'r') as f:
                # 跳過開始行
                for _ in range(start_line):
                    f.readline()
                
                # 讀取指定行數
                for i in range(max_lines):
                    line = f.readline()
                    if not line:  # 文件結束
                        break
                    
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        # 解析邏輯與原來相同
                        parts = line.split(',')
                        if len(parts) < 8:
                            skipped_lines += 1
                            continue

                        board_str = parts[0]
                        if len(board_str) != 42:
                            skipped_lines += 1
                            continue

                        board_state = [int(c) for c in board_str if c in '012']
                        if len(board_state) != 42:
                            skipped_lines += 1
                            continue

                        action_vals = []
                        for j in range(1, 8):
                            val_str = parts[j].strip()
                            if val_str == '':
                                action_vals.append(-999.0)
                            else:
                                try:
                                    action_vals.append(float(val_str))
                                except ValueError:
                                    action_vals.append(0.0)

                        if not any(val > -900 for val in action_vals):
                            skipped_lines += 1
                            continue

                        encoded_state = self.encode_state(board_state, 1)
                        states.append(encoded_state)
                        action_values.append(action_vals)

                    except Exception:
                        skipped_lines += 1
                        continue

            if len(states) == 0:
                return None, None

            return np.array(states, dtype=np.float32), np.array(action_values, dtype=np.float32)

        except Exception as e:
            logger.error(f"載入數據塊時出錯: {e}")
            return None, None

    def save_checkpoint(self, filename):
        """保存檢查點"""
        try:
            checkpoint = {
                'model_state_dict': self.policy_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'save_timestamp': datetime.now().isoformat(),
                'pytorch_version': str(torch.__version__)
            }

            checkpoint_path = f"checkpoints/{filename}"
            torch.save(checkpoint, checkpoint_path)

            file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
            logger.info(f"✅ 已保存檢查點: {filename} ({file_size:.2f} MB)")

        except Exception as e:
            logger.error(f"保存檢查點時出錯: {e}")

    def evaluate_random_games(self, num_games=100):
        """評估模型對隨機對手的性能"""
        logger.info(f"🎯 評估模型性能 ({num_games} 局遊戲)")

        self.policy_net.eval()
        wins = 0
        draws = 0
        losses = 0

        for i in range(num_games):
            try:
                # 創建遊戲環境
                env = make("connectx", debug=False)
                env.reset()

                # 簡單的遊戲循環
                done = False
                step_count = 0
                max_steps = 42

                while not done and step_count < max_steps:
                    current_player = step_count % 2

                    if current_player == 0:  # AI玩家
                        # 獲取當前狀態
                        obs = env.state[0]['observation']
                        board = obs['board']
                        mark = obs['mark']

                        # 編碼狀態
                        encoded_state = self.encode_state(board, mark)
                        state_tensor = torch.FloatTensor(encoded_state).unsqueeze(0).to(self.device)

                        # 獲取動作概率
                        with torch.no_grad():
                            action_probs, _ = self.policy_net(state_tensor)

                        # 選擇動作（貪婪策略）
                        valid_actions = [c for c in range(7) if board[c] == 0]
                        if valid_actions:
                            masked_probs = action_probs.cpu().numpy()[0]
                            valid_probs = [masked_probs[a] for a in valid_actions]
                            best_idx = np.argmax(valid_probs)
                            action = valid_actions[best_idx]
                        else:
                            action = 0

                    else:  # 隨機對手
                        obs = env.state[1]['observation']
                        board = obs['board']
                        valid_actions = [c for c in range(7) if board[c] == 0]
                        action = np.random.choice(valid_actions) if valid_actions else 0

                    # 執行動作
                    env.step([action, None] if current_player == 0 else [None, action])

                    # 檢查遊戲結束
                    if len(env.state) >= 2:
                        status_0 = env.state[0].get('status', 'ACTIVE')
                        status_1 = env.state[1].get('status', 'ACTIVE')

                        if status_0 != 'ACTIVE' or status_1 != 'ACTIVE':
                            done = True

                    step_count += 1

                # 計算結果
                if len(env.state) >= 2:
                    reward_0 = env.state[0].get('reward', 0)
                    reward_1 = env.state[1].get('reward', 0)

                    if reward_0 > reward_1:
                        wins += 1
                    elif reward_1 > reward_0:
                        losses += 1
                    else:
                        draws += 1

                if (i + 1) % 20 == 0:
                    current_wr = wins / (i + 1) * 100
                    logger.info(f"評估進度: {i+1}/{num_games}, 當前勝率: {current_wr:.1f}%")

            except Exception as e:
                logger.error(f"評估第 {i+1} 局時出錯: {e}")
                losses += 1

        win_rate = wins / num_games * 100
        logger.info(f"📊 評估結果:")
        logger.info(f"   勝利: {wins} ({win_rate:.1f}%)")
        logger.info(f"   平局: {draws} ({draws/num_games*100:.1f}%)")
        logger.info(f"   失敗: {losses} ({losses/num_games*100:.1f}%)")

        return win_rate

def create_config():
    """創建訓練配置"""
    config = {
        'agent': {
            'input_size': 126,      # 3個通道 × 42個位置
            'hidden_size': 150,     # 隱藏層大小
            'num_layers': 3,        # 隱藏層數量（修正為合理值）
            'learning_rate': 0.001, # 學習率
            'weight_decay': 0.0001  # 權重衰減
        },
        'training': {
            'epochs': 200,          # 訓練epochs
            'batch_size': 128,      # 批次大小
            'max_lines': 50000,     # 最大數據集行數
            'eval_games': 100,      # 評估遊戲數量
            'memory_efficient': True # 是否使用記憶體優化模式
        }
    }
    return config

def main():
    """主訓練函數"""
    print("🎮 ConnectX 監督學習訓練")
    print("=" * 50)

    # 創建必要目錄
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # 檢查數據集文件
    dataset_file = "connectx-state-action-value.txt"
    if not os.path.exists(dataset_file):
        logger.error(f"❌ 找不到數據集文件: {dataset_file}")
        return

    # 創建配置
    config = create_config()

    # 檢查設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🔧 使用設備: {device}")

    try:
        # 創建訓練器
        trainer = ConnectXTrainer(config)
        logger.info("✅ 訓練器創建成功")

        # 顯示配置
        print("\n📋 訓練配置:")
        print(f"   網絡結構: {config['agent']['hidden_size']} 隱藏單元, {config['agent']['num_layers']} 層")
        print(f"   學習率: {config['agent']['learning_rate']}")
        print(f"   訓練epochs: {config['training']['epochs']}")
        print(f"   批次大小: {config['training']['batch_size']}")
        print(f"   最大數據集行數: {config['training']['max_lines']}")

        # 開始訓練
        print("\n🚀 開始監督學習訓練...")
        start_time = time.time()

        trained_model = trainer.train(
            epochs=config['training']['epochs'],
            batch_size=config['training']['batch_size'],
            max_lines=config['training']['max_lines'],
            memory_efficient=config['training']['memory_efficient']
        )

        training_time = time.time() - start_time

        if trained_model is not None:
            logger.info(f"✅ 訓練完成！用時: {training_time:.1f}秒")

            # 保存最終模型
            final_checkpoint = f"supervised_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            trainer.save_checkpoint(final_checkpoint)

            # 評估模型
            print("\n🎯 評估模型性能...")
            win_rate = trainer.evaluate_random_games(num_games=config['training']['eval_games'])

            print(f"\n🎉 訓練完成!")
            print(f"   總用時: {training_time:.1f}秒 ({training_time/60:.1f}分鐘)")
            print(f"   最終勝率: {win_rate:.1f}%")
            print(f"   模型保存位置: checkpoints/{final_checkpoint}")

            # 使用建議
            if win_rate >= 80:
                print("\n🌟 模型性能優異！可以用於比賽")
            elif win_rate >= 60:
                print("\n👍 模型性能良好，建議進行更多訓練")
            else:
                print("\n⚠️ 模型性能需要改進，建議增加訓練時間或調整參數")

        else:
            logger.error("❌ 訓練失敗")

    except KeyboardInterrupt:
        logger.info("⏹️ 訓練被用戶中斷")
    except Exception as e:
        logger.error(f"❌ 訓練過程中出錯: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
