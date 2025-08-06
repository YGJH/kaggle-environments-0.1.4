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
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("Matplotlib not found. Installing for visualization...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.animation import FuncAnimation
        import matplotlib.colors as mcolors
        VISUALIZATION_AVAILABLE = True
    except Exception:
        print("⚠️ 無法安裝matplotlib，將跳過可視化功能")
        VISUALIZATION_AVAILABLE = False

from kaggle_environments import make, evaluate

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
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

        # 持續學習數據
        self.continuous_learning_data = None
        self.continuous_learning_targets = None

        # 創建目錄
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

    def load_state_action_dataset(self, file_path="connectx-state-action-value.txt", max_lines=-1):
        """
        載入狀態-動作價值數據集

        格式說明：
        - 每行包含：棋盤狀態(42字符) + 7個動作價值（逗號分隔）
        - 棋盤狀態：0=空，1=先手，2=後手（從左到右，從上到下）
        - 動作價值：正數=先手贏（步數），負數=後手贏，0=平局，空=無效動作
        """
        states = []
        action_values = []
        skipped_lines = 0

        try:
            logger.info(f"載入訓練數據集: {file_path}")

            # 如果沒有限制行數，先快速掃描文件總行數
            if max_lines == -1:
                logger.info("正在掃描數據集總行數...")
                with open(file_path, 'r') as f:
                    total_lines = sum(1 for _ in f)
                max_lines = total_lines
                logger.info(f"數據集總行數: {total_lines}")
            else:
                logger.info(f"限制載入行數: {max_lines}")

            # 只讀取需要的行數，避免記憶體問題
            lines = []
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line)

            logger.info(f"實際載入行數: {len(lines)}")
            try:
                from tqdm.auto import tqdm
            except ImportError:
                from tqdm import tqdm

            with tqdm(total=len(lines), desc="read dataset") as pbar:
                for line_idx, line in enumerate(lines):
                    pbar.update(1)

                    line = line.strip()
                    if not line:
                        continue

                    if line_idx >= max_lines:
                        break
                    try:
                        # 找到第一個逗號的位置，分割棋盤狀態和動作價值
                        comma_idx = line.find(',')
                        if comma_idx == -1:
                            logger.warning(f"第 {line_idx + 1} 行格式錯誤，找不到逗號分隔符")
                            skipped_lines += 1
                            continue

                        # 解析棋盤狀態 (逗號前的42個字符)
                        board_part = line[:comma_idx]
                        if len(board_part) != 42:
                            logger.warning(f"第 {line_idx + 1} 行棋盤狀態長度錯誤，期望42個字符，得到{len(board_part)}個")
                            skipped_lines += 1
                            continue

                        # 驗證棋盤狀態只包含 0, 1, 2
                        try:
                            board_state = []
                            for char in board_part:
                                if char not in '012':
                                    raise ValueError(f"無效字符: {char}")
                                board_state.append(int(char))
                        except ValueError as e:
                            logger.warning(f"第 {line_idx + 1} 行棋盤狀態包含無效字符: {e}")
                            skipped_lines += 1
                            continue

                        # 解析動作價值 (逗號後的7個值)
                        action_part = line[comma_idx+1:]
                        action_parts = action_part.split(',')

                        if len(action_parts) != 7:
                            logger.warning(f"第 {line_idx + 1} 行動作價值數量錯誤，期望7個值，得到{len(action_parts)}個")
                            skipped_lines += 1
                            continue

                        # 處理動作價值（包括空值）
                        action_vals = []
                        for i, val_str in enumerate(action_parts):
                            val_str = val_str.strip()

                            if val_str == '':
                                # 空值表示該列已滿，設為極大負值（不可下）
                                action_vals.append(-999)
                            else:
                                try:
                                    # 嘗試轉換為數字
                                    val = float(val_str)
                                    # 檢查合理範圍（ConnectX 最多42步）
                                    if abs(val) > 50:
                                        logger.warning(f"第 {line_idx + 1} 行列 {i} 的價值超出合理範圍: {val}")
                                    action_vals.append(val)
                                except ValueError:
                                    logger.warning(f"第 {line_idx + 1} 行列 {i} 的價值無法解析: '{val_str}'，設為0")
                                    action_vals.append(0.0)

                        # 數據質量檢查
                        valid_actions = [i for i, val in enumerate(action_vals) if val > -900]
                        if len(valid_actions) == 0:
                            logger.warning(f"第 {line_idx + 1} 行沒有有效動作，跳過")
                            skipped_lines += 1
                            continue

                        # 將棋盤狀態轉換為我們的編碼格式
                        # 注意：數據集中的先手玩家標記為1，這與我們的編碼一致
                        encoded_state = self.agent.encode_state(board_state, 1)

                        states.append(encoded_state)
                        action_values.append(action_vals)

                        # 每10000行報告一次進度
                      #  if (line_idx + 1) % 10000 == 0:
                       #     logger.info(f"已處理 {line_idx + 1} 行數據，成功解析 {len(states)} 個樣本")

                        # 調試：只顯示前幾行的詳細信息
                     #   if line_idx < 3:
                     #       logger.debug(f"樣本 {line_idx + 1}: 棋盤前10位={board_part[:10]}..., 動作價值={action_vals}")

                    except (ValueError, IndexError) as e:
                        logger.warning(f"第 {line_idx + 1} 行解析錯誤: {e}")
                        skipped_lines += 1
                        continue

            logger.info(f"數據載入完成:")
            logger.info(f"  總行數: {len(lines)}")
            logger.info(f"  成功解析: {len(states)} 個樣本")
            logger.info(f"  跳過行數: {skipped_lines}")
            logger.info(f"  成功率: {len(states)/(len(lines)-skipped_lines)*100:.1f}%")

            if len(states) == 0:
                logger.error("沒有成功解析任何數據樣本！")
                return None, None

            # 轉換為numpy數組並清理臨時變量以節省記憶體
            states_array = np.array(states)
            action_values_array = np.array(action_values)

            # 清理臨時變量
            del states, action_values, lines

            logger.info(f"數據載入完成，記憶體使用狀態：")
            logger.info(f"  狀態數組形狀: {states_array.shape}")
            logger.info(f"  動作價值數組形狀: {action_values_array.shape}")

            return states_array, action_values_array

        except FileNotFoundError:
            logger.error(f"找不到數據集文件: {file_path}")
            return None, None
        except Exception as e:
            logger.error(f"載入數據集時出錯: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def supervised_pretrain_batch_loading(self, file_path="connectx-state-action-value.txt", epochs=100, batch_size=128, memory_batch_size=10000, max_lines=-1):
        """
        使用批次載入進行監督學習預訓練

        優點：
        - 記憶體友善：只載入需要的數據批次
        - 支援大數據集：不受記憶體限制
        - 動態處理：邊載入邊訓練

        參數：
        - file_path: 數據集文件路徑
        - epochs: 訓練輪數
        - batch_size: 每個訓練批次的大小
        - memory_batch_size: 一次載入到記憶體的數據量
        """
        logger.info("開始批次載入監督學習預訓練...")

        if not os.path.exists(file_path):
            logger.error(f"找不到數據集文件: {file_path}")
            return False

        # 首先掃描文件獲取總行數
        logger.info("正在掃描數據集...")
        with open(file_path, 'r') as f:
            total_lines = sum(1 for line in f if line.strip())

        if max_lines == -1:
            max_lines = total_lines
        total_lines = min(max_lines, total_lines)
        logger.info(f"數據集總行數: {total_lines}")
        logger.info(f"記憶體批次大小: {memory_batch_size}")
        logger.info(f"訓練批次大小: {batch_size}")

        # 創建監督學習優化器
        supervised_lr = self.config['agent'].get('supervised_learning_rate', 1e-4)
        supervised_optimizer = optim.Adam(
            self.agent.policy_net.parameters(),
            lr=supervised_lr,
            weight_decay=self.config['agent']['weight_decay']
        )

        best_loss = float('inf')

        try:
            from tqdm.auto import tqdm
        except ImportError:
            from tqdm import tqdm


        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            epoch_losses = []
            epoch_policy_losses = []
            epoch_value_losses = []
            processed_samples = 0

            # 重新打開文件進行新的epoch
            with open(file_path, 'r') as f:
                current_batch_states = []
                current_batch_targets = []

                with tqdm(total=total_lines, desc=f"Epoch {epoch+1}") as pbar:
                    for line_idx, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue

                        if line_idx >= total_lines:
                            break
                        # 解析數據（復用之前的解析邏輯）
                        parsed_data = self._parse_line(line, line_idx)
                        if parsed_data is None:
                            pbar.update(1)
                            continue

                        state, action_vals = parsed_data

                        # 處理為訓練目標
                        processed_target = self._process_training_target(state, action_vals)
                        if processed_target is None:
                            pbar.update(1)
                            continue

                        processed_state, policy_target, value_target = processed_target
                        current_batch_states.append(processed_state)
                        current_batch_targets.append((policy_target, value_target))

                        # 當累積到記憶體批次大小時，進行訓練
                        if len(current_batch_states) >= memory_batch_size:
                            batch_losses = self._train_on_batch(
                                current_batch_states,
                                current_batch_targets,
                                supervised_optimizer,
                                batch_size,
                                max_lines=max_lines
                            )

                            epoch_losses.extend(batch_losses['total_losses'])
                            epoch_policy_losses.extend(batch_losses['policy_losses'])
                            epoch_value_losses.extend(batch_losses['value_losses'])
                            processed_samples += len(current_batch_states)

                            # 清空批次
                            current_batch_states = []
                            current_batch_targets = []

                        pbar.update(1)
                        pbar.set_postfix({
                            'processed': processed_samples,
                            'batch_size': len(current_batch_states)
                        })

                # 處理剩餘的數據
                if current_batch_states:
                    batch_losses = self._train_on_batch(
                        current_batch_states,
                        current_batch_targets,
                        supervised_optimizer,
                        batch_size,
                        max_lines=max_lines
                    )

                    epoch_losses.extend(batch_losses['total_losses'])
                    epoch_policy_losses.extend(batch_losses['policy_losses'])
                    epoch_value_losses.extend(batch_losses['value_losses'])
                    processed_samples += len(current_batch_states)

            # 計算epoch平均損失
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                avg_policy_loss = np.mean(epoch_policy_losses)
                avg_value_loss = np.mean(epoch_value_losses)

                logger.info(f"Epoch {epoch+1} 完成:")
                logger.info(f"  處理樣本數: {processed_samples}")
                logger.info(f"  總損失: {avg_loss:.4f}")
                logger.info(f"  策略損失: {avg_policy_loss:.4f}")
                logger.info(f"  價值損失: {avg_value_loss:.4f}")

                # 保存最佳模型
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.save_checkpoint("pretrained_best.pt")
                    logger.info(f"✅ 保存最佳模型 (損失: {best_loss:.4f})")
            else:
                logger.warning(f"Epoch {epoch+1} 沒有處理任何有效數據")

        logger.info(f"批次載入預訓練完成！")
        logger.info(f"  最佳總損失: {best_loss:.4f}")

        # 載入最佳預訓練模型
        success = self.load_checkpoint("checkpoints/pretrained_best.pt")
        if success:
            logger.info("✅ 最佳預訓練模型載入成功")
        else:
            logger.warning("⚠️ 最佳預訓練模型載入失敗，使用當前模型")

        return True

    def _parse_line(self, line, line_idx):
        """解析數據集中的一行"""
        try:
            # 找到第一個逗號的位置，分割棋盤狀態和動作價值
            comma_idx = line.find(',')
            if comma_idx == -1:
                return None

            # 解析棋盤狀態 (逗號前的42個字符)
            board_part = line[:comma_idx]
            if len(board_part) != 42:
                return None

            # 驗證棋盤狀態只包含 0, 1, 2
            try:
                board_state = []
                for char in board_part:
                    if char not in '012':
                        raise ValueError(f"無效字符: {char}")
                    board_state.append(int(char))
            except ValueError:
                return None

            # 解析動作價值 (逗號後的7個值)
            action_part = line[comma_idx+1:]
            action_parts = action_part.split(',')

            if len(action_parts) != 7:
                return None

            # 處理動作價值（包括空值）
            action_vals = []
            for i, val_str in enumerate(action_parts):
                val_str = val_str.strip()

                if val_str == '':
                    # 空值表示該列已滿，設為極大負值（不可下）
                    action_vals.append(-999)
                else:
                    try:
                        # 嘗試轉換為數字
                        val = float(val_str)
                        # 檢查合理範圍（ConnectX 最多42步）
                        if abs(val) > 50:
                            pass  # 警告但不跳過
                        action_vals.append(val)
                    except ValueError:
                        action_vals.append(0.0)

            # 數據質量檢查
            valid_actions = [i for i, val in enumerate(action_vals) if val > -900]
            if len(valid_actions) == 0:
                return None

            # 將棋盤狀態轉換為我們的編碼格式
            encoded_state = self.agent.encode_state(board_state, 1)

            return encoded_state, action_vals

        except Exception:
            return None

    def _process_training_target(self, state, action_vals):
        """處理訓練目標"""
        try:
            # 過濾無效動作（值為-999的列已滿）
            valid_mask = np.array(action_vals) > -900

            if not np.any(valid_mask):
                return None

            # 創建策略目標：更好的動作得到更高的概率
            policy_target = np.zeros(7)

            # 將價值轉換為偏好分數
            preferences = np.array(action_vals).copy()

            # 無效動作設為極小值
            preferences[~valid_mask] = -1000

            # 使用 softmax 創建策略分佈（溫度參數控制尖銳度）
            temperature = 2.0
            exp_prefs = np.exp(preferences / temperature)
            exp_prefs[~valid_mask] = 0  # 確保無效動作概率為0

            if np.sum(exp_prefs) > 0:
                policy_target = exp_prefs / np.sum(exp_prefs)
            else:
                # 備用方案：均勻分佈於有效動作
                policy_target[valid_mask] = 1.0 / np.sum(valid_mask)

            # 創建價值目標：使用最佳動作的價值
            valid_values = np.array(action_vals)[valid_mask]
            if len(valid_values) > 0:
                # 使用最大價值作為狀態價值（先手視角）
                state_value = np.max(valid_values)
                # 正規化到 [-1, 1] 範圍
                state_value = np.tanh(state_value / 10.0)
            else:
                state_value = 0.0

            return state, policy_target, state_value

        except Exception:
            return None

    def _train_on_batch(self, states, targets, optimizer, batch_size , max_lines=-1):
        """在一個記憶體批次上進行訓練"""
        if not states or not targets:
            return {'total_losses': [], 'policy_losses': [], 'value_losses': []}

        # 分離策略和價值目標
        policy_targets = [t[0] for t in targets]
        value_targets = [t[1] for t in targets]


        # 轉換為張量
        states_tensor = torch.FloatTensor(np.array(states)).to(self.agent.device)
        policies_tensor = torch.FloatTensor(np.array(policy_targets)).to(self.agent.device)
        values_tensor = torch.FloatTensor(np.array(value_targets)).unsqueeze(1).to(self.agent.device)

        if max_lines == -1:
          max_lines = len(states)
        dataset_size = min(max_lines ,len(states))
        total_losses = []
        policy_losses = []
        value_losses = []

        # 隨機打亂數據
        indices = torch.randperm(dataset_size)

        for i in range(0, dataset_size, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_states = states_tensor[batch_indices]
            batch_target_policies = policies_tensor[batch_indices]
            batch_target_values = values_tensor[batch_indices]

            # 前向傳播
            predicted_policies, predicted_values = self.agent.policy_net(batch_states)

            # 策略損失：使用交叉熵損失
            policy_loss = nn.KLDivLoss(reduction='batchmean')(
                torch.log(predicted_policies + 1e-8),
                batch_target_policies
            )

            # 價值損失：使用MSE損失
            value_loss = nn.MSELoss()(predicted_values, batch_target_values)

            # 總損失
            total_loss = policy_loss + 0.5 * value_loss

            # 反向傳播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.policy_net.parameters(), 1.0)
            optimizer.step()

            total_losses.append(total_loss.item())
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

        return {
            'total_losses': total_losses,
            'policy_losses': policy_losses,
            'value_losses': value_losses
        }

    def supervised_pretrain(self, states, action_values, epochs=100, batch_size=128):
        """
        使用監督學習進行預訓練 (原版，保留向後兼容)

        根據完美遊戲數據集訓練策略和價值網路：
        - 策略網路：學習最優動作選擇
        - 價值網路：學習位置評估
        """
        logger.info("開始監督學習預訓練...")

        if states is None or action_values is None:
            logger.error("無效的訓練數據")
            return False

        logger.info(f"預訓練數據統計:")
        logger.info(f"  樣本數量: {len(states)}")
        logger.info(f"  狀態維度: {states.shape}")
        logger.info(f"  動作價值維度: {action_values.shape}")

        # 數據預處理：創建訓練目標
        processed_states = []
        target_policies = []
        target_values = []

        try:
            from tqdm.auto import tqdm
        except ImportError:
            from tqdm import tqdm

        with tqdm(total=len(states), desc="preprocessing data") as pbar:
            for i, (state, action_vals) in enumerate(zip(states, action_values)):
                pbar.update(1)

                # 過濾無效動作（值為-999的列已滿）
                valid_mask = np.array(action_vals) > -900

                if not np.any(valid_mask):
                    continue  # 跳過沒有有效動作的狀態

                # 創建策略目標：更好的動作得到更高的概率
                policy_target = np.zeros(7)

                # 將價值轉換為偏好分數
                # 正值（先手贏）= 好動作，負值（先手輸）= 壞動作，0（平局）= 中性
                preferences = np.array(action_vals).copy()

                # 無效動作設為極小值
                preferences[~valid_mask] = -1000

                # 使用 softmax 創建策略分佈（溫度參數控制尖銳度）
                temperature = 2.0
                exp_prefs = np.exp(preferences / temperature)
                exp_prefs[~valid_mask] = 0  # 確保無效動作概率為0

                if np.sum(exp_prefs) > 0:
                    policy_target = exp_prefs / np.sum(exp_prefs)
                else:
                    # 備用方案：均勻分佈於有效動作
                    policy_target[valid_mask] = 1.0 / np.sum(valid_mask)

                # 創建價值目標：使用最佳動作的價值
                valid_values = np.array(action_vals)[valid_mask]
                if len(valid_values) > 0:
                    # 使用最大價值作為狀態價值（先手視角）
                    state_value = np.max(valid_values)
                    # 正規化到 [-1, 1] 範圍
                    state_value = np.tanh(state_value / 10.0)
                else:
                    state_value = 0.0

                processed_states.append(state)
                target_policies.append(policy_target)
                target_values.append(state_value)

        if len(processed_states) == 0:
            logger.error("沒有有效的訓練樣本！")
            return False

        logger.info(f"預處理後有效樣本數: {len(processed_states)}")

        # 轉換為張量
        states_tensor = torch.FloatTensor(np.array(processed_states)).to(self.agent.device)
        policies_tensor = torch.FloatTensor(np.array(target_policies)).to(self.agent.device)
        values_tensor = torch.FloatTensor(np.array(target_values)).unsqueeze(1).to(self.agent.device)

        # 創建監督學習優化器
        supervised_lr = self.config['agent'].get('supervised_learning_rate', 1e-4)
        supervised_optimizer = optim.Adam(
            self.agent.policy_net.parameters(),
            lr=supervised_lr,
            weight_decay=self.config['agent']['weight_decay']
        )

        dataset_size = len(processed_states)
        best_loss = float('inf')

        for epoch in range(epochs):
            epoch_losses = []
            epoch_policy_losses = []
            epoch_value_losses = []

            # 隨機打亂數據
            indices = torch.randperm(dataset_size)

            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_states = states_tensor[batch_indices]
                batch_target_policies = policies_tensor[batch_indices]
                batch_target_values = values_tensor[batch_indices]

                # 前向傳播
                predicted_policies, predicted_values = self.agent.policy_net(batch_states)

                # 策略損失：使用交叉熵損失
                policy_loss = nn.KLDivLoss(reduction='batchmean')(
                    torch.log(predicted_policies + 1e-8),
                    batch_target_policies
                )

                # 價值損失：使用MSE損失
                value_loss = nn.MSELoss()(predicted_values, batch_target_values)

                # 總損失
                total_loss = policy_loss + 0.5 * value_loss

                # 反向傳播
                supervised_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.policy_net.parameters(), 1.0)
                supervised_optimizer.step()

                epoch_losses.append(total_loss.item())
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())

            avg_loss = np.mean(epoch_losses)
            avg_policy_loss = np.mean(epoch_policy_losses)
            avg_value_loss = np.mean(epoch_value_losses)

            # 記錄進度
            if epoch % 10 == 0:
                logger.info(f"預訓練 Epoch {epoch}/{epochs}")
                logger.info(f"  總損失: {avg_loss:.4f}")
                logger.info(f"  策略損失: {avg_policy_loss:.4f}")
                logger.info(f"  價值損失: {avg_value_loss:.4f}")

            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint("pretrained_best.pt")

        logger.info(f"預訓練完成！")
        logger.info(f"  最佳總損失: {best_loss:.4f}")
        logger.info(f"  最終策略損失: {avg_policy_loss:.4f}")
        logger.info(f"  最終價值損失: {avg_value_loss:.4f}")

        # 載入最佳預訓練模型
        success = self.load_checkpoint("checkpoints/pretrained_best.pt")
        if success:
            logger.info("✅ 最佳預訓練模型載入成功")
        else:
            logger.warning("⚠️ 最佳預訓練模型載入失敗，使用當前模型")

        return True

    def continuous_supervised_learning(self, states, action_values, batch_size=64, learning_rate=1e-5):
        """
        在強化學習過程中持續進行監督學習
        使用較小的批次和學習率，避免過度干擾RL訓練
        """
        if states is None or action_values is None or len(states) == 0:
            return False

        # 隨機採樣一個批次
        indices = np.random.choice(len(states), min(batch_size, len(states)), replace=False)
        batch_states = states[indices]
        batch_action_values = action_values[indices]

        # 處理批次數據
        processed_states = []
        target_policies = []
        target_values = []

        for state, action_vals in zip(batch_states, batch_action_values):
            # 過濾無效動作
            valid_mask = np.array(action_vals) > -900

            if not np.any(valid_mask):
                continue

            # 創建策略目標
            policy_target = np.zeros(7)
            preferences = np.array(action_vals).copy()
            preferences[~valid_mask] = -1000

            # 使用較高的溫度，讓分佈更平滑（減少對RL的干擾）
            temperature = 3.0
            exp_prefs = np.exp(preferences / temperature)
            exp_prefs[~valid_mask] = 0

            if np.sum(exp_prefs) > 0:
                policy_target = exp_prefs / np.sum(exp_prefs)
            else:
                policy_target[valid_mask] = 1.0 / np.sum(valid_mask)

            # 創建價值目標
            valid_values = np.array(action_vals)[valid_mask]
            if len(valid_values) > 0:
                state_value = np.max(valid_values)
                state_value = np.tanh(state_value / 10.0)
            else:
                state_value = 0.0

            processed_states.append(state)
            target_policies.append(policy_target)
            target_values.append(state_value)

        if len(processed_states) == 0:
            return False

        # 轉換為張量
        states_tensor = torch.FloatTensor(np.array(processed_states)).to(self.agent.device)
        policies_tensor = torch.FloatTensor(np.array(target_policies)).to(self.agent.device)
        values_tensor = torch.FloatTensor(np.array(target_values)).unsqueeze(1).to(self.agent.device)

        # 創建臨時優化器（使用較小的學習率）
        temp_optimizer = optim.Adam(
            self.agent.policy_net.parameters(),
            lr=learning_rate,
            weight_decay=self.config['agent']['weight_decay']
        )

        # 前向傳播
        predicted_policies, predicted_values = self.agent.policy_net(states_tensor)

        # 計算損失
        policy_loss = nn.KLDivLoss(reduction='batchmean')(
            torch.log(predicted_policies + 1e-8),
            policies_tensor
        )
        value_loss = nn.MSELoss()(predicted_values, values_tensor)

        # 使用較小的權重避免干擾RL學習
        total_loss = 0.3 * policy_loss + 0.1 * value_loss

        # 反向傳播
        temp_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.policy_net.parameters(), 0.5)
        temp_optimizer.step()

        return {
            'supervised_policy_loss': policy_loss.item(),
            'supervised_value_loss': value_loss.item(),
            'supervised_total_loss': total_loss.item()
        }

    def enable_continuous_learning(self, states, action_values):
        """
        啟用持續學習模式，儲存數據集供RL訓練期間使用
        """
        if states is not None and action_values is not None:
            self.continuous_learning_data = states
            self.continuous_learning_targets = action_values
            logger.info(f"已啟用持續學習模式，載入 {len(states)} 個訓練樣本")
            return True
        return False

    def play_game(self, agent1_func, agent2_func, training=True):
        """進行一場遊戲"""
        env = make("connectx", debug=False)
        env.reset()

        episode_transitions = []
        move_count = 0
        max_moves = 50  # 防止無限循環

        while not env.done and move_count < max_moves:
            # 為兩個玩家獲取動作
            actions = []

            for player_idx in range(2):
                if env.state[player_idx]['status'] == 'ACTIVE':
                    # 使用新的提取方法
                    board, current_player = self.agent.extract_board_and_mark(env.state, player_idx)

                    # 編碼狀態
                    state = self.agent.encode_state(board, current_player)
                    valid_actions = self.agent.get_valid_actions(board)

                    # 選擇動作
                    if current_player == 1:
                        result = agent1_func(state, valid_actions, training)
                    else:
                        result = agent2_func(state, valid_actions, training)

                    # 處理不同的返回格式
                    if len(result) == 4:  # 包含危險標記的新格式
                        action, prob, value, is_dangerous = result
                    else:  # 傳統格式
                        action, prob, value = result
                        is_dangerous = False

                    # 只為玩家1儲存轉換
                    if training and current_player == 1:
                        episode_transitions.append({
                            'state': state,
                            'action': action,
                            'prob': prob,
                            'value': value,
                            'is_dangerous': is_dangerous
                        })

                    actions.append(action)
                else:
                    # 非活躍玩家 - 使用虛擬動作
                    actions.append(0)

            # 執行動作
            try:
                env.step(actions)
                move_count += 1
            except Exception as e:
                logger.error(f"執行動作時出錯: {e}")
                break

        # 計算獎勵
        if training and episode_transitions:
            reward = 0
            game_length = len(episode_transitions)

            try:
                if env.state[0]['status'] == 'DONE':
                    if env.state[0]['reward'] == 1:  # 玩家1獲勝
                        # 快速獲勝給予更高獎勵，長遊戲獲勝獎勵較低
                        if game_length <= 7:
                            reward = 2  # 快速獲勝
                        elif game_length <= 15:
                            reward = 15  # 正常獲勝
                        else:
                            reward = 20  # 長遊戲獲勝
                    elif env.state[0]['reward'] == -1:  # 玩家1失敗
                        # 快速失敗懲罰更重
                        if game_length <= 7:
                            reward = -150  # 快速失敗
                        elif game_length <= 15:
                            reward = -100  # 正常失敗
                        else:
                            reward = -80   # 長遊戲失敗（至少撐得久）
                    else:  # 平局
                        # 平局根據遊戲長度給不同獎勵
                        if game_length >= 20:
                            reward = 5   # 長遊戲平局是好的
                        else:
                            reward = -10  # 短遊戲平局可能是策略問題
            except KeyError:
                reward = -5  # 異常情況給小懲罰

            # 分配獎勵給轉換（加入位置獎勵和危險動作懲罰）
            for i, transition in enumerate(episode_transitions):
                # 基礎獎勵
                shaped_reward = reward

                # 危險動作懲罰
                if transition.get('is_dangerous', False):
                    danger_penalty = -0.8  # 危險動作懲罰
                    shaped_reward += danger_penalty
                    logger.debug(f"對危險動作給予懲罰: {danger_penalty}")

                # 時序獎勵：越靠近遊戲結尾的動作影響越大
                position_weight = (i + 1) / game_length
                shaped_reward = shaped_reward * (0.5 + 0.5 * position_weight)

                # 添加小的探索獎勵（鼓勵嘗試不同策略）
                exploration_bonus = 10 if i % 3 == 0 else 0
                shaped_reward += exploration_bonus

                self.agent.store_transition(
                    transition['state'],
                    transition['action'],
                    transition['prob'],
                    shaped_reward,
                    i == len(episode_transitions) - 1  # done 標誌
                )

        try:
            final_reward = env.state[0]['reward']
        except (KeyError, IndexError):
            final_reward = 0

        return final_reward, len(episode_transitions)

    def self_play_episode(self):
        """自對弈回合（帶隨機性增強多樣性）"""
        # 隨機選擇自對弈策略以增加多樣性
        strategy = np.random.choice(['standard', 'noisy', 'exploration', 'temperature'], p=[0.4, 0.2, 0.2, 0.2])

        def agent_func(state, valid_actions, training):
            # 獲取原始動作
            action, prob, value = self.agent.select_action(state, valid_actions, training)

            if training:
                # 根據策略添加不同類型的隨機性
                if strategy == 'noisy':
                    # 噪音策略：有一定概率選擇隨機動作
                    if np.random.random() < 0.15:  # 15%概率選擇隨機動作
                        action = int(np.random.choice(valid_actions))  # 確保返回 Python int
                        logger.debug(f"自對弈使用噪音策略，隨機選擇動作: {action}")

                elif strategy == 'exploration':
                    # 探索策略：基於動作概率的探索性採樣
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                    self.agent.policy_net.eval()  # 避免 BatchNorm 問題
                    with torch.no_grad():
                        action_probs, _ = self.agent.policy_net(state_tensor)

                    # 遮罩無效動作
                    action_probs = action_probs.cpu().numpy()[0]
                    masked_probs = np.zeros_like(action_probs)
                    masked_probs[valid_actions] = action_probs[valid_actions]

                    if masked_probs.sum() > 0:
                        masked_probs /= masked_probs.sum()
                        # 使用溫度採樣增加探索
                        temperature = 1.5
                        temp_probs = np.power(masked_probs, 1/temperature)
                        temp_probs[valid_actions] /= temp_probs[valid_actions].sum()
                        action = int(np.random.choice(7, p=temp_probs))  # 確保返回 Python int
                        logger.debug(f"自對弈使用探索策略，溫度採樣動作: {action}")

                elif strategy == 'temperature':
                    # 溫度策略：隨機調整決策溫度
                    temperature = np.random.uniform(0.8, 2.0)
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                    self.agent.policy_net.eval()  # 避免 BatchNorm 問題
                    with torch.no_grad():
                        action_probs, _ = self.agent.policy_net(state_tensor)

                    action_probs = action_probs.cpu().numpy()[0]
                    masked_probs = np.zeros_like(action_probs)
                    masked_probs[valid_actions] = action_probs[valid_actions]

                    if masked_probs.sum() > 0:
                        # 應用溫度
                        temp_probs = np.power(masked_probs, 1/temperature)
                        temp_probs /= temp_probs.sum()
                        action = int(valid_actions[np.argmax(temp_probs[valid_actions])])  # 確保返回 Python int
                        logger.debug(f"自對弈使用溫度策略 (T={temperature:.2f})，動作: {action}")

                # 檢查危險動作
                board = state[:42].reshape(6, 7).astype(int)
                mark = 1  # 在自對弈中，當前玩家總是1

                if self.is_dangerous_move(board, mark, action, look_ahead_steps=3):
                    logger.debug(f"自對弈中agent選擇了危險動作 {action} (策略: {strategy})")
                    return int(action), prob, value, True  # 確保返回 Python int

            return int(action), prob, value, False  # 確保返回 Python int

        reward, episode_length = self.play_game(agent_func, agent_func, training=True)
        logger.debug(f"自對弈完成，策略: {strategy}, 回合長度: {episode_length}")
        return reward, episode_length

    def adaptive_self_play_episode(self, episode_num):
        """自適應自對弈回合（根據訓練進度調整多樣性）"""
        # 根據訓練進度動態調整參數
        progress = min(episode_num / 5000, 1.0)  # 5000回合達到穩定

        # 多樣性策略選擇概率（隨訓練進度變化）
        if progress < 0.3:  # 早期階段：高探索
            strategy_probs = [0.2, 0.4, 0.3, 0.1]  # [standard, noisy, exploration, temperature]
        elif progress < 0.7:  # 中期階段：平衡探索
            strategy_probs = [0.4, 0.3, 0.2, 0.1]
        else:  # 後期階段：更多標準玩法
            strategy_probs = [0.6, 0.2, 0.15, 0.05]

        strategy = np.random.choice(['standard', 'noisy', 'exploration', 'temperature'], p=strategy_probs)

        def adaptive_agent_func(state, valid_actions, training):
            action, prob, value = self.agent.select_action(state, valid_actions, training)

            if training:
                if strategy == 'noisy':
                    noise_level = 0.2 * (1 - progress) + 0.05 * progress  # 隨進度降低噪音
                    if np.random.random() < noise_level:
                        action = int(np.random.choice(valid_actions))  # 確保返回 Python int

                elif strategy == 'exploration':
                    temperature = 1.8 - 0.6 * progress  # 溫度隨進度降低
                    exploration_bonus = 0.15 * (1 - progress)  # 探索獎勵隨進度降低
                    action, prob, value = self.agent.select_action(
                        state, valid_actions, training, temperature=temperature, exploration_bonus=exploration_bonus)

                elif strategy == 'temperature':
                    # 溫度範圍隨進度收窄
                    temp_range = (0.8 + 0.4 * progress, 2.0 - 0.5 * progress)
                    temperature = np.random.uniform(*temp_range)
                    action, prob, value = self.agent.select_action(
                        state, valid_actions, training, temperature=temperature)

                # 檢查危險動作
                board = state[:42].reshape(6, 7).astype(int)
                mark = 1

                if self.is_dangerous_move(board, mark, action, look_ahead_steps=3):
                    return int(action), prob, value, True  # 確保返回 Python int

            return int(action), prob, value, False  # 確保返回 Python int

        reward, episode_length = self.play_game(adaptive_agent_func, adaptive_agent_func, training=True)
        return reward, episode_length

    def diverse_training_episode(self, episode_num):
        """多樣性訓練回合 - 對抗不同強度的對手"""
        # 根據訓練進度選擇對手
        if episode_num < 2000:
            # 早期：主要自對弈（高隨機性）
            if episode_num % 30 == 0:
                # 每10回合對抗隨機對手（保持對弱對手的統治力）
                return self.play_against_random_agent()
            else:
                return self.play_against_minimax_agent()
            # 每8回合對抗隨機對手（保持對弱對手的統治力）
        else:
            # 其他時候標準自對弈
            if episode_num % 100 == 0:
                return self.play_against_minimax_agent()
            if episode_num % 500 == 0:
                return self.play_against_random_agent()
            if episode_num % 3 == 0:
                return self.self_play_episode()
            if episode_num % 3 == 1:
                return self.adaptive_self_play_episode(episode_num)
            if episode_num % 3 == 2:
                return self.diverse_self_play_episode(episode_num)

    def diverse_self_play_episode(self, episode_num):
        """高多樣性自對弈回合"""
        # 根據訓練進度調整隨機性強度
        progress = min(episode_num / 10000, 1.0)  # 10000回合後達到最低隨機性
        base_randomness = 0.3 * (1 - progress) + 0.1 * progress  # 從30%降到10%

        def diverse_agent_func(state, valid_actions, training):
            # 獲取原始動作
            action, prob, value = self.agent.select_action(state, valid_actions, training)

            if training:
                # 動態調整的多樣性策略
                randomness_level = base_randomness + np.random.uniform(-0.05, 0.05)

                if np.random.random() < randomness_level:
                    # 多種隨機化方式
                    rand_type = np.random.choice(['pure_random', 'weighted_random', 'anti_pattern'])

                    if rand_type == 'pure_random':
                        # 純隨機選擇
                        action = int(np.random.choice(valid_actions))  # 確保返回 Python int
                        logger.debug(f"多樣性自對弈：純隨機動作 {action}")

                    elif rand_type == 'weighted_random':
                        # 帶權重的隨機選擇（偏好中央列）
                        weights = np.array([1, 2, 3, 4, 3, 2, 1])  # 中央權重更高
                        valid_weights = weights[valid_actions]
                        valid_weights = valid_weights / valid_weights.sum()
                        action = int(np.random.choice(valid_actions, p=valid_weights))  # 確保返回 Python int
                        logger.debug(f"多樣性自對弈：加權隨機動作 {action}")

                    elif rand_type == 'anti_pattern':
                        # 反模式選擇：選擇模型認為不太好的動作
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                        self.agent.policy_net.eval()  # 避免 BatchNorm 問題
                        with torch.no_grad():
                            action_probs, _ = self.agent.policy_net(state_tensor)

                        action_probs = action_probs.cpu().numpy()[0]
                        masked_probs = np.zeros_like(action_probs)
                        masked_probs[valid_actions] = action_probs[valid_actions]

                        if masked_probs.sum() > 0:
                            # 反轉概率（選擇模型不喜歡的動作）
                            inv_probs = 1.0 - masked_probs
                            inv_probs[valid_actions] /= inv_probs[valid_actions].sum()
                            action = int(np.random.choice(valid_actions, p=inv_probs[valid_actions]))  # 確保返回 Python int
                            logger.debug(f"多樣性自對弈：反模式動作 {action}")

                # 檢查危險動作
                board = state[:42].reshape(6, 7).astype(int)
                mark = 1

                if self.is_dangerous_move(board, mark, action, look_ahead_steps=3):
                    logger.debug(f"多樣性自對弈中選擇了危險動作 {action}")
                    return int(action), prob, value, True  # 確保返回 Python int

            return int(action), prob, value, False  # 確保返回 Python int

        reward, episode_length = self.play_game(diverse_agent_func, diverse_agent_func, training=True)
        logger.debug(f"高多樣性自對弈完成，隨機性: {base_randomness:.2f}, 回合長度: {episode_length}")
        return reward, episode_length

    def check_win_move(self, board, mark, col):
        """檢查在指定列放置棋子後是否能獲勝"""
        # 模擬放置棋子
        test_board = board.copy()
        row = -1
        for r in range(5, -1, -1):  # 從下往上找空位
            if test_board[r][col] == 0:
                test_board[r][col] = mark
                row = r
                break

        if row == -1:  # 該列已滿
            return False

        # 檢查四個方向是否連成四子
        directions = [
            (0, 1),   # 水平
            (1, 0),   # 垂直
            (1, 1),   # 主對角線
            (1, -1)   # 反對角線
        ]

        for dr, dc in directions:
            count = 1

            # 正方向檢查
            r, c = row + dr, col + dc
            while 0 <= r < 6 and 0 <= c < 7 and test_board[r][c] == mark:
                count += 1
                r, c = r + dr, c + dc

            # 反方向檢查
            r, c = row - dr, col - dc
            while 0 <= r < 6 and 0 <= c < 7 and test_board[r][c] == mark:
                count += 1
                r, c = r - dr, c - dc

            if count >= 4:
                return True

        return False

    def if_i_can_finish(self, board, mark, valid_actions):
        """檢查是否有直接獲勝的動作"""
        for col in valid_actions:
            if self.check_win_move(board, mark, col):
                return col
        return -1

    def if_i_will_lose(self, board, mark, valid_actions):
        """檢查對手是否能在下一步獲勝，如果是則返回阻擋的動作"""
        opponent_mark = 3 - mark  # 對手標記
        for col in valid_actions:
            if self.check_win_move(board, opponent_mark, col):
                return col  # 返回需要阻擋的列
        return -1

    def is_dangerous_move(self, board, mark, action, look_ahead_steps=2):
        """檢查動作是否危險（會讓對手在指定步數內獲勝）"""
        # 模擬放置我們的棋子
        test_board = board.copy()
        row = -1
        for r in range(5, -1, -1):
            if test_board[r][action] == 0:
                test_board[r][action] = mark
                row = r
                break

        if row == -1:  # 該列已滿
            return True  # 無效動作視為危險

        opponent_mark = 3 - mark
        return self._check_opponent_can_win_in_steps(test_board, opponent_mark, look_ahead_steps)

    def _check_opponent_can_win_in_steps(self, board, opponent_mark, steps_remaining):
        """遞歸檢查對手是否能在指定步數內獲勝"""
        if steps_remaining <= 0:
            return False

        # 檢查對手是否能立即獲勝
        valid_actions = [c for c in range(7) if board[0][c] == 0]
        for action in valid_actions:
            if self.check_win_move(board, opponent_mark, action):
                return True

        # 如果只剩1步，對手無法立即獲勝則安全
        if steps_remaining == 1:
            return False

        # 模擬對手的每個可能動作
        for action in valid_actions:
            test_board = board.copy()
            row = -1
            for r in range(5, -1, -1):
                if test_board[r][action] == 0:
                    test_board[r][action] = opponent_mark
                    row = r
                    break

            if row != -1:
                # 對手放置後，檢查我們是否能阻止他們在剩餘步數內獲勝
                our_mark = 3 - opponent_mark
                can_prevent = False

                our_valid_actions = [c for c in range(7) if test_board[0][c] == 0]
                for our_action in our_valid_actions:
                    our_test_board = test_board.copy()
                    our_row = -1
                    for r in range(5, -1, -1):
                        if our_test_board[r][our_action] == 0:
                            our_test_board[r][our_action] = our_mark
                            our_row = r
                            break

                    if our_row != -1:
                        # 檢查對手在剩餘步數內是否仍能獲勝
                        if not self._check_opponent_can_win_in_steps(our_test_board, opponent_mark, steps_remaining - 2):
                            can_prevent = True
                            break

                # 如果我們無法阻止對手獲勝，則當前動作危險
                if not can_prevent:
                    return True

        return False

    def filter_safe_actions(self, board, mark, valid_actions, look_ahead_steps=3):
        """過濾出安全的動作（不會讓對手快速獲勝）"""
        safe_actions = []
        dangerous_actions = []

        for action in valid_actions:
            if not self.is_dangerous_move(board, mark, action, look_ahead_steps):
                safe_actions.append(action)
            else:
                dangerous_actions.append(action)

        # 如果所有動作都危險，返回原始動作（避免無動作可選）
        if not safe_actions:
            return valid_actions, dangerous_actions

        return safe_actions, dangerous_actions

    def play_against_random_agent(self):
        """對抗隨機對手的訓練回合"""
        def random_agent(state, valid_actions, training):
            # 將狀態轉換為 6x7 棋盤
            board = state[:42].reshape(6, 7).astype(int)
            mark = 2  # 隨機agent通常是玩家2

            # 首先檢查是否可以直接獲勝
            winning_move = self.if_i_can_finish(board, mark, valid_actions)
            if winning_move != -1:
                return winning_move, 1.0, 0.0

            # 其次檢查是否需要阻擋對手獲勝
            blocking_move = self.if_i_will_lose(board, mark, valid_actions)
            if blocking_move != -1:
                return blocking_move, 1.0, 0.0

            # 過濾危險動作
            safe_actions, dangerous_actions = self.filter_safe_actions(board, mark, valid_actions)

            # 優先選擇安全動作
            if safe_actions:
                action = random.choice(safe_actions)
            else:
                # 如果沒有安全動作，從所有動作中選擇（但記錄警告）
                action = random.choice(valid_actions)
                if training:
                    logger.debug(f"隨機agent被迫選擇危險動作: {action}")

            return action, 1.0/len(valid_actions), 0.0

        def trained_agent(state, valid_actions, training):
            # 獲取原始動作
            action, prob, value = self.agent.select_action(state, valid_actions, training)

            if training:
                # 在訓練模式下檢測危險動作並給予懲罰
                board = state[:42].reshape(6, 7).astype(int)
                mark = 1  # 訓練的agent通常是玩家1

                # 檢查選擇的動作是否危險
                if self.is_dangerous_move(board, mark, action, look_ahead_steps=3):
                    # 危險動作懲罰：給予負獎勵
                    danger_penalty = -0.5
                    # 注意：這裡不能直接store_transition，因為這會在正常的遊戲流程之外
                    # 我們將在play_game函數中處理這個懲罰
                    logger.debug(f"訓練agent選擇了危險動作 {action}")
                    # 返回動作時標記危險
                    return action, prob, value, True  # 額外返回危險標記

            return action, prob, value, False  # 返回安全標記

        reward, episode_length = self.play_game(trained_agent, random_agent, training=True)
        return reward, episode_length

    def play_against_minimax_agent(self):
        """對抗minimax對手的訓練回合"""
        def minimax_agent(state, valid_actions, training):
            # 將狀態轉換為 6x7 棋盤
            board = state[:42].reshape(6, 7).astype(int)
            mark = 2  # minimax agent通常是玩家2

            # 首先檢查是否可以直接獲勝
            winning_move = self.if_i_can_finish(board, mark, valid_actions)
            if winning_move != -1:
                return winning_move, 1.0, 0.0

            # 其次檢查是否需要阻擋對手獲勝
            blocking_move = self.if_i_will_lose(board, mark, valid_actions)
            if blocking_move != -1:
                return blocking_move, 1.0, 0.0

            # 過濾危險動作
            safe_actions, dangerous_actions = self.filter_safe_actions(board, mark, valid_actions)

            # 如果有安全動作，在安全動作中使用minimax
            if safe_actions:
                best_action = self._minimax_move(board, safe_actions, depth=2)
            else:
                # 如果沒有安全動作，仍使用minimax但記錄警告
                best_action = self._minimax_move(board, valid_actions, depth=2)
                if training:
                    logger.debug(f"Minimax agent被迫選擇危險動作: {best_action}")

            return best_action, 1.0, 0.0

        def trained_agent(state, valid_actions, training):
            # 獲取原始動作
            action, prob, value = self.agent.select_action(state, valid_actions, training)

            if training:
                # 在訓練模式下檢測危險動作並給予懲罰
                board = state[:42].reshape(6, 7).astype(int)
                mark = 1  # 訓練的agent通常是玩家1

                # 檢查選擇的動作是否危險
                if self.is_dangerous_move(board, mark, action, look_ahead_steps=3):
                    logger.debug(f"訓練agent選擇了危險動作 {action}")
                    return action, prob, value, True  # 額外返回危險標記

            return action, prob, value, False  # 返回安全標記

        reward, episode_length = self.play_game(trained_agent, minimax_agent, training=True)
        return reward, episode_length

    def evaluate_model(self, model=None, episode=0):
        """通用模型評估方法"""
        eval_mode = self.config.get('evaluation', {}).get('mode', 'random')
        num_games = self.config.get('evaluation', {}).get('num_games', 50)

        logger.info(f"使用 {eval_mode} 模式評估模型...")

        if eval_mode == 'comprehensive':
            # 綜合評估
            eval_results = self.evaluate_comprehensive(num_games)
            return eval_results['comprehensive_score']

        elif eval_mode == 'detailed':
            # 詳細指標評估
            metrics = self.evaluate_with_metrics(num_games)
            return metrics['win_rate']

        elif eval_mode == 'minimax':
            # 只對Minimax評估
            return self.evaluate_against_minimax(num_games // 2)

        else:
            # 默認：對隨機對手評估
            return self.evaluate_against_random(num_games)

    def evaluate_against_random(self, num_games=50):
        """對隨機對手評估"""
        def random_agent(state, valid_actions, training=False):
            # 將狀態轉換為 6x7 棋盤
            board = state[:42].reshape(6, 7).astype(int)
            mark = 2  # 隨機agent通常是玩家2

            # 首先檢查是否可以直接獲勝
            winning_move = self.if_i_can_finish(board, mark, valid_actions)
            if winning_move != -1:
                return winning_move, 1.0, 0.0

            # 其次檢查是否需要阻擋對手獲勝
            blocking_move = self.if_i_will_lose(board, mark, valid_actions)
            if blocking_move != -1:
                return blocking_move, 1.0, 0.0

            # 過濾危險動作
            safe_actions, dangerous_actions = self.filter_safe_actions(board, mark, valid_actions)

            # 優先選擇安全動作
            if safe_actions:
                action = random.choice(safe_actions)
            else:
                action = random.choice(valid_actions)

            return action, 1.0/len(valid_actions), 0.0

        def trained_agent(state, valid_actions, training=False):
            return self.agent.select_action(state, valid_actions, training=False)

        wins = 0
        for _ in range(num_games):
            reward, _ = self.play_game(trained_agent, random_agent, training=False)
            if reward == 1:
                wins += 1

        return wins / num_games

    def evaluate_comprehensive(self, num_games=50):
        """綜合評估函數 - 對多種對手測試"""
        results = {}

        # 1. 對隨機對手評估
        results['vs_random'] = self.evaluate_against_random(num_games)

        # 2. 對minimax對手評估
        results['vs_minimax'] = self.evaluate_against_minimax(num_games // 2)

        # 3. 對自己評估（自對弈）
        results['self_play'] = self.evaluate_self_play(num_games // 2)

        # 4. 計算綜合分數
        comprehensive_score = (
            results['vs_random'] * 0.1 +
            results['vs_minimax'] * 0.1 +
            results['self_play'] * 0.8
        )
        results['comprehensive_score'] = comprehensive_score

        return results

    def evaluate_against_minimax(self, num_games=25):
        """對Minimax智能體評估"""
        def minimax_agent(state, valid_actions, training=False):
            # 將狀態轉換為 6x7 棋盤
            board = state[:42].reshape(6, 7).astype(int)
            mark = 2  # minimax agent通常是玩家2

            # 首先檢查是否可以直接獲勝
            winning_move = self.if_i_can_finish(board, mark, valid_actions)
            if winning_move != -1:
                return winning_move, 1.0, 0.0

            # 其次檢查是否需要阻擋對手獲勝
            blocking_move = self.if_i_will_lose(board, mark, valid_actions)
            if blocking_move != -1:
                return blocking_move, 1.0, 0.0

            # 過濾危險動作
            safe_actions, dangerous_actions = self.filter_safe_actions(board, mark, valid_actions)

            # 如果有安全動作，在安全動作中使用minimax
            if safe_actions:
                best_action = self._minimax_move(board, safe_actions)
            else:
                best_action = self._minimax_move(board, valid_actions)

            return best_action, 1.0, 0.0

        def trained_agent(state, valid_actions, training=False):
            return self.agent.select_action(state, valid_actions, training=False)

        wins = 0
        for _ in range(num_games):
            reward, _ = self.play_game(trained_agent, minimax_agent, training=False)
            if reward == 1:
                wins += 1

        return wins / num_games

    def evaluate_self_play(self, num_games=25):
        """自對弈評估 - 對抗不同策略的智能對手"""

        def trained_agent(state, valid_actions, training=False):
            """訓練好的標準 agent"""
            return self.agent.select_action(state, valid_actions, training=False)

        def diverse_tactical_agent(state, valid_actions, training=False):
            """多樣化戰術 agent - 結合戰術判斷和策略多樣性"""
            # 將狀態轉換為棋盤
            board = state[:42].reshape(6, 7).astype(int)
            mark = 1  # 在評估中，這個 agent 總是作為玩家1

            # 1. 首先檢查是否能直接獲勝
            winning_move = self.if_i_can_finish(board, mark, valid_actions)
            if winning_move != -1:
                logger.debug(f"多樣化agent找到獲勝動作: {winning_move}")
                return int(winning_move), 1.0, 1.0

            # 2. 檢查是否需要阻擋對手獲勝
            block_move = self.if_i_will_lose(board, mark, valid_actions)
            if block_move != -1:
                logger.debug(f"多樣化agent阻擋對手獲勝: {block_move}")
                return int(block_move), 1.0, 0.8

            # 3. 過濾掉危險動作
            safe_actions, dangerous_actions = self.filter_safe_actions(board, mark, valid_actions, look_ahead_steps=2)
            if not safe_actions:
                safe_actions = valid_actions  # 如果沒有安全動作，使用所有有效動作
                logger.debug(f"多樣化agent: 所有動作都危險，使用全部有效動作")
            elif len(dangerous_actions) > 0:
                logger.debug(f"多樣化agent: 過濾掉危險動作 {dangerous_actions}")

            # 4. 使用多樣化策略選擇動作（更激進的策略分布）
            strategy = np.random.choice(['greedy', 'exploration', 'positional', 'aggressive'],
                                     p=[0.3, 0.3, 0.2, 0.2])

            if strategy == 'greedy':
                # 貪婪策略：直接選擇模型認為最好的動作
                action, prob, value = self.agent.select_action(state, safe_actions, training=False)

            elif strategy == 'exploration':
                # 探索策略：帶溫度的採樣
                action, prob, value = self.agent.select_action(
                    state, safe_actions, training=True, temperature=1.5, exploration_bonus=0.1)

            elif strategy == 'positional':
                # 位置策略：偏好中央位置和戰略位置
                weights = np.array([1, 3, 5, 6, 5, 3, 1])  # 中央權重更高
                valid_weights = weights[safe_actions]
                valid_weights = valid_weights / valid_weights.sum()
                action = int(np.random.choice(safe_actions, p=valid_weights))

                # 獲取對應的概率和價值
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                self.agent.policy_net.eval()
                with torch.no_grad():
                    action_probs, state_value = self.agent.policy_net(state_tensor)
                prob = action_probs.cpu().numpy()[0][action]
                value = state_value.item()

            elif strategy == 'aggressive':
                # 激進策略：偏好攻擊性動作，如果安全的話偶爾冒險
                action, prob, value = self.agent.select_action(
                    state, safe_actions, training=True, temperature=2.0)

                # 10% 機率選擇危險動作進行激進攻擊
                if dangerous_actions and np.random.random() < 0.1:
                    action = int(np.random.choice(dangerous_actions))
                    logger.debug(f"多樣化agent選擇激進動作: {action}")

            logger.debug(f"多樣化agent使用{strategy}策略，選擇動作: {action}")
            return int(action), prob, value

        # 進行評估遊戲
        wins = 0
        losses = 0
        draws = 0
        tactical_wins = 0  # 戰術獲勝（使用了戰術函數）

        for game in range(num_games):
            # 隨機決定誰先手
            if game % 2 == 0:
                reward, episode_length = self.play_game(trained_agent, diverse_tactical_agent, training=False)
                player1_is_trained = True
            else:
                reward, episode_length = self.play_game(diverse_tactical_agent, trained_agent, training=False)
                player1_is_trained = False

            # 根據先手情況調整結果統計
            if player1_is_trained:
                if reward == 1:
                    wins += 1
                elif reward == -1:
                    losses += 1
                else:
                    draws += 1
            else:
                if reward == 1:
                    losses += 1
                elif reward == -1:
                    wins += 1
                else:
                    draws += 1

        # 計算各種指標
        win_rate = wins / num_games
        loss_rate = losses / num_games
        draw_rate = draws / num_games

        logger.info(f"多樣化自對弈評估結果: 勝利={wins}, 失敗={losses}, 平局={draws}")
        logger.info(f"對抗多樣化戰術對手: 勝率={win_rate:.3f}, 敗率={loss_rate:.3f}, 平局率={draw_rate:.3f}")

        # 評分邏輯：
        # 1. 基礎勝率分數（勝率越高越好）
        base_score = win_rate

        # 2. 平局獎勵（適量平局說明策略穩健，但不應該主導評分）
        if 0.1 <= draw_rate <= 0.4:
            draw_bonus = 0.2  # 給予平局獎勵
        elif draw_rate > 0.6:
            draw_bonus = 0.1  # 平局太多稍微減分
        else:
            draw_bonus = 0.05  # 平局很少給少量獎勵

        # 3. 不敗獎勵（勝利+平局的比例）
        undefeated_rate = win_rate + draw_rate
        if undefeated_rate >= 0.8:
            undefeated_bonus = 0.1
        else:
            undefeated_bonus = 0.0

        # 4. 避免完全碾壓的情況（保持一定挑戰性）
        if win_rate > 0.9:
            dominance_penalty = -0.05  # 稍微減分以保持挑戰
        elif win_rate < 0.2 and draw_rate < 0.5:
            dominance_penalty = -0.1  # 表現太差減分
        else:
            dominance_penalty = 0.0

        # 最終評分計算
        final_score = base_score * 0.6 + draw_bonus + undefeated_bonus + dominance_penalty
        final_score = max(0.0, min(1.0, final_score))  # 限制在[0,1]範圍內

        logger.info(f"評分詳情: 基礎分={base_score:.3f}, 平局獎勵={draw_bonus:.3f}, "
                   f"不敗獎勵={undefeated_bonus:.3f}, 統治懲罰={dominance_penalty:.3f}")
        logger.info(f"多樣化自對弈最終評分: {final_score:.3f}")
        return final_score

    def evaluate_with_metrics(self, num_games=50):
        """詳細指標評估"""
        def random_agent(state, valid_actions, training=False):
            # 將狀態轉換為 6x7 棋盤
            board = state[:42].reshape(6, 7).astype(int)
            mark = 2  # 隨機agent通常是玩家2

            # 首先檢查是否可以直接獲勝
            winning_move = self.if_i_can_finish(board, mark, valid_actions)
            if winning_move != -1:
                return winning_move, 1.0, 0.0

            # 其次檢查是否需要阻擋對手獲勝
            blocking_move = self.if_i_will_lose(board, mark, valid_actions)
            if blocking_move != -1:
                return blocking_move, 1.0, 0.0

            # 過濾危險動作
            safe_actions, dangerous_actions = self.filter_safe_actions(board, mark, valid_actions)

            # 優先選擇安全動作
            if safe_actions:
                action = random.choice(safe_actions)
            else:
                action = random.choice(valid_actions)

            return action, 1.0/len(valid_actions), 0.0

        def trained_agent(state, valid_actions, training=False):
            return self.agent.select_action(state, valid_actions, training=False)

        metrics = {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'total_moves': 0,
            'avg_game_length': 0,
            'quick_wins': 0,  # 10步內獲勝
            'comeback_wins': 0,  # 長遊戲獲勝
        }

        game_lengths = []

        for game_idx in range(num_games):
            reward, episode_length = self.play_game(trained_agent, random_agent, training=False)
            game_lengths.append(episode_length)

            if reward == 1:
                metrics['wins'] += 1
                if episode_length <= 10:
                    metrics['quick_wins'] += 1
                elif episode_length >= 20:
                    metrics['comeback_wins'] += 1
            elif reward == -1:
                metrics['losses'] += 1
            else:
                metrics['draws'] += 1

            metrics['total_moves'] += episode_length

        metrics['win_rate'] = metrics['wins'] / num_games
        metrics['avg_game_length'] = np.mean(game_lengths)
        metrics['game_length_std'] = np.std(game_lengths)

        return metrics

    def _minimax_move(self, board, valid_actions, depth=4):
        """簡單的Minimax實現"""
        best_action = valid_actions[0]
        best_score = float('-inf')

        for action in valid_actions:
            # 模擬放置棋子
            test_board = board.copy()
            row = self._get_drop_row(test_board, action)
            if row is not None:
                test_board[row][action] = 1  # AI是玩家1
                score = self._minimax(test_board, depth-1, False, action)
                if score > best_score:
                    best_score = score
                    best_action = action

        return best_action

    def _minimax(self, board, depth, is_maximizing, last_action):
        """Minimax算法核心"""
        # 檢查遊戲結束
        if self._check_win(board, last_action):
            return 1000 if not is_maximizing else -1000

        if depth == 0 or self._is_board_full(board):
            return self._evaluate_board(board)

        valid_actions = [c for c in range(7) if board[0][c] == 0]

        if is_maximizing:
            max_score = float('-inf')
            for action in valid_actions:
                test_board = board.copy()
                row = self._get_drop_row(test_board, action)
                if row is not None:
                    test_board[row][action] = 1
                    score = self._minimax(test_board, depth-1, False, action)
                    max_score = max(max_score, score)
            return max_score
        else:
            min_score = float('inf')
            for action in valid_actions:
                test_board = board.copy()
                row = self._get_drop_row(test_board, action)
                if row is not None:
                    test_board[row][action] = 2
                    score = self._minimax(test_board, depth-1, True, action)
                    min_score = min(min_score, score)
            return min_score

    def _get_drop_row(self, board, col):
        """獲取棋子掉落的行"""
        for row in range(5, -1, -1):
            if board[row][col] == 0:
                return row
        return None

    def _check_win(self, board, last_col):
        """檢查是否獲勝"""
        if last_col is None:
            return False

        last_row = self._get_drop_row(board, last_col)
        if last_row is None:
            last_row = 0
        else:
            last_row += 1

        if last_row >= 6:
            return False

        player = board[last_row][last_col]
        if player == 0:
            return False

        # 檢查四個方向
        directions = [(0,1), (1,0), (1,1), (1,-1)]

        for dr, dc in directions:
            count = 1
            # 正方向
            r, c = last_row + dr, last_col + dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r][c] == player:
                count += 1
                r, c = r + dr, c + dc
            # 負方向
            r, c = last_row - dr, last_col - dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r][c] == player:
                count += 1
                r, c = r - dr, c - dc

            if count >= 4:
                return True

        return False

    def _is_board_full(self, board):
        """檢查棋盤是否已滿"""
        return all(board[0][c] != 0 for c in range(7))

    def _evaluate_board(self, board):
        """評估棋盤局勢"""
        score = 0

        # 簡單的位置評分
        center_col = 3
        for row in range(6):
            for col in range(7):
                if board[row][col] == 1:  # AI棋子
                    score += abs(col - center_col) * -1 + 3
                elif board[row][col] == 2:  # 對手棋子
                    score -= abs(col - center_col) * -1 + 3

        return score

    def train(self):
        """主要訓練循環"""
        logger.info("開始 ConnectX 強化學習訓練")
        logger.info(f"設備: {self.agent.device}")

        # 初始化最佳勝率（如果有載入的歷史，使用歷史最佳值）
        best_win_rate = max(self.win_rates) if self.win_rates else 0
        if best_win_rate > 0:
            logger.info(f"從載入模型繼續訓練，當前最佳勝率: {best_win_rate:.3f}")

        episodes_since_improvement = 0
        start_episode = len(self.episode_rewards)  # 從已完成的回合數開始

        if start_episode > 0:
            logger.info(f"從第 {start_episode} 回合繼續訓練")

        logger.info(f"最大訓練回合: {self.config['training']['max_episodes']}")
        for episode in range(start_episode, self.config['training']['max_episodes']):
            try:
                # 多樣性訓練回合（根據配置選擇）
                if self.config.get('training', {}).get('opponent_diversity', False):
                    reward, episode_length = self.diverse_training_episode(episode)
                else:
                    reward, episode_length = self.self_play_episode()

                self.episode_rewards.append(reward)

                # 更新策略
                if len(self.agent.memory) >= self.config['agent']['min_batch_size']:
                    loss_info = self.agent.update_policy()
                    if loss_info:
                        self.training_losses.append(loss_info)

                # 持續監督學習（每隔一定間隔）
                continuous_learning_frequency = self.config.get('training', {}).get('continuous_learning_frequency', 10)
                if (self.continuous_learning_data is not None and
                    episode % continuous_learning_frequency == 0 and
                    episode > 0):

                    supervised_loss = self.continuous_supervised_learning(
                        self.continuous_learning_data,
                        self.continuous_learning_targets,
                        batch_size=self.config.get('training', {}).get('continuous_batch_size', 64),
                        learning_rate=self.config.get('training', {}).get('continuous_learning_rate', 1e-5)
                    )

                    if supervised_loss and episode % (continuous_learning_frequency * 10) == 0:
                        logger.info(f"持續監督學習 - 策略損失: {supervised_loss['supervised_policy_loss']:.4f}, "
                                  f"價值損失: {supervised_loss['supervised_value_loss']:.4f}")

                # 評估
                if episode % self.config['training']['eval_frequency'] == 0:
                    # 根據配置選擇評估方式
                    eval_mode = self.config.get('evaluation', {}).get('mode', 'random')

                    if eval_mode == 'comprehensive':
                        # 綜合評估
                        eval_results = self.evaluate_comprehensive(
                            self.config['training']['eval_games']
                        )
                        win_rate = eval_results['comprehensive_score']

                        logger.info(
                            f"回合 {episode}: 綜合評估分數: {win_rate:.3f}\n"
                            f"  vs 隨機: {eval_results['vs_random']:.3f}\n"
                            f"  vs Minimax: {eval_results['vs_minimax']:.3f}\n"
                            f"  自對弈: {eval_results['self_play']:.3f}"
                        )

                    elif eval_mode == 'detailed':
                        # 詳細指標評估
                        metrics = self.evaluate_with_metrics(
                            self.config['training']['eval_games']
                        )
                        win_rate = metrics['win_rate']

                        logger.info(
                            f"回合 {episode}: 詳細評估\n"
                            f"  勝率: {metrics['win_rate']:.3f}\n"
                            f"  平均步數: {metrics['avg_game_length']:.1f}\n"
                            f"  快速獲勝: {metrics['quick_wins']}/{metrics['wins']}\n"
                            f"  長遊戲獲勝: {metrics['comeback_wins']}/{metrics['wins']}"
                        )

                    elif eval_mode == 'minimax':
                        # 只對Minimax評估
                        win_rate = self.evaluate_against_minimax(
                            self.config['training']['eval_games'] // 2
                        )

                        logger.info(
                            f"回合 {episode}: vs Minimax 勝率: {win_rate:.3f}"
                        )

                    else:
                        # 默認：對隨機對手評估
                        win_rate = self.evaluate_against_random(
                            self.config['training']['eval_games']
                        )

                        logger.info(
                            f"回合 {episode}: vs 隨機對手勝率: {win_rate:.3f}"
                        )

                    self.win_rates.append(win_rate)
                    avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0

                    logger.info(f"平均獎勵: {avg_reward:.3f}, 回合長度: {episode_length}")

                    # 更新學習率調度器
                    self.agent.scheduler.step(win_rate)
                    current_lr = self.agent.optimizer.param_groups[0]['lr']
                    logger.info(f"當前學習率: {current_lr:.2e}")

                    # 保存最佳模型
                    if win_rate > best_win_rate and episode > 2000:
                        best_win_rate = win_rate
                        episodes_since_improvement = 0
                        self.save_checkpoint(f"best_model_wr_{win_rate:.3f}.pt")
                        logger.info(f"新的最佳模型！評估分數: {win_rate:.3f}")
                    else:
                        episodes_since_improvement += self.config['training']['eval_frequency']

                    # 早停
                    if episodes_since_improvement >= self.config['training']['early_stopping_patience']:
                        logger.info(f"在第 {episode} 回合早停")
                        break

                    # 每100回合進行遊戲可視化
                    if episode % 100 == 0 and VISUALIZATION_AVAILABLE:
                        try:
                            # 確定對手類型
                            if episode >= 5000:
                                opponent_type = "minimax"
                            elif episode >= 3000:
                                opponent_type = "self_play"
                            else:
                                opponent_type = "random"
                            
                            logger.info(f"第 {episode} 回合：展示對戰 {opponent_type} 對手")
                            self.demo_game_with_visualization(opponent_type)
                        except Exception as e:
                            logger.warning(f"可視化第 {episode} 回合時出錯: {e}")

                # 定期檢查點
                if episode % self.config['training']['checkpoint_frequency'] == 0 and episode > 0:
                    self.save_checkpoint(f"checkpoint_episode_{episode}.pt")

            except Exception as e:
                logger.error(f"訓練回合 {episode} 時出錯: {e}")
                continue

        logger.info("訓練完成！")
        logger.info(f"最佳勝率: {best_win_rate:.3f}")
        return self.agent

    def visualize_game(self, agent1_func, agent2_func, agent1_name="Agent1", agent2_name="Agent2", save_path=None):
        """可視化一局遊戲的過程"""
        if not VISUALIZATION_AVAILABLE:
            logger.warning("⚠️ matplotlib不可用，跳過遊戲可視化")
            return None
        
        logger.info(f"🎮 開始可視化遊戲: {agent1_name} vs {agent2_name}")
        
        # 記錄遊戲狀態
        game_history = []
        move_history = []
        
        env = make("connectx", debug=False)
        env.reset()
        
        move_count = 0
        max_moves = 50
        
        while not env.done and move_count < max_moves:
            # 記錄當前狀態
            current_board = None
            current_player = None
            
            for player_idx in range(2):
                if env.state[player_idx]['status'] == 'ACTIVE':
                    board, player_mark = self.agent.extract_board_and_mark(env.state, player_idx)
                    current_board = board
                    current_player = player_mark
                    break
            
            if current_board is not None:
                game_history.append(current_board.copy())
            
            # 獲取動作
            actions = []
            for player_idx in range(2):
                if env.state[player_idx]['status'] == 'ACTIVE':
                    board, player_mark = self.agent.extract_board_and_mark(env.state, player_idx)
                    state = self.agent.encode_state(board, player_mark)
                    valid_actions = self.agent.get_valid_actions(board)
                    
                    if player_mark == 1:
                        result = agent1_func(state, valid_actions, False)
                        agent_name = agent1_name
                    else:
                        result = agent2_func(state, valid_actions, False)
                        agent_name = agent2_name
                    
                    # 處理返回值
                    if len(result) >= 3:
                        action = result[0]
                    else:
                        action = result
                    
                    actions.append(action)
                    move_history.append({
                        'move': move_count + 1,
                        'player': player_mark,
                        'agent_name': agent_name,
                        'action': action,
                        'board_before': board.copy()
                    })
                    break
            
            # 執行動作
            if actions:
                env.step(actions)
            
            move_count += 1
        
        # 獲取最終狀態和結果
        final_board = None
        final_result = "進行中"
        
        if env.done and len(env.state) >= 2:
            for player_idx in range(2):
                board, _ = self.agent.extract_board_and_mark(env.state, player_idx)
                final_board = board
                break
            
            if final_board is not None:
                game_history.append(final_board.copy())
            
            # 判斷結果
            if env.state[0].get('reward', 0) == 1:
                final_result = f"{agent1_name} 獲勝!"
            elif env.state[1].get('reward', 0) == 1:
                final_result = f"{agent2_name} 獲勝!"
            else:
                final_result = "平局"
        
        # 創建可視化
        self._create_game_visualization(game_history, move_history, agent1_name, agent2_name, final_result, save_path)
        
        logger.info(f"遊戲結束: {final_result}, 總步數: {len(move_history)}")
        return final_result

    def _create_game_visualization(self, game_history, move_history, agent1_name, agent2_name, final_result, save_path=None):
        """創建遊戲可視化"""
        if not game_history:
            return
        
        fig, axes = plt.subplots(1, min(len(game_history), 6), figsize=(18, 3))
        if len(game_history) == 1:
            axes = [axes]
        elif len(game_history) > 6:
            # 如果超過6步，只顯示關鍵步數
            indices = [0, len(game_history)//4, len(game_history)//2, 3*len(game_history)//4, len(game_history)-1]
            game_history = [game_history[i] for i in indices if i < len(game_history)]
            axes = axes[:len(game_history)]
        
        fig.suptitle(f'🎮 ConnectX 對戰: {agent1_name} vs {agent2_name}\n結果: {final_result}', 
                    fontsize=14, fontweight='bold')
        
        colors = {0: 'white', 1: 'red', 2: 'blue'}
        player_names = {1: agent1_name, 2: agent2_name}
        
        for idx, (ax, board_state) in enumerate(zip(axes, game_history)):
            # 轉換為6x7矩陣
            board_matrix = np.array(board_state).reshape(6, 7)
            
            # 創建顏色映射
            board_colors = np.zeros((6, 7, 3))
            for i in range(6):
                for j in range(7):
                    if board_matrix[i, j] == 1:
                        board_colors[i, j] = [1, 0.3, 0.3]  # 紅色
                    elif board_matrix[i, j] == 2:
                        board_colors[i, j] = [0.3, 0.3, 1]  # 藍色
                    else:
                        board_colors[i, j] = [0.95, 0.95, 0.95]  # 淺灰色
            
            ax.imshow(board_colors)
            
            # 添加網格
            for i in range(7):
                ax.axvline(i - 0.5, color='black', linewidth=1)
            for i in range(6):
                ax.axhline(i - 0.5, color='black', linewidth=1)
            
            # 添加標題
            if idx == 0:
                ax.set_title('初始狀態', fontsize=10)
            elif idx == len(game_history) - 1:
                ax.set_title('最終狀態', fontsize=10)
            else:
                step_num = idx if len(game_history) <= 6 else [0, len(game_history)//4, len(game_history)//2, 3*len(game_history)//4, len(game_history)-1][idx]
                ax.set_title(f'第 {step_num} 步', fontsize=10)
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        # 添加圖例
        legend_elements = [
            patches.Patch(color=[1, 0.3, 0.3], label=f'{agent1_name} (紅)'),
            patches.Patch(color=[0.3, 0.3, 1], label=f'{agent2_name} (藍)')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=2)
        
        plt.tight_layout()
        
        # 保存或顯示
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"遊戲可視化已保存: {save_path}")
        else:
            # 創建保存目錄
            os.makedirs('game_visualizations', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = f'game_visualizations/game_{agent1_name}_vs_{agent2_name}_{timestamp}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"遊戲可視化已保存: {save_path}")
        
        plt.close()

    def demo_game_with_visualization(self, episode_num):
        """每100個episode進行一次可視化演示"""
        if not VISUALIZATION_AVAILABLE:
            return
        
        logger.info(f"🎬 Episode {episode_num}: 開始可視化演示")
        
        # 根據當前訓練階段選擇不同的對手
        if episode_num < 500:
            # 早期：主要對抗隨機對手
            opponent_types = ["隨機對手", "簡單Minimax"]
            weights = [0.7, 0.3]
        elif episode_num < 2000:
            # 中期：混合對手
            opponent_types = ["隨機對手", "Minimax", "自對弈"]
            weights = [0.4, 0.4, 0.2]
        else:
            # 後期：主要自對弈和高級對手
            opponent_types = ["Minimax", "自對弈"]
            weights = [0.6, 0.4]
        
        # 隨機選擇對手類型
        opponent_type = np.random.choice(opponent_types, p=weights)
        
        # 訓練中的智能體
        def trained_agent(state, valid_actions, training=False):
            return self.agent.select_action(state, valid_actions, training=training)
        
        # 根據選擇創建對手
        if opponent_type == "隨機對手":
            def opponent(state, valid_actions, training=False):
                action = np.random.choice(valid_actions)
                return action, 1.0 / len(valid_actions), 0.0
            opponent_name = "Random"
            
        elif opponent_type == "簡單Minimax" or opponent_type == "Minimax":
            def opponent(state, valid_actions, training=False):
                # 從狀態重建棋盤
                board = [0] * 42
                # 這裡需要從編碼狀態重建，簡化處理
                action = np.random.choice(valid_actions)  # 簡化版本
                return action, 1.0, 0.0
            opponent_name = "Minimax"
            
        else:  # 自對弈
            def opponent(state, valid_actions, training=False):
                return self.agent.select_action(state, valid_actions, training=training)
            opponent_name = "Self-Play"
        
        # 隨機決定誰先手
        if np.random.random() < 0.5:
            agent1_func, agent1_name = trained_agent, "PPO-Agent"
            agent2_func, agent2_name = opponent, opponent_name
        else:
            agent1_func, agent1_name = opponent, opponent_name
            agent2_func, agent2_name = trained_agent, "PPO-Agent"
        
        # 執行可視化
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'game_visualizations/episode_{episode_num}_{agent1_name}_vs_{agent2_name}_{timestamp}.png'
        
        result = self.visualize_game(agent1_func, agent2_func, agent1_name, agent2_name, save_path)
        
        logger.info(f"🎯 Episode {episode_num} 演示完成: {agent1_name} vs {agent2_name} - {result}")

    def save_checkpoint(self, filename):
        """保存檢查點"""
        try:
            # 確保checkpoints目錄存在
            os.makedirs("checkpoints", exist_ok=True)

            checkpoint = {
                'model_state_dict': self.agent.policy_net.state_dict(),
                'optimizer_state_dict': self.agent.optimizer.state_dict(),
                'episode_rewards': self.episode_rewards,
                'win_rates': self.win_rates,
                'config': self.config,
                # 添加模型結構信息以便載入時驗證
                'model_architecture': {
                    'input_size': self.config['agent']['input_size'],
                    'hidden_size': self.config['agent']['hidden_size'],
                    'num_layers': self.config['agent']['num_layers']
                },
                # 添加保存時間戳
                'save_timestamp': datetime.now().isoformat(),
                # 使用字符串而不是TorchVersion對象以避免序列化問題
                'pytorch_version': str(torch.__version__)
            }

            checkpoint_path = f"checkpoints/{filename}"
            torch.save(checkpoint, checkpoint_path)

            # 計算文件大小
            file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB

            logger.info(f"✅ 已保存檢查點: {filename}")
            logger.info(f"   文件大小: {file_size:.2f} MB")
            logger.info(f"   模型結構: {self.config['agent']['num_layers']} 層, {self.config['agent']['hidden_size']} 隱藏單元")

        except Exception as e:
            logger.error(f"保存檢查點時出錯: {e}")
            import traceback
            traceback.print_exc()

    def load_checkpoint(self, checkpoint_path):
        """載入檢查點並恢復訓練狀態"""
        try:
            if not os.path.exists(checkpoint_path):
                logger.error(f"檢查點文件不存在: {checkpoint_path}")
                return False

            logger.info(f"載入檢查點: {checkpoint_path}")

            # 為了兼容性，先嘗試使用 weights_only=False
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.agent.device, weights_only=False)
            except Exception as e1:
                logger.warning(f"使用 weights_only=False 載入失敗: {e1}")
                # 如果失敗，嘗試使用安全的全局對象
                try:
                    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                        checkpoint = torch.load(checkpoint_path, map_location=self.agent.device, weights_only=True)
                except Exception as e2:
                    logger.error(f"使用安全全局對象載入也失敗: {e2}")
                    # 最後嘗試只載入模型權重
                    try:
                        checkpoint = torch.load(checkpoint_path, map_location=self.agent.device, weights_only=True)
                        # 如果成功但是格式不對，嘗試包裝成正確格式
                        if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
                            # 假設直接載入的是state_dict
                            checkpoint = {'model_state_dict': checkpoint}
                            logger.warning("檢查點格式不標準，嘗試包裝為標準格式")
                    except Exception as e3:
                        logger.error(f"所有載入方法都失敗: {e3}")
                        return False

            # 檢查模型結構兼容性
            saved_state_dict = checkpoint['model_state_dict']
            current_state_dict = self.agent.policy_net.state_dict()

            # 檢查關鍵參數匹配
            if 'config' in checkpoint:
                saved_config = checkpoint['config']
                current_agent_config = self.config['agent']
                saved_agent_config = saved_config.get('agent', {})

                # 檢查關鍵參數是否匹配
                key_params = ['input_size', 'hidden_size', 'num_layers']
                mismatch_found = False
                for param in key_params:
                    if (param in current_agent_config and param in saved_agent_config):
                        if current_agent_config[param] != saved_agent_config[param]:
                            logger.error(f"模型結構不匹配 {param}: 當前={current_agent_config[param]}, 保存的={saved_agent_config[param]}")
                            mismatch_found = True

                if mismatch_found:
                    logger.error("模型結構不匹配，無法載入檢查點")
                    return False

            # 檢查state_dict鍵是否匹配
            saved_keys = set(saved_state_dict.keys())
            current_keys = set(current_state_dict.keys())

            missing_keys = current_keys - saved_keys
            unexpected_keys = saved_keys - current_keys

            if missing_keys:
                logger.warning(f"當前模型缺少的權重鍵: {list(missing_keys)[:10]}...")  # 只顯示前10個
            if unexpected_keys:
                logger.warning(f"保存模型中多餘的權重鍵: {list(unexpected_keys)[:10]}...")  # 只顯示前10個

            # 如果鍵不匹配太多，拒絕載入
            if len(missing_keys) > 0 or len(unexpected_keys) > 10:  # 允許少量不匹配
                logger.error("模型權重鍵嚴重不匹配，拒絕載入")
                logger.info("提示：可能需要使用相同的模型配置，或從頭開始訓練")
                return False

            # 載入模型狀態（strict=False允許部分匹配）
            missing, unexpected = self.agent.policy_net.load_state_dict(saved_state_dict, strict=False)
            if missing:
                logger.warning(f"載入時缺少的鍵: {missing[:5]}...")
            if unexpected:
                logger.warning(f"載入時未使用的鍵: {unexpected[:5]}...")

            logger.info("✅ 模型權重載入成功")

            # 載入優化器狀態（可選，因為結構可能不匹配）
            if 'optimizer_state_dict' in checkpoint:
                try:
                    self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.info("✅ 優化器狀態載入成功")
                except Exception as e:
                    logger.warning(f"優化器狀態載入失敗（將使用新的優化器狀態）: {e}")

            # 載入訓練歷史（可選）
            if 'episode_rewards' in checkpoint:
                self.episode_rewards = checkpoint['episode_rewards']
                logger.info(f"✅ 載入訓練歷史: {len(self.episode_rewards)} 個回合")

            if 'win_rates' in checkpoint:
                self.win_rates = checkpoint['win_rates']
                logger.info(f"✅ 載入勝率歷史: {len(self.win_rates)} 個評估點")

            logger.info("🎉 檢查點載入完成！")
            return True

        except Exception as e:
            logger.error(f"載入檢查點時出錯: {e}")
            logger.error("可能的解決方案:")
            logger.error("1. 檢查模型配置是否與保存時一致")
            logger.error("2. 確認檢查點文件未損壞")
            logger.error("3. 考慮從頭開始訓練")
            import traceback
            traceback.print_exc()
            return False

    def fix_checkpoint_compatibility(self, old_checkpoint_path, new_checkpoint_path=None):
        """修復舊檢查點的兼容性問題"""
        if new_checkpoint_path is None:
            new_checkpoint_path = old_checkpoint_path.replace('.pt', '_fixed.pt')

        try:
            logger.info(f"嘗試修復檢查點: {old_checkpoint_path}")

            # 嘗試使用舊方法載入
            checkpoint = torch.load(old_checkpoint_path, map_location='cpu', weights_only=False)

            # 修復可能的問題
            if 'pytorch_version' in checkpoint and hasattr(checkpoint['pytorch_version'], '__version__'):
                checkpoint['pytorch_version'] = str(checkpoint['pytorch_version'])
                logger.info("已修復PyTorch版本序列化問題")

            # 保存修復後的檢查點
            torch.save(checkpoint, new_checkpoint_path)
            logger.info(f"已保存修復後的檢查點: {new_checkpoint_path}")

            return new_checkpoint_path

        except Exception as e:
            logger.error(f"修復檢查點失敗: {e}")
            return None

def create_default_config():
    """創建默認配置"""
    return {
        'agent': {
            'input_size': 126,  # 6*7*3 通道
            'hidden_size': 512,
            'num_layers': 4,  # 修正：使用合理的層數
            'learning_rate': 1e-4,  # 提高學習率，加快初期學習
            'weight_decay': 1e-4,
            'gamma': 0.995,  # 提高折扣因子，更重視長期獎勵
            'eps_clip': 0.15,  # 降低剪裁範圍，更保守的更新
            'k_epochs': 6,  # 增加更新次數
            'entropy_coef': 0.02,  # 增加探索
            'value_coef': 0.8,  # 增加價值函數權重
            'buffer_size': 8000,  # 增加經驗緩衝區
            'min_batch_size': 256,  # 增加批次大小
            'gae_lambda': 0.97  # 提高GAE lambda
        },
        'training': {
            'max_episodes': 2000000,
            'eval_frequency': 200,  # 更頻繁評估
            'eval_games': 30,  # 減少評估遊戲數，節省時間
            'checkpoint_frequency': 1000,  # 更頻繁保存
            'early_stopping_patience': 3000,  # 更早停止
            'continuous_learning_frequency': 5,  # 更頻繁的持續學習
            'continuous_batch_size': 32,          # 減小批次，減少干擾
            'continuous_learning_rate': 5e-6,     # 降低學習率
            'use_batch_loading': True,
            'memory_batch_size': 10000,
            'curriculum_learning': True,           # 啟用課程學習
            'opponent_diversity': True,            # 啟用對手多樣性
            # 新增多樣性參數
            'self_play_diversity': True,           # 啟用自對弈多樣性
            'base_randomness': 0.25,               # 基礎隨機性水平
            'diversity_decay': 0.0001,             # 隨機性衰減率
            'exploration_strategies': ['noisy', 'exploration', 'temperature', 'anti_pattern'],  # 可用策略
            'strategy_weights': [0.3, 0.3, 0.2, 0.2]  # 策略權重
        },
        'evaluation': {
            'mode': 'comprehensive',  # random, comprehensive, detailed, minimax
            'minimax_depth': 10,
            'weights': {
                'vs_random': 0.4,
                'vs_minimax': 0.4,
                'self_play': 0.2
            }
        }
    }

def main():
    config_path = 'config.yaml'
    load = 'checkpoints/checkpoint_episode_70000.pt'
    # load = False

    pretrain_epochs = 1
    pretrain = False
    skip_rl=False
    pretrain_dataset='connectx-state-action-value.txt'
    continuous_learning = True  # 啟用持續學習
    use_batch_loading = True    # 使用批次載入預訓練
    memory_batch_size = 10000   # 批次載入記憶體大小
    max_lines=50000

    # 記憶體安全檢查
    logger.info(f"當前配置 - max_lines: {max_lines}, memory_batch_size: {memory_batch_size}")

    # 如果記憶體不足，自動調整參數
    import psutil
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    logger.info(f"可用記憶體: {available_memory_gb:.1f} GB")

    if available_memory_gb < 2.0:  # 少於2GB記憶體時自動調整
        logger.warning("檢測到記憶體不足，自動調整參數...")
        max_lines = min(max_lines, 2000)  # 最多載入2000行
        memory_batch_size = min(memory_batch_size, 50)  # 減小批次大小
        logger.info(f"調整後 - max_lines: {max_lines}, memory_batch_size: {memory_batch_size}")

    # 對於持續學習，進一步限制記憶體使用
    if continuous_learning:
        if max_lines == -1:
            with open(pretrain_dataset, 'r') as f:
                total_lines = sum(1 for line in f if line.strip())

        continuous_learning_max = max_lines
        logger.info(f"持續學習將使用最多 {continuous_learning_max} 個樣本")

    # 創建默認配置
    import os
    if not os.path.exists(config_path):
        logger.info(f"創建默認配置文件: {config_path}")
        config = create_default_config()
        # 添加監督學習配置
        config['agent']['supervised_learning_rate'] = 1e-4
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # 開始訓練
    try:
        trainer = ConnectXTrainer(config_path)

        # 如果指定了載入路徑，嘗試載入模型
        if load:
            logger.info(f"嘗試載入模型: {load}")
            success = trainer.load_checkpoint(load)

            if not success:
                logger.warning("直接載入失敗，嘗試修復檢查點...")
                fixed_path = trainer.fix_checkpoint_compatibility(load)
                if fixed_path:
                    logger.info(f"嘗試載入修復後的檢查點: {fixed_path}")
                    success = trainer.load_checkpoint(fixed_path)

            if success:
                logger.info("模型載入成功，繼續訓練...")
            else:
                logger.warning("模型載入失敗，將從頭開始訓練")
        else:
            logger.info("從頭開始訓練新模型...")

        # 監督學習預訓練
        pretrain_states = None
        pretrain_action_values = None

        if pretrain:
            logger.info("開始監督學習預訓練階段...")

            # 選擇預訓練方式
            if use_batch_loading:
                logger.info("使用批次載入模式進行預訓練...")
                success = trainer.supervised_pretrain_batch_loading(
                    file_path=pretrain_dataset,
                    epochs=pretrain_epochs,
                    batch_size=128,
                    memory_batch_size=memory_batch_size,
                    max_lines=max_lines
                )

                if success:
                    logger.info("批次載入預訓練成功完成！")

                    # 為持續學習載入少量數據（避免記憶體問題）
                    if continuous_learning:
                        logger.info("為持續學習載入少量數據樣本...")
                        # 使用預設的少量樣本數
                        pretrain_states, pretrain_action_values = trainer.load_state_action_dataset(pretrain_dataset, max_lines=continuous_learning_max)
                        if pretrain_states is not None and pretrain_action_values is not None:
                            logger.info(f"載入 {len(pretrain_states)} 個樣本用於持續學習")
                        else:
                            logger.warning("無法載入持續學習數據，將禁用持續學習功能")
                            continuous_learning = False
                else:
                    logger.error("批次載入預訓練失敗！")
                    return
            else:
                logger.info("使用傳統模式進行預訓練...")
                pretrain_states, pretrain_action_values = trainer.load_state_action_dataset(pretrain_dataset, max_lines=max_lines)

                if pretrain_states is not None and pretrain_action_values is not None:
                    success = trainer.supervised_pretrain(
                        pretrain_states, pretrain_action_values,
                        epochs=pretrain_epochs
                    )

                    if success:
                        logger.info("預訓練成功完成！")
                    else:
                        logger.error("預訓練失敗！")
                        return
                else:
                    logger.error("無法載入預訓練數據集！")
                    return

        # 啟用持續學習（如果有數據集且啟用了該功能）
        if continuous_learning:
            if pretrain_states is not None and pretrain_action_values is not None:
                # 使用已載入的預訓練數據
                trainer.enable_continuous_learning(pretrain_states, pretrain_action_values)
                logger.info(f"已啟用持續學習，使用 {len(pretrain_states)} 個樣本")
            else:
                # 如果沒有預訓練數據，禁用持續學習
                logger.warning("沒有可用的預訓練數據，持續學習功能已禁用")
                continuous_learning = False

        # 強化學習訓練
        if not skip_rl:
            logger.info("開始PPO強化學習訓練...")
            trained_agent = trainer.train()

        else:
            logger.info("只進行預訓練，跳過強化學習階段。")

        # 輸出最佳模型路徑
        best_models = [f for f in os.listdir('checkpoints') if f.startswith('best_model')]
        if best_models:
            best_model_path = f"checkpoints/{sorted(best_models)[-1]}"
            logger.info(f"最佳模型位於: {best_model_path}")

        logger.info("訓練完成！準備用於 Kaggle 提交。")

    except Exception as e:
        logger.error(f"訓練過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()




    import requests
    import os
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("請設置環境變量 TELEGRAM_BOT_TOKEN")
        return

    BOT_TOKEN = token

    CHAT_ID   = "6166024220"

    def send_telegram(msg: str):
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": msg}
        r = requests.post(url, data=payload)
        if not r.ok:
            print("❌ 發送失敗：", r.text)

    send_telegram("🎉 訓練完畢！！")


if __name__ == "__main__":
    main()
