#!/usr/bin/env python3
"""
ConnectX 監督學習訓練腳本
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
    """强化学习用的 ConnectX 深度神经网络"""

    def __init__(self, input_size=126, hidden_size=150, num_layers=4):
        super(ConnectXNet, self).__init__()

        # 确保参数是正确的Python int类型
        input_size = int(input_size)
        hidden_size = int(hidden_size)
        num_layers = int(num_layers)

        # 输入层
        print(f"Initializing ConnectXNet with input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}")
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 隐藏层（残差连接 + 层正规化代替批量正规化）
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

        # 策略头（动作概率）
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),  # 使用 LayerNorm 代替 BatchNorm1d
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 7),  # 7 列
            nn.Softmax(dim=-1)
        )

        # 价值头（状态价值）
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),  # 使用 LayerNorm 代替 BatchNorm1d
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # 输入处理
        x = self.input_layer(x)

        # 残差连接
        for hidden_layer in self.hidden_layers:
            residual = x
            x = hidden_layer(x)
            x = torch.relu(x + residual)

        # 输出头
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

    def __init__(self, config_path_or_dict="config_supervised.yaml"):
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
                            logger.warning(f"第 {line_idx + 1} 行棋盤狀態長度錯誤，期望42，得到{len(board_part)}")
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
                            logger.warning(f"第 {line_idx + 1} 行動作價值數量錯誤，期望7個，得到{len(action_parts)}個")
                            skipped_lines += 1
                            continue

                        # 處理動作價值（包括空值）
                        action_vals = []
                        for i, val_str in enumerate(action_parts):
                            val_str = val_str.strip()

                            if val_str == '':
                                # 空值表示該列已滿，設為極大負值（不可下）
                                action_vals.append(-999.0)
                            else:
                                try:
                                    # 嘗試轉換為數字
                                    val = float(val_str)
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
                        encoded_state = self.agent.encode_state(board_state, 1)

                        states.append(encoded_state)
                        action_values.append(action_vals)

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
            states_array = np.array(states, dtype=np.float32)
            action_values_array = np.array(action_values, dtype=np.float32)

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


    def create_connectx_environment(self):
        """創建ConnectX遊戲環境"""
        try:
            env = make("connectx", debug=False)
            logger.info("✅ ConnectX環境創建成功")
            return env
        except Exception as e:
            logger.error(f"❌ 創建ConnectX環境失敗: {e}")
            return None

    def play_game(self, agent1_func, agent2_func, training=True):
        """執行一局遊戲"""
        try:
            env = self.create_connectx_environment()
            if env is None:
                return 0, 0

            # 重置環境
            env.reset()
            config = env.configuration
            
            # 遊戲狀態
            done = False
            step_count = 0
            max_steps = config.rows * config.columns  # 最大步數
            
            # 儲存遊戲歷史
            game_states = []
            game_actions = []
            game_probs = []
            
            while not done and step_count < max_steps:
                # 獲取當前狀態
                current_state = env.state
                current_player = step_count % 2  # 0 or 1
                
                # 提取棋盤和玩家標記
                board, mark = self.agent.extract_board_and_mark(current_state, current_player)
                
                # 編碼狀態
                encoded_state = self.agent.encode_state(board, mark)
                
                # 獲取有效動作
                valid_actions = self.agent.get_valid_actions(board)
                
                if not valid_actions:
                    logger.warning("沒有有效動作，遊戲結束")
                    break
                
                # 選擇智能體函數
                if current_player == 0:
                    agent_func = agent1_func
                else:
                    agent_func = agent2_func
                
                # 獲取動作
                try:
                    action, prob, value, is_dangerous = agent_func(encoded_state, valid_actions, training)
                    action = int(action)  # 確保是Python int
                    
                    if action not in valid_actions:
                        logger.warning(f"無效動作 {action}，選擇隨機動作")
                        action = np.random.choice(valid_actions)
                        action = int(action)
                    
                except Exception as e:
                    logger.error(f"智能體選擇動作時出錯: {e}")
                    action = int(np.random.choice(valid_actions))
                    prob = 1.0 / len(valid_actions)
                    value = 0.0
                
                # 儲存訓練數據
                if training and current_player == 0:  # 只儲存玩家1的數據
                    game_states.append(encoded_state)
                    game_actions.append(action)
                    game_probs.append(prob)
                
                # 執行動作
                try:
                    env.step([action, None] if current_player == 0 else [None, action])
                except Exception as e:
                    logger.error(f"執行動作時出錯: {e}")
                    break
                
                # 檢查遊戲是否結束
                if len(env.state) >= 2:
                    status_0 = env.state[0].get('status', 'ACTIVE')
                    status_1 = env.state[1].get('status', 'ACTIVE')
                    
                    if status_0 != 'ACTIVE' or status_1 != 'ACTIVE':
                        done = True
                
                step_count += 1
            
            # 計算獎勵
            reward = 0
            if len(env.state) >= 2:
                reward_0 = env.state[0].get('reward', 0)
                reward_1 = env.state[1].get('reward', 0)
                
                if reward_0 > reward_1:
                    reward = 1  # 玩家1獲勝
                elif reward_1 > reward_0:
                    reward = -1  # 玩家2獲勝
                else:
                    reward = 0  # 平局
            
            # 儲存遊戲轉換
            if training and game_states:
                for i, (state, action, prob) in enumerate(zip(game_states, game_actions, game_probs)):
                    # 計算折扣獎勵
                    discounted_reward = reward * (self.agent.gamma ** (len(game_states) - i - 1))
                    self.agent.store_transition(state, action, prob, discounted_reward, i == len(game_states) - 1)
            
            return reward, step_count
            
        except Exception as e:
            logger.error(f"遊戲執行出錯: {e}")
            import traceback
            traceback.print_exc()
            return 0, 0

    def create_agent_function(self, strategy='standard'):
        """創建智能體函數"""
        def agent_func(state, valid_actions, training):
            try:
                action, prob, value = self.agent.select_action(state, valid_actions, training)
                return int(action), prob, value, False
            except Exception as e:
                logger.error(f"智能體選擇動作出錯: {e}")
                action = int(np.random.choice(valid_actions))
                return action, 1.0/len(valid_actions), 0.0, False
        
        return agent_func

    def random_agent_func(self, state, valid_actions, training=True):
        """隨機智能體"""
        action = int(np.random.choice(valid_actions))
        return action, 1.0/len(valid_actions), 0.0, False

    def supervised_train(self, epochs=100, batch_size=128, max_lines=10000):
        """使用監督學習進行訓練"""
        logger.info("🚀 開始監督學習訓練")
        
        # 載入數據集
        states, action_values = self.load_state_action_dataset(max_lines=max_lines)
        if states is None or action_values is None:
            logger.error("❌ 數據集載入失敗")
            return None
        
        logger.info(f"📊 數據集載入成功: {len(states)} 個樣本")
        
        # 準備訓練
        self.agent.policy_net.train()
        total_samples = len(states)
        
        # 訓練循環
        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            num_batches = 0
            
            # 隨機打亂數據
            indices = np.random.permutation(total_samples)
            
            # 批次訓練
            for batch_start in range(0, total_samples, batch_size):
                batch_end = min(batch_start + batch_size, total_samples)
                batch_indices = indices[batch_start:batch_end]
                
                # 準備批次數據
                batch_states = torch.FloatTensor(states[batch_indices]).to(self.agent.device)
                batch_action_values = torch.FloatTensor(action_values[batch_indices]).to(self.agent.device)
                
                # 前向傳播
                predicted_probs, predicted_values = self.agent.policy_net(batch_states)
                
                # 計算目標
                # 對於動作概率，使用softmax將動作價值轉換為概率分佈
                # 首先處理無效動作（值為-999的動作）
                valid_mask = (batch_action_values > -900).float()
                masked_action_values = batch_action_values * valid_mask + (-1000) * (1 - valid_mask)
                
                # Softmax轉換為目標概率
                target_probs = F.softmax(masked_action_values / 0.1, dim=1)  # 溫度參數0.1
                
                # 對於價值，使用最大動作價值作為目標
                target_values = torch.max(batch_action_values * valid_mask + (-1000) * (1 - valid_mask), dim=1)[0].unsqueeze(1)
                # 正規化價值到[-1, 1]範圍
                target_values = torch.tanh(target_values / 10.0)
                
                # 計算損失
                policy_loss = F.kl_div(torch.log(predicted_probs + 1e-8), target_probs, reduction='batchmean')
                value_loss = F.mse_loss(predicted_values, target_values)
                
                total_loss_batch = policy_loss + 0.5 * value_loss
                
                # 反向傳播
                self.agent.optimizer.zero_grad()
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.policy_net.parameters(), 0.5)
                self.agent.optimizer.step()
                
                # 累積損失
                total_loss += total_loss_batch.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1
            
            # 計算平均損失
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            avg_policy_loss = total_policy_loss / num_batches if num_batches > 0 else 0
            avg_value_loss = total_value_loss / num_batches if num_batches > 0 else 0
            
            epoch_time = time.time() - epoch_start_time
            
            # 每10個epoch報告一次
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] "
                          f"Loss: {avg_loss:.6f} "
                          f"Policy: {avg_policy_loss:.6f} "
                          f"Value: {avg_value_loss:.6f} "
                          f"Time: {epoch_time:.2f}s")
            
            # 學習率調度
            self.agent.scheduler.step(avg_loss)
            
            # 保存檢查點
            if (epoch + 1) % 1000 == 0 and epoch > 2000:
                checkpoint_name = f"supervised_epoch_{epoch+1}.pt"
                self.save_checkpoint(checkpoint_name)
        
        logger.info("✅ 監督學習訓練完成")
        return self.agent

    def evaluate_agent(self, num_games=100):
        """評估智能體性能"""
        logger.info(f"🎯 開始評估智能體性能 ({num_games} 局遊戲)")
        
        wins = 0
        draws = 0
        losses = 0
        
        agent_func = self.create_agent_function()
        
        for i in range(num_games):
            try:
                # 與隨機對手對弈
                reward, steps = self.play_game(agent_func, self.random_agent_func, training=False)
                
                if reward > 0:
                    wins += 1
                elif reward == 0:
                    draws += 1
                else:
                    losses += 1
                
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

# 首先修复配置文件的加载问题
def create_training_config():
    """创建训练配置"""
    config = {
        'agent': {
            'input_size': 126,      # 3个通道 × 42个位置 = 126
            'hidden_size': 150,     # 隐藏层大小
            'num_layers': 300,        # 隐藏层数量 (改为合理数量)
            'learning_rate': 0.001, # 学习率
            'weight_decay': 0.0001, # 权重衰减
            'gamma': 0.99,          # 折扣因子
            'eps_clip': 0.2,        # PPO剪裁参数
            'k_epochs': 4,          # PPO更新次数
            'entropy_coef': 0.01,   # 熵系数
            'value_coef': 0.5,      # 价值系数
            'gae_lambda': 0.99,     # GAE参数
            'buffer_size': 10000,   # 经验缓冲区大小
            'min_batch_size': 64    # 最小批次大小
        },
        'training': {
            'supervised_epochs': 200,     # 监督学习epochs
            'batch_size': 128,           # 批次大小
            'max_dataset_lines': -1,  # 最大数据集行数
            'evaluation_games': 100,     # 评估游戏数量
            'checkpoint_frequency': 50,  # 检查点保存频率
            'log_frequency': 100         # 日志记录频率
        }
    }
    return config
def save_config(config, filename='config_supervised.yaml'):
    """保存配置到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"✅ 配置已保存到 {filename}")
def main():
    """主训练函数"""
    print("🎮 ConnectX 监督学习训练")
    print("=" * 50)
    
    # 创建必要目录
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 创建并使用配置字典而不是文件
    config = create_training_config()
    
    # 检查数据集文件
    dataset_file = "connectx-state-action-value.txt"
    if not os.path.exists(dataset_file):
        logger.error(f"❌ 找不到数据集文件: {dataset_file}")
        logger.error("请确保文件存在于当前目录中")
        return
    
    logger.info(f"✅ 找到数据集文件: {dataset_file}")
    
    # 检查CUDA可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🔧 使用设备: {device}")
    
    try:
        # 创建训练器，直接传入配置字典
        trainer = ConnectXTrainer(config)
        logger.info("✅ 训练器创建成功")
        
        # 显示训练配置
        print("\n📋 训练配置:")
        print(f"   网络结构: {config['agent']['hidden_size']} 隐藏单元, {config['agent']['num_layers']} 层")
        print(f"   学习率: {config['agent']['learning_rate']}")
        print(f"   监督学习epochs: {config['training']['supervised_epochs']}")
        print(f"   批次大小: {config['training']['batch_size']}")
        print(f"   最大数据集行数: {config['training']['max_dataset_lines']}")
        
        # 开始监督学习训练
        print("\n🚀 开始监督学习训练...")
        start_time = time.time()
        
        trained_agent = trainer.supervised_train(
            epochs=config['training']['supervised_epochs'],
            batch_size=config['training']['batch_size'],
            max_lines=config['training']['max_dataset_lines']
        )
        
        training_time = time.time() - start_time
        
        if trained_agent is not None:
            logger.info(f"✅ 监督学习训练完成！用时: {training_time:.1f}秒")
            
            # 保存最终模型
            final_checkpoint = f"supervised_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            trainer.save_checkpoint(final_checkpoint)
            logger.info(f"💾 最终模型已保存: checkpoints/{final_checkpoint}")
            
            # 评估模型性能
            print("\n🎯 评估模型性能...")
            win_rate = trainer.evaluate_agent(num_games=config['training']['evaluation_games'])
            
            print(f"\n🎉 训练完成!")
            print(f"   总用时: {training_time:.1f}秒 ({training_time/60:.1f}分钟)")
            print(f"   最终胜率: {win_rate:.1f}%")
            print(f"   模型保存位置: checkpoints/{final_checkpoint}")
            
            # 提供使用建议
            print("\n💡 使用建议:")
            if win_rate >= 80:
                print("   🌟 模型性能优异！可以直接用于比赛")
            elif win_rate >= 60:
                print("   👍 模型性能良好，建议进行更多训练或微调")
            else:
                print("   ⚠️ 模型性能需要改进，建议:")
                print("      - 增加训练epochs")
                print("      - 调整学习率")
                print("      - 增加数据集大小")
                print("      - 检查网络结构")
        
        else:
            logger.error("❌ 监督学习训练失败")
            
    except KeyboardInterrupt:
        logger.info("⏹️ 训练被用户中断")
    except Exception as e:
        logger.error(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    main()
