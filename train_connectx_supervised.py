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
    """ConnectX 深度神經網路"""

    def __init__(self, input_size=126, hidden_size=256, num_layers=3):
        super(ConnectXNet, self).__init__()

        # 輸入層
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 隱藏層
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size)
            ) for _ in range(num_layers)
        ])

        # 策略頭（動作概率）
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 7),  # 7 列
            nn.Softmax(dim=-1)
        )

        # 價值頭（狀態價值）
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
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
        """載入訓練數據集"""
        states = []
        action_values = []
        skipped_lines = 0

        try:
            logger.info(f"載入訓練數據集: {file_path}")
            logger.info(f"限制載入行數: {max_lines}")

            with open(file_path, 'r') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line.strip())

            logger.info(f"實際載入行數: {len(lines)}")

            with tqdm(total=len(lines), desc="載入數據") as pbar:
                for line_idx, line in enumerate(lines):
                    pbar.update(1)

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

                        # 轉換棋盤狀態
                        board_state = []
                        for char in board_str:
                            if char not in '012':
                                raise ValueError(f"無效字符: {char}")
                            board_state.append(int(char))

                        # 解析動作價值
                        action_vals = []
                        for i in range(1, 8):  # 7個動作值
                            val_str = parts[i].strip()
                            if val_str == '':
                                action_vals.append(-999.0)  # 無效動作
                            else:
                                try:
                                    action_vals.append(float(val_str))
                                except ValueError:
                                    action_vals.append(0.0)

                        # 檢查有效動作
                        valid_actions = [i for i, val in enumerate(action_vals) if val > -900]
                        if len(valid_actions) == 0:
                            skipped_lines += 1
                            continue

                        # 編碼狀態
                        encoded_state = self.encode_state(board_state, 1)  # 假設從先手視角

                        states.append(encoded_state)
                        action_values.append(action_vals)

                    except Exception as e:
                        logger.debug(f"第 {line_idx + 1} 行解析錯誤: {e}")
                        skipped_lines += 1
                        continue

            logger.info(f"數據載入完成:")
            logger.info(f"  成功解析: {len(states)} 個樣本")
            logger.info(f"  跳過行數: {skipped_lines}")

            if len(states) == 0:
                logger.error("沒有成功解析任何數據！")
                return None, None

            return np.array(states), np.array(action_values)

        except FileNotFoundError:
            logger.error(f"找不到數據集文件: {file_path}")
            return None, None
        except Exception as e:
            logger.error(f"載入數據集時出錯: {e}")
            return None, None

    def train(self, epochs=100, batch_size=128, max_lines=10000):
        """監督學習訓練"""
        logger.info("🚀 開始監督學習訓練")

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

            # 學習率調度
            self.scheduler.step(avg_loss)

            # 記錄最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint("best_supervised_model.pt")

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

        logger.info("✅ 監督學習訓練完成")
        return self.policy_net

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
            'hidden_size': 256,     # 隱藏層大小
            'num_layers': 3,        # 隱藏層數量
            'learning_rate': 0.001, # 學習率
            'weight_decay': 0.0001  # 權重衰減
        },
        'training': {
            'epochs': 200,          # 訓練epochs
            'batch_size': 128,      # 批次大小
            'max_lines': 50000,     # 最大數據集行數
            'eval_games': 100       # 評估遊戲數量
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
            max_lines=config['training']['max_lines']
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
