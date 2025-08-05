#!/usr/bin/env python3
"""
ConnectX 基於規則的智能體
直接使用 connectx-state-action-value.txt 數據集進行決策
"""

import os
import sys
import numpy as np
import logging
from kaggle_environments import make
from tqdm import tqdm

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rule_based.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RuleBasedConnectXAgent:
    """基於規則的ConnectX智能體"""
    
    def __init__(self, dataset_file="connectx-state-action-value.txt"):
        self.dataset_file = dataset_file
        self.state_value_dict = {}  # 存儲狀態->動作價值的映射
        self.load_dataset()
    
    def load_dataset(self):
        """載入數據集並建立狀態-動作價值字典"""
        logger.info(f"載入數據集: {self.dataset_file}")
        
        if not os.path.exists(self.dataset_file):
            logger.error(f"找不到數據集文件: {self.dataset_file}")
            return
        
        loaded_count = 0
        skipped_count = 0
        
        with open(self.dataset_file, 'r') as f:
            lines = f.readlines()
        
        logger.info(f"開始處理 {len(lines)} 行數據...")
        
        with tqdm(total=len(lines), desc="載入數據集") as pbar:
            for line_idx, line in enumerate(lines):
                pbar.update(1)
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # 解析一行數據
                    parts = line.split(',')
                    if len(parts) < 8:
                        skipped_count += 1
                        continue
                    
                    # 棋盤狀態（42個字符）
                    board_state = parts[0]
                    if len(board_state) != 42:
                        skipped_count += 1
                        continue
                    
                    # 動作價值（7個值）
                    action_values = []
                    for i in range(1, 8):
                        val_str = parts[i].strip()
                        if val_str == '':
                            action_values.append(None)  # 無效動作
                        else:
                            try:
                                action_values.append(float(val_str))
                            except ValueError:
                                action_values.append(None)
                    
                    # 存儲到字典中
                    self.state_value_dict[board_state] = action_values
                    loaded_count += 1
                    
                except Exception as e:
                    logger.debug(f"第 {line_idx + 1} 行解析錯誤: {e}")
                    skipped_count += 1
                    continue
        
        logger.info(f"數據集載入完成:")
        logger.info(f"  成功載入: {loaded_count} 個狀態")
        logger.info(f"  跳過: {skipped_count} 行")
        logger.info(f"  字典大小: {len(self.state_value_dict)}")
    
    def board_to_string(self, board):
        """將棋盤轉換為字符串格式"""
        return ''.join(map(str, board))
    
    def get_valid_actions(self, board):
        """獲取有效動作（列表頂部為空的列）"""
        return [col for col in range(7) if board[col] == 0]
    
    def check_winning_move(self, board, player, col):
        """檢查在指定列放置棋子是否能獲勝"""
        # 模擬放置棋子
        temp_board = board.copy()
        
        # 找到該列的最底部空位
        row = -1
        for r in range(5, -1, -1):  # 從下往上找
            if temp_board[r * 7 + col] == 0:
                temp_board[r * 7 + col] = player
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
            while 0 <= r < 6 and 0 <= c < 7 and temp_board[r * 7 + c] == player:
                count += 1
                r, c = r + dr, c + dc
            
            # 反方向檢查
            r, c = row - dr, col - dc
            while 0 <= r < 6 and 0 <= c < 7 and temp_board[r * 7 + c] == player:
                count += 1
                r, c = r - dr, c - dc
            
            if count >= 4:
                return True
        
        return False
    
    def select_action(self, board, player):
        """
        選擇動作的主要邏輯：
        1. 首先查找數據集中的最佳動作
        2. 如果數據集沒有，檢查是否能直接獲勝
        3. 如果對手下一步能獲勝，進行阻擋
        4. 否則隨機選擇
        """
        valid_actions = self.get_valid_actions(board)
        if not valid_actions:
            return 0  # 無有效動作，返回默認值
        
        # 1. 首先嘗試從數據集中找到最佳動作
        board_str = self.board_to_string(board)
        if board_str in self.state_value_dict:
            action_values = self.state_value_dict[board_str]
            
            # 找到有效動作中價值最高的
            best_action = -1
            best_value = float('-inf')
            
            for col in valid_actions:
                if col < len(action_values) and action_values[col] is not None:
                    value = action_values[col]
                    if value > best_value:
                        best_value = value
                        best_action = col
            
            if best_action != -1:
                logger.debug(f"數據集決策: 選擇列 {best_action}, 價值 {best_value}")
                return best_action
        
        # 2. 檢查是否能直接獲勝
        for col in valid_actions:
            if self.check_winning_move(board, player, col):
                logger.debug(f"獲勝機會: 選擇列 {col}")
                return col
        
        # 3. 檢查是否需要阻擋對手獲勝
        opponent = 3 - player  # 對手玩家號
        for col in valid_actions:
            if self.check_winning_move(board, opponent, col):
                logger.debug(f"阻擋對手: 選擇列 {col}")
                return col
        
        # 4. 如果以上都沒有，進行智能猜測
        # 優先選擇中間列（策略性更好）
        center_cols = [3, 2, 4, 1, 5, 0, 6]  # 按中心優先排序
        for col in center_cols:
            if col in valid_actions:
                logger.debug(f"中心策略: 選擇列 {col}")
                return col
        
        # 5. 最後隨機選擇
        action = np.random.choice(valid_actions)
        logger.debug(f"隨機選擇: 選擇列 {action}")
        return action
    
    def play_game_against_random(self):
        """與隨機對手進行一局遊戲"""
        try:
            env = make("connectx", debug=False)
            env.reset()
            
            done = False
            step_count = 0
            max_steps = 42
            
            while not done and step_count < max_steps:
                current_player = step_count % 2
                
                if current_player == 0:  # 我們的智能體
                    obs = env.state[0]['observation']
                    board = obs['board']
                    mark = obs['mark']
                    
                    action = self.select_action(board, mark)
                    
                else:  # 隨機對手
                    obs = env.state[1]['observation']
                    board = obs['board']
                    
                    valid_actions = self.get_valid_actions(board)
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
                    return 1, step_count  # 勝利
                elif reward_1 > reward_0:
                    return -1, step_count  # 失敗
                else:
                    return 0, step_count  # 平局
            
            return 0, step_count
            
        except Exception as e:
            logger.error(f"遊戲執行出錯: {e}")
            return -1, 0
    
    def evaluate(self, num_games=100):
        """評估智能體性能"""
        logger.info(f"開始評估智能體性能 ({num_games} 局遊戲)")
        
        wins = 0
        draws = 0
        losses = 0
        total_steps = 0
        
        for i in range(num_games):
            try:
                result, steps = self.play_game_against_random()
                total_steps += steps
                
                if result > 0:
                    wins += 1
                elif result == 0:
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
        avg_steps = total_steps / num_games if num_games > 0 else 0
        
        logger.info(f"📊 評估結果:")
        logger.info(f"   勝利: {wins} ({win_rate:.1f}%)")
        logger.info(f"   平局: {draws} ({draws/num_games*100:.1f}%)")
        logger.info(f"   失敗: {losses} ({losses/num_games*100:.1f}%)")
        logger.info(f"   平均步數: {avg_steps:.1f}")
        
        return win_rate


def main():
    """主函數"""
    print("🎮 ConnectX 基於規則的智能體")
    print("=" * 50)
    
    # 創建日誌目錄
    os.makedirs('logs', exist_ok=True)
    
    # 檢查數據集文件
    dataset_file = "connectx-state-action-value.txt"
    if not os.path.exists(dataset_file):
        logger.error(f"❌ 找不到數據集文件: {dataset_file}")
        return
    
    try:
        # 創建智能體
        agent = RuleBasedConnectXAgent(dataset_file)
        logger.info("✅ 基於規則的智能體創建成功")
        
        # 顯示數據集信息
        print(f"\n📊 數據集信息:")
        print(f"   已載入狀態數: {len(agent.state_value_dict)}")
        
        # 進行性能評估
        print(f"\n🎯 開始性能評估...")
        win_rate = agent.evaluate(num_games=200)
        
        print(f"\n🎉 評估完成!")
        print(f"   最終勝率: {win_rate:.1f}%")
        
        # 性能分析
        if win_rate >= 90:
            print("🌟 優異性能！基於規則的方法非常有效")
        elif win_rate >= 70:
            print("👍 良好性能，規則邏輯運作良好")
        elif win_rate >= 50:
            print("⚖️ 一般性能，可能需要改進決策邏輯")
        else:
            print("⚠️ 性能較差，建議檢查數據集或邏輯")
        
        # 提供建議
        print(f"\n💡 使用建議:")
        print(f"   - 這個智能體直接使用數據集中的最佳動作")
        print(f"   - 當數據集沒有對應狀態時，使用獲勝/阻擋/中心策略")
        print(f"   - 可以直接用於ConnectX比賽")
        
    except KeyboardInterrupt:
        logger.info("⏹️ 評估被用戶中斷")
    except Exception as e:
        logger.error(f"❌ 運行過程中出錯: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
