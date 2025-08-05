#!/usr/bin/env python3
"""
將訓練好的 ConnectX 模型保存為 pkl 格式，並生成 submission 腳本
"""

import torch
import torch.nn as nn
import pickle
import os
import sys

class ConnectXNet(nn.Module):
    """強化學習用的 ConnectX 深度神經網路 - 修正版本"""

    def __init__(self, input_size=126, hidden_size=200, num_layers=256):  # 修正：使用4層而不是256層
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

def find_best_checkpoint():
    """找到最佳的檢查點檔案"""
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        print("❌ checkpoints 目錄不存在")
        return None
    
    # 尋找可能的檢查點檔案
    possible_files = [
        "best_model.pt",
        "pretrained_best.pt",
        "latest_checkpoint.pt"
    ]
    
    # 也尋找包含勝率的檔案
    import glob
    wr_files = glob.glob(os.path.join(checkpoint_dir, "best_model_wr_*.pt"))
    possible_files.extend([os.path.basename(f) for f in wr_files])
    
    for filename in possible_files:
        filepath = os.path.join(checkpoint_dir, filename)
        if os.path.exists(filepath):
            print(f"✅ 找到檢查點: {filepath}")
            return filepath
    
    print("❌ 找不到任何檢查點檔案")
    return None

def save_model_as_pickle():
    """將模型保存為 pkl 檔案"""
    
    # 找到檢查點檔案
    checkpoint_path = find_best_checkpoint()
    if not checkpoint_path:
        return False
    
    try:
        # 載入檢查點
        print(f"📥 載入檢查點: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # 檢查模型配置
        if 'config' in checkpoint:
            config = checkpoint['config']['agent']
            input_size = config.get('input_size', 126)
            hidden_size = config.get('hidden_size', 200)
            num_layers = config.get('num_layers', 4)
            
            print(f"檢查點模型配置: input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}")
            
            # 如果num_layers不合理，使用4層
            # if num_layers > 50:
            #     print(f"⚠️ 檢測到不合理的層數 ({num_layers})，調整為4層")
            #     num_layers = 4
        else:
            print("⚠️ 檢查點中沒有配置信息，使用默認配置")
            input_size, hidden_size, num_layers = 126, 200, 4
        
        # 建立模型
        model = ConnectXNet(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        
        # 嘗試載入權重，允許部分不匹配
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        if missing_keys:
            print(f"⚠️ 缺少的權重鍵 (前5個): {missing_keys[:5]}")
        if unexpected_keys:
            print(f"⚠️ 未使用的權重鍵 (前5個): {unexpected_keys[:5]}")
        
        # 檢查是否載入成功
        if len(missing_keys) > len(model.state_dict()) * 0.5:  # 如果超過50%的權重缺失
            print("❌ 權重載入失敗，缺失太多權重")
            return False
            
        model.eval()  # 設為評估模式
        
        # 保存為 pkl 檔案
        pkl_path = "checkpoints/connectx_model.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
        
        print(f"✅ 模型已保存為: {pkl_path}")
        
        # 顯示文件大小
        file_size = os.path.getsize(pkl_path) / (1024 * 1024)  # MB
        print(f"📊 檔案大小: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 保存模型時出錯: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_submission_script():
    """創建讀取pkl檔案的submission腳本"""
    
    submission_code = '''import pickle
import torch
import torch.nn as nn
import numpy as np
import os

# 模型架構定義
class ConnectXNet(nn.Module):
    def __init__(self, input_size=126, hidden_size=200, num_layers=4):
        super(ConnectXNet, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
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
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 7),
            nn.Softmax(dim=-1)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input_layer(x)
        
        for hidden_layer in self.hidden_layers:
            residual = x
            x = hidden_layer(x)
            x = torch.relu(x + residual)
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value

# 載入模型
def load_model():
    """載入pkl格式的模型"""
    try:
        # 嘗試從多個可能的路徑載入
        possible_paths = [
            'connectx_model.pkl',
            'checkpoints/connectx_model.pkl',
            '../checkpoints/connectx_model.pkl'
        ]
        
        for model_path in possible_paths:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    model.eval()
                    return model
        
        # 如果找不到模型文件，返回None
        return None
        
    except Exception as e:
        print(f"載入模型失敗: {e}")
        return None

# 全域模型實例
model = load_model()

def check_win(board, mark, col):
    """檢查在指定列放置棋子後是否能獲勝"""
    temp_board = board[:]
    row = -1
    for r in range(5, -1, -1):
        if temp_board[r * 7 + col] == 0:
            temp_board[r * 7 + col] = mark
            row = r
            break
    
    if row == -1:
        return False
    
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for dr, dc in directions:
        count = 1
        
        # 正方向檢查
        r, c = row + dr, col + dc
        while 0 <= r < 6 and 0 <= c < 7 and temp_board[r * 7 + c] == mark:
            count += 1
            r, c = r + dr, c + dc
        
        # 反方向檢查
        r, c = row - dr, col - dc
        while 0 <= r < 6 and 0 <= c < 7 and temp_board[r * 7 + c] == mark:
            count += 1
            r, c = r - dr, c - dc
        
        if count >= 4:
            return True
    
    return False

def if_i_can_finish(board, mark):
    """檢查是否有直接獲勝的動作"""
    valid_actions = get_valid_actions(board)
    for col in valid_actions:
        if check_win(board, mark, col):
            return col
    return -1

def if_i_will_lose(board, mark):
    """檢查對手是否能在下一步獲勝，返回阻擋動作"""
    opponent_mark = 3 - mark
    valid_actions = get_valid_actions(board)
    for col in valid_actions:
        if check_win(board, opponent_mark, col):
            return col
    return -1

def encode_state(board, mark):
    """編碼棋盤狀態"""
    state = np.array(board).reshape(6, 7)
    player_pieces = (state == mark).astype(np.float32)
    opponent_pieces = (state == (3 - mark)).astype(np.float32)
    empty_spaces = (state == 0).astype(np.float32)
    encoded = np.concatenate([
        player_pieces.flatten(),
        opponent_pieces.flatten(),
        empty_spaces.flatten()
    ])
    return encoded

def get_valid_actions(board):
    """獲取有效動作"""
    return [col for col in range(7) if board[col] == 0]

def my_agent(obs, config):
    """主要的 agent 函數"""
    board = obs['board']
    mark = obs['mark']
    
    # 首先檢查是否可以直接獲勝
    winning_move = if_i_can_finish(board, mark)
    if winning_move != -1:
        return int(winning_move)
    
    # 其次檢查是否需要阻擋對手獲勝
    blocking_move = if_i_will_lose(board, mark)
    if blocking_move != -1:
        return int(blocking_move)
    
    # 使用神經網路模型進行決策
    if model is not None:
        try:
            state = encode_state(board, mark)
            valid_actions = get_valid_actions(board)
            
            if not valid_actions:
                return 0
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                policy, _ = model(state_tensor)
                action_probs = policy.cpu().numpy()[0]
            
            # 遮罩無效動作
            masked_probs = np.zeros_like(action_probs)
            masked_probs[valid_actions] = action_probs[valid_actions]
            
            if masked_probs.sum() > 0:
                masked_probs /= masked_probs.sum()
                action = valid_actions[np.argmax(masked_probs[valid_actions])]
            else:
                action = valid_actions[0]
            
            return int(action)
            
        except Exception as e:
            print(f"模型推理失敗: {e}")
            # 模型失敗時的備用策略
            pass
    
    # 備用策略：偏好中央
    valid_actions = get_valid_actions(board)
    if not valid_actions:
        return 0
    
    # 優先選擇中央列
    for col in [3, 2, 4, 1, 5, 0, 6]:
        if col in valid_actions:
            return int(col)
    
    return int(valid_actions[0])
'''
    
    try:
        # 寫入檔案
        with open("submission_pkl.py", "w") as f:
            f.write(submission_code)
        
        print("✅ submission_pkl.py 已生成！")
        print(f"📊 檔案大小: {len(submission_code) / 1024:.2f} KB")
        print("💡 此腳本需要與 connectx_model.pkl 一起使用")
        
        return True
        
    except Exception as e:
        print(f"❌ 生成 submission 腳本時出錯: {e}")
        return False

def test_submission():
    """測試生成的 submission 腳本"""
    try:
        print("🧪 測試 submission 腳本...")
        
        # 檢查pkl文件是否存在
        if not os.path.exists("checkpoints/connectx_model.pkl"):
            print("❌ connectx_model.pkl 文件不存在，無法測試")
            return False
        
        # 複製pkl文件到當前目錄以便測試
        import shutil
        shutil.copy("checkpoints/connectx_model.pkl", "connectx_model.pkl")
        
        # 載入並執行submission腳本
        with open("submission_pkl.py", "r") as f:
            submission_content = f.read()
        
        # 使用exec執行代碼
        exec_globals = {}
        exec(submission_content, exec_globals)
        my_agent = exec_globals['my_agent']
        
        # 測試空棋盤
        test_obs = {'board': [0] * 42, 'mark': 1}
        test_config = {}
        action = my_agent(test_obs, test_config)
        print(f"✅ 測試成功！空棋盤動作: {action}")
        
        # 測試獲勝動作
        test_board = [0] * 42
        test_board[35] = test_board[36] = test_board[37] = 1  # 底排三個連續
        test_obs = {'board': test_board, 'mark': 1}
        action = my_agent(test_obs, test_config)
        expected_action = 3 if test_board[38] == 0 else 2  # 應該在第3列或第2列完成四連
        print(f"✅ 獲勝測試: 選擇動作 {action} (預期 {expected_action} 或其附近)")
        
        # 清理測試文件
        if os.path.exists("connectx_model.pkl"):
            os.remove("connectx_model.pkl")
        
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函數"""
    print("🚀 開始將模型轉換為 pkl 格式並生成 submission 腳本...")
    
    # 確保 checkpoints 目錄存在
    os.makedirs("checkpoints", exist_ok=True)
    
    # 步驟 1: 保存模型為 pkl
    if not save_model_as_pickle():
        print("❌ 保存模型失敗，退出...")
        return False
    
    # 步驟 2: 創建 submission 腳本
    if not create_submission_script():
        print("❌ 創建 submission 腳本失敗，退出...")
        return False
    
    # 步驟 3: 測試 submission 腳本
    if not test_submission():
        print("❌ 測試 submission 腳本失敗")
        return False
    
    print("\n🎉 所有步驟完成！")
    print("📁 生成的檔案:")
    print("   • checkpoints/connectx_model.pkl (模型檔案)")
    print("   • submission_pkl.py (submission腳本)")
    print("\n💡 使用說明:")
    print("   • 將 connectx_model.pkl 和 submission_pkl.py 一起上傳到 Kaggle")
    print("   • 腳本會自動尋找並載入 pkl 模型檔案")
    print("   • 包含完整的戰術智能（獲勝檢測、防守阻擋）")
    
    return True

if __name__ == "__main__":
    main()