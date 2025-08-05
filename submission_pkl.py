import pickle
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
