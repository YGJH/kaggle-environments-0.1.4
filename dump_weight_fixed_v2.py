#!/usr/bin/env python3
import torch
import numpy as np
import base64
import pickle

# 載入模型
checkpoint = torch.load("checkpoints/checkpoint_episode_50000.pt", map_location="cpu")
state_dict = checkpoint['model_state_dict']

# 存成 numpy
np.savez_compressed("model_weights.npz", **{k: v.numpy() for k, v in state_dict.items()})

# 使用 Base64 編碼壓縮權重
with open("model_weights.npz", "rb") as f:
    weights_bytes = f.read()
    weights_b64 = base64.b64encode(weights_bytes).decode('utf-8')

# Create submission.py with embedded weights and corrected model structure
submission_code = f'''import numpy as np
import base64
import io
WEIGHTS_B64 = "{weights_b64}"
def load_weights():
    weights_bytes = base64.b64decode(WEIGHTS_B64)
    weights_buffer = io.BytesIO(weights_bytes)
    data = np.load(weights_buffer)
    return {{k: data[k] for k in data.files}}
weights = load_weights()
def check_win(board, mark, col):
    """檢查在指定列放置棋子後是否能獲勝"""
    # 模擬放置棋子
    temp_board = board[:]
    row = -1
    for r in range(5, -1, -1):  # 從下往上找空位
        if temp_board[r * 7 + col] == 0:
            temp_board[r * 7 + col] = mark
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
    """檢查對手是否能在下一步獲勝，如果是則返回阻擋的動作"""
    opponent_mark = 3 - mark  # 對手標記
    valid_actions = get_valid_actions(board)
    for col in valid_actions:
        if check_win(board, opponent_mark, col):
            return col  # 返回需要阻擋的列
    return -1
def forward_pass(state):
    x = np.array(state, dtype=np.float32)
    
    # 輸入層: input_layer.0 = Linear, input_layer.1 = ReLU, input_layer.2 = Dropout
    x = np.dot(x, weights['input_layer.0.weight'].T) + weights['input_layer.0.bias']
    x = np.maximum(0, x)  # ReLU
    
    # 隱藏層 (使用 LayerNorm)
    num_layers = 256  # 根據真實的 num_layers
    for block_idx in range(min(num_layers, 256)):  # 限制層數避免無限循環
        residual = x.copy()
        
        # 第一個 Linear + LayerNorm + ReLU + Dropout
        w1_key = f'hidden_layers.{{block_idx}}.0.weight'
        b1_key = f'hidden_layers.{{block_idx}}.0.bias'
        ln1_weight_key = f'hidden_layers.{{block_idx}}.1.weight'
        ln1_bias_key = f'hidden_layers.{{block_idx}}.1.bias'
        
        if w1_key in weights and b1_key in weights:
            x = np.dot(x, weights[w1_key].T) + weights[b1_key]
            
            # LayerNorm (不是 BatchNorm)
            if ln1_weight_key in weights and ln1_bias_key in weights:
                # LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
                mean = np.mean(x, axis=-1, keepdims=True)
                var = np.var(x, axis=-1, keepdims=True)
                x = (x - mean) / np.sqrt(var + 1e-5)
                x = x * weights[ln1_weight_key] + weights[ln1_bias_key]
            
            x = np.maximum(0, x)  # ReLU
            
            # 第二個 Linear + LayerNorm
            w2_key = f'hidden_layers.{{block_idx}}.4.weight'
            b2_key = f'hidden_layers.{{block_idx}}.4.bias'
            ln2_weight_key = f'hidden_layers.{{block_idx}}.5.weight'
            ln2_bias_key = f'hidden_layers.{{block_idx}}.5.bias'
            
            if w2_key in weights and b2_key in weights:
                x = np.dot(x, weights[w2_key].T) + weights[b2_key]
                
                # LayerNorm
                if ln2_weight_key in weights and ln2_bias_key in weights:
                    mean = np.mean(x, axis=-1, keepdims=True)
                    var = np.var(x, axis=-1, keepdims=True)
                    x = (x - mean) / np.sqrt(var + 1e-5)
                    x = x * weights[ln2_weight_key] + weights[ln2_bias_key]
                
                # 殘差連接 + ReLU
                x = np.maximum(0, x + residual)
        else:
            # 如果沒有這一層的權重，跳出循環
            break
    
    # 策略頭: Linear + LayerNorm + ReLU + Dropout + Linear + Softmax
    # policy_head.0 = Linear, policy_head.1 = LayerNorm, policy_head.2 = ReLU, 
    # policy_head.3 = Dropout, policy_head.4 = Linear, policy_head.5 = Softmax
    policy = np.dot(x, weights['policy_head.0.weight'].T) + weights['policy_head.0.bias']
    
    # LayerNorm
    if 'policy_head.1.weight' in weights and 'policy_head.1.bias' in weights:
        mean = np.mean(policy, axis=-1, keepdims=True)
        var = np.var(policy, axis=-1, keepdims=True)
        policy = (policy - mean) / np.sqrt(var + 1e-5)
        policy = policy * weights['policy_head.1.weight'] + weights['policy_head.1.bias']
    
    policy = np.maximum(0, policy)  # ReLU
    
    # 最後的 Linear 層 (policy_head.4)
    if 'policy_head.4.weight' in weights and 'policy_head.4.bias' in weights:
        policy = np.dot(policy, weights['policy_head.4.weight'].T) + weights['policy_head.4.bias']
    
    # Softmax
    policy_exp = np.exp(policy - np.max(policy))
    policy = policy_exp / np.sum(policy_exp)
    
    return policy
def encode_state(board, mark):
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
    return [col for col in range(7) if board[col] == 0]
def my_agent(obs, config):
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
    
    # 如果既不能獲勝也不需要阻擋，則使用模型進行決策
    state = encode_state(board, mark)
    valid_actions = get_valid_actions(board)
    
    if not valid_actions:
        return 0
    
    action_probs = forward_pass(state)
    
    # 遮罩無效動作
    masked_probs = np.zeros_like(action_probs)
    masked_probs[valid_actions] = action_probs[valid_actions]
    
    if masked_probs.sum() > 0:
        masked_probs /= masked_probs.sum()
        action = valid_actions[np.argmax(masked_probs[valid_actions])]
    else:
        action = valid_actions[0]
    
    return int(action)
'''

# Write submission.py
print("正在寫入 submission.py...")
with open('submission.py', 'w') as f:
    f.write(submission_code)

print("\n✅ 生成 submission.py 成功！")
print("📊 文件統計:")
with open('submission.py', 'r') as f:
    content = f.read()
    lines = content.count('\n')
    chars = len(content)
    print(f"  - 行數: {lines:,}")
    print(f"  - 字符數: {chars:,}")
    print(f"  - 大小: {chars / 1024 / 1024:.2f} MB")

print("\n💡 提示:")
print("   ✓ submission.py 已經包含了所有必要的權重數據")
print("   ✓ 使用 Base64 編碼壓縮，無需額外文件")
print("   ✓ 模型結構已更新匹配新的 hidden_layers 結構")
print("   ✓ 可以直接提交到 Kaggle 競賽！")

# Test the generated submission
print("\n🧪 測試生成的 submission.py...")
try:
    # Load and execute the submission code
    with open('submission.py', 'r') as f:
        submission_content = f.read()
    
    exec(submission_content)
    
    # Test with a simple observation
    test_obs = {
        'board': [0] * 42,
        'mark': 1
    }
    test_config = {}
    action = my_agent(test_obs, test_config)
    print(f"✅ 測試成功！測試動作: {action}")
    print(f"   動作類型: {type(action)}")
    
    # Test with different mark
    test_obs['mark'] = 2
    action2 = my_agent(test_obs, test_config)
    print(f"✅ 第二次測試成功！測試動作: {action2}")
    
except Exception as e:
    print(f"❌ 測試失敗: {e}")
    import traceback
    traceback.print_exc()
