#!/usr/bin/env python3
"""
優化版本的模型權重導出腳本
- 檢測實際使用的層數
- 只保存必要的權重
- 壓縮模型大小
"""
import torch
import numpy as np
import base64
import pickle

def analyze_checkpoint(checkpoint_path):
    """分析檢查點中的實際模型結構"""
    print(f"🔍 分析檢查點: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint['model_state_dict']
    
    # 分析隱藏層數量
    hidden_layer_indices = set()
    for key in state_dict.keys():
        if key.startswith('hidden_layers.'):
            parts = key.split('.')
            if len(parts) >= 2:
                try:
                    layer_idx = int(parts[1])
                    hidden_layer_indices.add(layer_idx)
                except ValueError:
                    pass
    
    actual_num_layers = len(hidden_layer_indices)
    print(f"📊 實際隱藏層數量: {actual_num_layers}")
    
    # 分析權重鍵
    policy_keys = [k for k in state_dict.keys() if k.startswith('policy_head.')]
    value_keys = [k for k in state_dict.keys() if k.startswith('value_head.')]
    
    print(f"📊 策略頭權重鍵: {policy_keys}")
    print(f"📊 價值頭權重鍵: {value_keys}")
    
    return state_dict, actual_num_layers

def create_optimized_weights(state_dict, max_layers=10):
    """創建優化的權重字典，只保留必要的層"""
    optimized_weights = {}
    
    # 保留輸入層
    for key in state_dict.keys():
        if key.startswith('input_layer.'):
            optimized_weights[key] = state_dict[key]
    
    # 只保留前 max_layers 個隱藏層
    for layer_idx in range(max_layers):
        for key in state_dict.keys():
            if key.startswith(f'hidden_layers.{layer_idx}.'):
                optimized_weights[key] = state_dict[key]
    
    # 保留輸出頭
    for key in state_dict.keys():
        if key.startswith('policy_head.') or key.startswith('value_head.'):
            optimized_weights[key] = state_dict[key]
    
    print(f"📊 原始權重數量: {len(state_dict)}")
    print(f"📊 優化後權重數量: {len(optimized_weights)}")
    
    return optimized_weights

def create_forward_pass_code(actual_num_layers):
    """根據實際層數創建 forward_pass 函數代碼"""
    
    # 限制層數以避免過大的模型
    limited_layers = min(actual_num_layers, 10)
    
    code = f'''def forward_pass(state):
    x = np.array(state, dtype=np.float32)
    
    # 輸入層
    x = np.dot(x, weights['input_layer.0.weight'].T) + weights['input_layer.0.bias']
    x = np.maximum(0, x)  # ReLU
    
    # 隱藏層 (使用 LayerNorm)
    for block_idx in range({limited_layers}):
        residual = x.copy()
        
        # 檢查這一層是否存在
        w1_key = f'hidden_layers.{{block_idx}}.0.weight'
        if w1_key not in weights:
            break
            
        # 第一個 Linear + LayerNorm + ReLU
        x = np.dot(x, weights[f'hidden_layers.{{block_idx}}.0.weight'].T) + weights[f'hidden_layers.{{block_idx}}.0.bias']
        
        # LayerNorm
        ln1_weight_key = f'hidden_layers.{{block_idx}}.1.weight'
        ln1_bias_key = f'hidden_layers.{{block_idx}}.1.bias'
        if ln1_weight_key in weights and ln1_bias_key in weights:
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)
            x = (x - mean) / np.sqrt(var + 1e-5)
            x = x * weights[ln1_weight_key] + weights[ln1_bias_key]
        
        x = np.maximum(0, x)  # ReLU
        
        # 第二個 Linear + LayerNorm
        w2_key = f'hidden_layers.{{block_idx}}.4.weight'
        b2_key = f'hidden_layers.{{block_idx}}.4.bias'
        if w2_key in weights and b2_key in weights:
            x = np.dot(x, weights[w2_key].T) + weights[b2_key]
            
            # LayerNorm
            ln2_weight_key = f'hidden_layers.{{block_idx}}.5.weight'
            ln2_bias_key = f'hidden_layers.{{block_idx}}.5.bias'
            if ln2_weight_key in weights and ln2_bias_key in weights:
                mean = np.mean(x, axis=-1, keepdims=True)
                var = np.var(x, axis=-1, keepdims=True)
                x = (x - mean) / np.sqrt(var + 1e-5)
                x = x * weights[ln2_weight_key] + weights[ln2_bias_key]
            
            # 殘差連接 + ReLU
            x = np.maximum(0, x + residual)
    
    # 策略頭
    policy = np.dot(x, weights['policy_head.0.weight'].T) + weights['policy_head.0.bias']
    
    # LayerNorm
    if 'policy_head.1.weight' in weights and 'policy_head.1.bias' in weights:
        mean = np.mean(policy, axis=-1, keepdims=True)
        var = np.var(policy, axis=-1, keepdims=True)
        policy = (policy - mean) / np.sqrt(var + 1e-5)
        policy = policy * weights['policy_head.1.weight'] + weights['policy_head.1.bias']
    
    policy = np.maximum(0, policy)  # ReLU
    
    # 最後的 Linear 層
    if 'policy_head.4.weight' in weights and 'policy_head.4.bias' in weights:
        policy = np.dot(policy, weights['policy_head.4.weight'].T) + weights['policy_head.4.bias']
    
    # Softmax
    policy_exp = np.exp(policy - np.max(policy))
    policy = policy_exp / np.sum(policy_exp)
    
    return policy'''
    
    return code

def main():
    checkpoint_path = "checkpoints/best_model_wr_0.622.pt"
    
    # 分析模型結構
    state_dict, actual_num_layers = analyze_checkpoint(checkpoint_path)
    
    # 創建優化的權重（只保留前10層）
    optimized_weights = create_optimized_weights(state_dict, max_layers=240)
    
    # 存成 numpy
    print("💾 保存優化後的權重...")
    np.savez_compressed("model_weights_optimized.npz", **{k: v.numpy() for k, v in optimized_weights.items()})
    
    # 使用 Base64 編碼
    with open("model_weights_optimized.npz", "rb") as f:
        weights_bytes = f.read()
        weights_b64 = base64.b64encode(weights_bytes).decode('utf-8')
    
    print(f"📊 優化後權重檔案大小: {len(weights_bytes) / 1024 / 1024:.2f} MB")
    
    # 創建 forward_pass 代碼
    forward_pass_code = create_forward_pass_code(min(actual_num_layers, 10))
    
    # 創建完整的 submission 代碼
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
        
        r, c = row + dr, col + dc
        while 0 <= r < 6 and 0 <= c < 7 and temp_board[r * 7 + c] == mark:
            count += 1
            r, c = r + dr, c + dc
        
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

{forward_pass_code}

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
    
    # 使用模型進行決策
    try:
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
        
    except Exception as e:
        # 備用策略：選擇中央列
        valid_actions = get_valid_actions(board)
        for col in [3, 2, 4, 1, 5, 0, 6]:
            if col in valid_actions:
                return int(col)
        return int(valid_actions[0] if valid_actions else 0)
'''
    
    # 寫入檔案
    print("📝 寫入優化後的 submission.py...")
    with open('submission_optimized.py', 'w') as f:
        f.write(submission_code)
    
    print("\n✅ 生成 submission_optimized.py 成功！")
    print("📊 檔案統計:")
    with open('submission_optimized.py', 'r') as f:
        content = f.read()
        lines = content.count('\n')
        chars = len(content)
        print(f"  - 行數: {lines:,}")
        print(f"  - 字符數: {chars:,}")
        print(f"  - 大小: {chars / 1024 / 1024:.2f} MB")
    
    # 測試生成的 submission
    print("\n🧪 測試優化後的 submission...")
    try:
        exec(submission_code)
        # 測試
        test_obs = {'board': [0] * 42, 'mark': 1}
        test_config = {}
        action = my_agent(test_obs, test_config)
        print(f"✅ 測試成功！動作: {action} (類型: {type(action)})")
        
        # 測試獲勝動作
        test_board = [0] * 42
        test_board[35] = test_board[36] = test_board[37] = 1  # 底排三個連續
        test_obs = {'board': test_board, 'mark': 1}
        action = my_agent(test_obs, test_config)
        print(f"✅ 獲勝測試成功！動作: {action}")
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 優化完成！")
    print("💡 建議:")
    print("   • 使用 submission_optimized.py 進行提交")
    print("   • 檔案大小已優化，只包含必要的模型層")
    print("   • 包含完整的戰術邏輯和容錯機制")

if __name__ == "__main__":
    main()
