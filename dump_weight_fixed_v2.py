#!/usr/bin/env python3
import os
import glob
import sys
import re
import torch
import numpy as np
import base64
from datetime import datetime

PREFERRED_CKPT = os.getenv("CKPT_PATH", "checkpoints/best_model_wr_0.212.pt")


def try_load_checkpoint(path: str):
    try:
        ckpt = torch.load(path, map_location="cpu")
        print(f"Loaded checkpoint: {path}")
        return ckpt
    except Exception as e:
        print(f"⚠️  無法載入 {path}: {e}")
        return None


def find_working_checkpoint():
    # 先嘗試使用偏好路徑
    print(PREFERRED_CKPT)
    if os.path.isfile(PREFERRED_CKPT):
        ckpt = try_load_checkpoint(PREFERRED_CKPT)
        if ckpt is not None:
            return PREFERRED_CKPT, ckpt

    # 否則從 checkpoints 依修改時間由新到舊嘗試
    candidates = sorted(glob.glob(os.path.join("checkpoints", "*.pt")), key=os.path.getmtime, reverse=True)
    for p in candidates:
        ckpt = try_load_checkpoint(p)
        if ckpt is not None:
            return p, ckpt
    return None, None


def extract_state_dict(ckpt) -> dict:
    # 支援兩種格式：包含 model_state_dict 的完整檔；或直接是 state_dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        # 可能就是 state_dict 本身
        return ckpt
    else:
        raise ValueError("Unsupported checkpoint format: not a dict")


FINAL_RE = re.compile(r"^final_(\d{8})_(\d{6})\.pt$")
TS_TAIL_RE = re.compile(r"_(\d{8})_(\d{6})\.pt$")


def parse_ts_from_name(fname: str):
    m = FINAL_RE.match(fname)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except Exception:
            return None
    m = TS_TAIL_RE.search(fname)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except Exception:
            return None
    return None


def choose_latest_checkpoint_by_name(files):
    latest = None
    latest_time = None
    for p in files:
        fname = os.path.basename(p)
        ts = parse_ts_from_name(fname)
        try:
            t = ts.timestamp() if ts is not None else os.path.getmtime(p)
        except OSError:
            t = 0
        if latest is None or t > latest_time:
            latest = p
            latest_time = t
    return latest


def generate_submission_code(weights_b64, has_trunk):
    """生成 submission.py 代碼"""
    
    if has_trunk:
        # 高級架構
        code = '''import numpy as np
import base64, io
# Advanced architecture: BottleneckSE blocks + optional SpatialSelfAttention + coord embeddings
WEIGHTS_B64 = "''' + weights_b64 + '''"

def load_weights():
    buf = io.BytesIO(base64.b64decode(WEIGHTS_B64))
    data = np.load(buf)
    return {k: data[k] for k in data.files}
weights = load_weights()

# ---------------- Tactical helpers ----------------

def get_valid_actions(board):
    return [c for c in range(7) if board[c] == 0]

def drop_row(board, col):
    for r in range(5, -1, -1):
        if board[r*7+col] == 0:
            return r
    return -1

def check_win_after(board, mark, col):
    row = drop_row(board, col)
    if row < 0:
        return False
    b = board[:]
    b[row*7+col] = mark
    dirs = [(0,1),(1,0),(1,1),(1,-1)]
    for dr, dc in dirs:
        cnt = 1
        for s in (1,-1):
            rr, cc = row+dr*s, col+dc*s
            while 0 <= rr < 6 and 0 <= cc < 7 and b[rr*7+cc] == mark:
                cnt += 1
                rr += dr*s; cc += dc*s
        if cnt >= 4:
            return True
    return False

def immediate_win(board, mark):
    for c in get_valid_actions(board):
        if check_win_after(board, mark, c):
            return c
    return -1

def immediate_block(board, mark):
    opp = 3-mark
    for c in get_valid_actions(board):
        if check_win_after(board, opp, c):
            return c
    return -1

def gives_opp_win(board, move_col, mark):
    opp = 3-mark
    row = drop_row(board, move_col)
    if row < 0:
        return True
    b = board[:]
    b[row*7+move_col] = mark
    for c in get_valid_actions(b):
        if check_win_after(b, opp, c):
            return True
    return False

def safe_moves(board, mark):
    valids = get_valid_actions(board)
    return [c for c in valids if not gives_opp_win(board, c, mark)]

# ---------------- Numpy NN primitives ----------------

def relu(x):
    return np.maximum(0, x)

def conv2d(x, w, b=None, pad=0):
    # x: (C_in,H,W) w: (C_out,C_in,k,k)
    C_out, C_in, K, _ = w.shape
    H,W = x.shape[1], x.shape[2]
    xp = np.pad(x, ((0,0),(pad,pad),(pad,pad)), mode='constant') if pad>0 else x
    outH = H; outW = W
    cols = []
    for i in range(K):
        for j in range(K):
            cols.append(xp[:, i:i+outH, j:j+outW].reshape(C_in,-1))
    col = np.concatenate(cols, axis=0)  # (C_in*K*K, H*W)
    wcol = w.reshape(C_out, -1)
    y = wcol @ col
    if b is not None:
        y += b.reshape(-1,1)
    return y.reshape(C_out,outH,outW)

def conv1x1(x, w, b=None):
    # treat as conv2d with K=1
    C_out, C_in, _, _ = w.shape
    H,W = x.shape[1], x.shape[2]
    x_flat = x.reshape(C_in, -1)
    w_flat = w.reshape(C_out, C_in)
    y = w_flat @ x_flat
    if b is not None:
        y += b.reshape(-1,1)
    return y.reshape(C_out,H,W)

def group_norm(x, gamma, beta, num_groups=8, eps=1e-5):
    C,H,W = x.shape
    G = num_groups
    gs = C//G
    xr = x.reshape(G, gs, H, W)
    mean = xr.mean(axis=(1,2,3), keepdims=True)
    var = xr.var(axis=(1,2,3), keepdims=True)
    xn = (xr - mean)/np.sqrt(var+eps)
    xn = xn.reshape(C,H,W)
    return xn*gamma.reshape(C,1,1)+beta.reshape(C,1,1)

def linear(x, w, b):
    return x @ w.T + b

# Enumerate trunk blocks
_trunk_indices = sorted(set(int(k.split('.')[1]) for k in weights if k.startswith('trunk.') ))
_is_attn = {}
for idx in _trunk_indices:
    _is_attn[idx] = 'trunk.' + str(idx) + '.qkv.weight' in weights

# Infer channels / hidden / heads
stem_w = weights['stem.0.weight']
C = stem_w.shape[0]
hidden = weights['head.1.weight'].shape[0]
half_hidden = hidden//2

# Dynamically infer attention heads from the first attention block
attention_heads = 24  # Default to our new configuration
if any(_is_attn.values()):
    # Find first attention block to infer heads
    first_attn_idx = min([idx for idx in _trunk_indices if _is_attn[idx]])
    qkv_key = 'trunk.' + str(first_attn_idx) + '.qkv.weight'
    if qkv_key in weights:
        qkv_weight = weights[qkv_key]
        # qkv produces 3*C channels, so each of q,k,v has C channels
        # If C is divisible by common head counts, use the largest reasonable one
        possible_heads = [h for h in [24, 12, 8, 6, 4, 3, 2, 1] if C % h == 0]
        if possible_heads:
            attention_heads = possible_heads[0]  # Use the largest possible

# Precompute coord planes (-1..1)
_row = np.tile(np.linspace(-1,1,6).reshape(6,1), (1,7))
_col = np.tile(np.linspace(-1,1,7), (6,1))

def forward_pass(state):
    # state: 126 -> (3,6,7), add coord (2,6,7)
    x = np.array(state, dtype=np.float32).reshape(3,6,7)
    coord = np.stack([_row, _col], axis=0).astype(np.float32)
    x = np.concatenate([x, coord], axis=0)
    # stem
    x = conv2d(x, weights['stem.0.weight'], None, pad=1)
    x = group_norm(x, weights['stem.1.weight'], weights['stem.1.bias'], num_groups=8)
    x = relu(x)
    # trunk
    for idx in _trunk_indices:
        if _is_attn[idx]:
            # Spatial self-attention block: qkv 1x1 conv produce 3C
            qkv_key = 'trunk.' + str(idx) + '.qkv.weight'
            proj_key = 'trunk.' + str(idx) + '.proj.weight'
            qkv = conv1x1(x, weights[qkv_key], None)
            Bq = qkv.shape[0]//3
            q, k, v = np.split(qkv, 3, axis=0)
            # Use dynamically inferred heads
            heads = attention_heads
            dim = q.shape[0]//heads
            HW = q.shape[1]*q.shape[2]
            def reshape_heads(t):
                return t.reshape(heads, dim, HW)
            qh, kh, vh = map(reshape_heads, (q,k,v))
            scale = dim ** -0.5
            out_heads = []
            for h in range(heads):
                attn = (qh[h].T @ kh[h]) * scale  # (HW,HW)
                attn = np.exp(attn - attn.max(axis=1, keepdims=True))
                attn /= attn.sum(axis=1, keepdims=True)+1e-8
                out = attn @ vh[h].T  # (HW,dim)
                out_heads.append(out.T)
            out = np.concatenate(out_heads, axis=0).reshape(C, q.shape[1], q.shape[2])
            out = conv1x1(out, weights[proj_key], None)
            x = out  # no residual in original implementation
        else:
            # BottleneckSE
            prefix = 'trunk.' + str(idx) + '.'
            c1 = conv1x1(x, weights[prefix + 'conv1.weight'], None)
            c1 = group_norm(c1, weights[prefix + 'gn1.weight'], weights[prefix + 'gn1.bias'])
            c1 = relu(c1)
            c2 = conv2d(c1, weights[prefix + 'conv2.weight'], None, pad=1)
            c2 = group_norm(c2, weights[prefix + 'gn2.weight'], weights[prefix + 'gn2.bias'])
            c2 = relu(c2)
            c3 = conv1x1(c2, weights[prefix + 'conv3.weight'], None)
            c3 = group_norm(c3, weights[prefix + 'gn3.weight'], weights[prefix + 'gn3.bias'])
            # SE
            se_vec = c3.mean(axis=(1,2))  # (C,)
            # fc1
            w1 = weights[prefix + 'se_fc1.weight']; b1 = weights[prefix + 'se_fc1.bias']
            w2 = weights[prefix + 'se_fc2.weight']; b2 = weights[prefix + 'se_fc2.bias']
            se_h = relu(se_vec @ w1.T + b1)
            se_s = 1/(1+np.exp(-(se_h @ w2.T + b2)))
            c3 = c3 * se_s.reshape(-1,1,1)
            # DropPath ignored (eval)
            x = relu(x + c3)
    # head
    flat = x.reshape(-1)
    h = relu(linear(flat, weights['head.1.weight'], weights['head.1.bias']))
    # policy head
    ph1 = relu(linear(h, weights['policy_head.0.weight'], weights['policy_head.0.bias']))
    logits = linear(ph1, weights['policy_head.2.weight'], weights['policy_head.2.bias'])
    # softmax
    exp = np.exp(logits - np.max(logits))
    probs = exp / (exp.sum() + 1e-8)
    return probs

# -------------- State encoding (unchanged 3 planes) --------------

def detect_player_identity(board, mark):
    c1 = sum(1 for v in board if v == 1)
    c2 = sum(1 for v in board if v == 2)
    if c1 == c2:
        return 1
    if c1 == c2 + 1:
        return 2
    return mark

def encode_state(board, mark):
    cur = detect_player_identity(board, mark)
    arr = np.array(board).reshape(6,7)
    p = (arr == cur).astype(np.float32)
    o = (arr == (3-cur)).astype(np.float32)
    e = (arr == 0).astype(np.float32)
    return np.concatenate([p.ravel(), o.ravel(), e.ravel()])

# -------------- Safe action checker --------------

def find_safe_action(board, mark, preferred_action, max_attempts=7):
    """
    檢查動作是否會讓對方下一步就勝利，如果會就隨機挑選其他動作
    最多嘗試7次，如果都不安全就返回原動作
    """
    import random
    
    def will_opponent_win_next(board, action, mark):
        # 模擬我方下這步棋
        row = drop_row(board, action)
        if row < 0:
            return True  # 無效動作視為危險
        
        # 複製棋盤並下棋
        temp_board = board[:]
        temp_board[row * 7 + action] = mark
        
        # 檢查對方是否有即時獲勝機會
        opponent_mark = 3 - mark
        opponent_win = immediate_win(temp_board, opponent_mark)
        return opponent_win != -1
    
    valids = get_valid_actions(board)
    if not valids:
        return 0
    
    # 如果首選動作是安全的，直接返回
    if preferred_action in valids and not will_opponent_win_next(board, preferred_action, mark):
        return preferred_action
    
    # 否則嘗試找到安全動作
    attempted = set()
    for _ in range(max_attempts):
        # 隨機選擇一個還沒嘗試過的動作
        available = [a for a in valids if a not in attempted]
        if not available:
            break
            
        action = random.choice(available)
        attempted.add(action)
        
        if not will_opponent_win_next(board, action, mark):
            return action
    
    # 如果7次都找不到安全動作，返回首選動作或隨機動作
    if preferred_action in valids:
        return preferred_action
    return random.choice(valids) if valids else 0

# -------------- Agent main --------------

def agent(obs, config):
    board = obs['board']; mark = obs['mark']
    valids = get_valid_actions(board)
    if not valids:
        return 0
    s = encode_state(board, mark)
    probs = forward_pass(s)
    # Mask invalid
    mask = np.zeros_like(probs)
    mask[valids] = probs[valids]
    if mask.sum() <= 0:
        safe = safe_moves(board, mark)
        if safe:
            preferred = int(np.random.choice(safe))
        else:
            preferred = int(valids[0])
    else:
        mask /= mask.sum()
        # Prefer safe moves with highest prob
        s_moves = safe_moves(board, mark)
        if s_moves:
            preferred = max(s_moves, key=lambda c: mask[c])
        else:
            preferred = max(valids, key=lambda c: mask[c])
    
    # 使用安全動作檢查器
    final_action = find_safe_action(board, mark, preferred)
    return int(final_action)
'''
    else:
        # 簡單/通用架構：同時支援我們的舊簡單MLP命名與 SB3 MlpPolicy 命名
        code = '''import numpy as np
import base64, io
# Simple CNN architecture
WEIGHTS_B64 = "''' + weights_b64 + '''"

def load_weights():
    buf = io.BytesIO(base64.b64decode(WEIGHTS_B64))
    data = np.load(buf)
    return {k: data[k] for k in data.files}
weights = load_weights()

# 檢測是否為 Stable-Baselines3 的 MlpPolicy 權重命名（更穩健）
_HAS_POLICY_NET = any(k.startswith('mlp_extractor.policy_net.') and k.endswith('.weight') for k in weights.keys())
_HAS_ACTION_HEAD = any(k.startswith('action_net') and (k.endswith('.weight') or k.endswith('.bias')) for k in weights.keys())
_IS_SB3 = _HAS_POLICY_NET and _HAS_ACTION_HEAD

# 戰術輔助函數（與高級版本相同）
def get_valid_actions(board):
    return [c for c in range(7) if board[c] == 0]

def drop_row(board, col):
    for r in range(5, -1, -1):
        if board[r*7+col] == 0:
            return r
    return -1

def check_win_after(board, mark, col):
    row = drop_row(board, col)
    if row < 0:
        return False
    b = board[:]
    b[row*7+col] = mark
    dirs = [(0,1),(1,0),(1,1),(1,-1)]
    for dr, dc in dirs:
        cnt = 1
        for s in (1,-1):
            rr, cc = row+dr*s, col+dc*s
            while 0 <= rr < 6 and 0 <= cc < 7 and b[rr*7+cc] == mark:
                cnt += 1
                rr += dr*s; cc += dc*s
        if cnt >= 4:
            return True
    return False

def immediate_win(board, mark):
    for c in get_valid_actions(board):
        if check_win_after(board, mark, c):
            return c
    return -1

def immediate_block(board, mark):
    opp = 3-mark
    for c in get_valid_actions(board):
        if check_win_after(board, opp, c):
            return c
    return -1

def gives_opp_win(board, move_col, mark):
    opp = 3-mark
    row = drop_row(board, move_col)
    if row < 0:
        return True
    b = board[:]
    b[row*7+move_col] = mark
    for c in get_valid_actions(b):
        if check_win_after(b, opp, c):
            return True
    return False

def safe_moves(board, mark):
    valids = get_valid_actions(board)
    return [c for c in valids if not gives_opp_win(board, c, mark)]

# 簡單神經網路函數
def relu(x):
    return np.maximum(0, x)

def linear(x, w, b):
    return x @ w.T + b

def encode_state(board, mark):
    """狀態編碼：
    - 若為 SB3 MlpPolicy 權重，輸入為原始 42 維棋盤（與訓練一致）。
    - 否則回退到舊簡單 3 平面編碼（126 維）。
    """
    if _IS_SB3:
        return np.array(board, dtype=np.float32)
    # fallback: 三平面
    arr = np.array(board).reshape(6,7)
    # 由於無法得知訓練時的玩家標記約定，這裡沿用原簡單實作：用當前玩家/對手/空格三平面
    p = (arr == mark).astype(np.float32)
    o = (arr == (3-mark)).astype(np.float32)
    e = (arr == 0).astype(np.float32)
    return np.concatenate([p.ravel(), o.ravel(), e.ravel()])

def forward_pass(state):
    # 通用 MLP 前向傳播
    x = np.array(state, dtype=np.float32)
    if _IS_SB3:
        # 依序套用 policy_net.* 層（線性 + ReLU），然後 action_net 輸出
        # key 格式: 'mlp_extractor.policy_net.{idx}.weight'
        idxs = []
        for k in weights.keys():
            if k.startswith('mlp_extractor.policy_net.') and k.endswith('.weight'):
                parts = k.split('.')
                # parts: ['mlp_extractor','policy_net','{idx}','weight']
                if len(parts) >= 4:
                    try:
                        idxs.append(int(parts[2]))
                    except Exception:
                        pass
        idxs = sorted(set(idxs))
        for i in idxs:
            w = weights[f'mlp_extractor.policy_net.{i}.weight']
            b_key = f'mlp_extractor.policy_net.{i}.bias'
            b = weights[b_key] if b_key in weights else np.zeros(w.shape[0], dtype=np.float32)
            x = relu(linear(x, w, b))
        # 輸出層（動作 logits）
        # 某些版本可能使用 'action_net.0.weight' 形式；做回退搜尋
        if 'action_net.weight' in weights:
            w_out = weights['action_net.weight']
            b_out = weights['action_net.bias'] if 'action_net.bias' in weights else np.zeros(w_out.shape[0], dtype=np.float32)
        else:
            # 找到第一個 action_net.*.weight
            cand_w = sorted([k for k in weights.keys() if k.startswith('action_net') and k.endswith('.weight')])
            if not cand_w:
                raise KeyError('action head weight not found')
            w_key = cand_w[0]
            b_key = w_key[:-6] + 'bias'  # replace '.weight' -> '.bias'
            w_out = weights[w_key]
            b_out = weights[b_key] if b_key in weights else np.zeros(w_out.shape[0], dtype=np.float32)
        logits = linear(x, w_out, b_out)
    else:
        # 舊簡單命名：hidden*.{weight,bias} + output.{weight,bias}
        hidden_layers = sorted([k for k in weights.keys() if k.startswith('hidden') and k.endswith('.weight')])
        # 只取層名，不含後綴
        hidden_bases = [h[:-7] for h in hidden_layers]  # 去掉 '.weight'
        for base in hidden_bases:
            w = weights[base + '.weight']
            b_key = base + '.bias'
            b = weights[b_key] if b_key in weights else np.zeros(w.shape[0], dtype=np.float32)
            x = relu(linear(x, w, b))
        w_out = weights['output.weight']
        b_out = weights['output.bias'] if 'output.bias' in weights else np.zeros(w_out.shape[0], dtype=np.float32)
        logits = linear(x, w_out, b_out)
    
    # Softmax
    exp = np.exp(logits - np.max(logits))
    probs = exp / (exp.sum() + 1e-8)
    return probs

# -------------- Safe action checker --------------

def find_safe_action(board, mark, preferred_action, max_attempts=7):
    """
    檢查動作是否會讓對方下一步就勝利，如果會就隨機挑選其他動作
    最多嘗試7次，如果都不安全就返回原動作
    """
    import random
    
    def will_opponent_win_next(board, action, mark):
        # 模擬我方下這步棋
        row = drop_row(board, action)
        if row < 0:
            return True  # 無效動作視為危險
        
        # 複製棋盤並下棋
        temp_board = board[:]
        temp_board[row * 7 + action] = mark
        
        # 檢查對方是否有即時獲勝機會
        opponent_mark = 3 - mark
        opponent_win = immediate_win(temp_board, opponent_mark)
        return opponent_win != -1
    
    valids = get_valid_actions(board)
    if not valids:
        return 0
    
    # 如果首選動作是安全的，直接返回
    if preferred_action in valids and not will_opponent_win_next(board, preferred_action, mark):
        return preferred_action
    
    # 否則嘗試找到安全動作
    attempted = set()
    for _ in range(max_attempts):
        # 隨機選擇一個還沒嘗試過的動作
        available = [a for a in valids if a not in attempted]
        if not available:
            break
            
        action = random.choice(available)
        attempted.add(action)
        
        if not will_opponent_win_next(board, action, mark):
            return action
    
    # 如果7次都找不到安全動作，返回首選動作或隨機動作
    if preferred_action in valids:
        return preferred_action
    return random.choice(valids) if valids else 0

def agent(obs, config):
    board = obs['board']; mark = obs['mark']
    valids = get_valid_actions(board)
    if not valids:
        return 0
    s = encode_state(board, mark)
    probs = forward_pass(s)
    # Mask invalid
    mask = np.zeros_like(probs)
    mask[valids] = probs[valids]
    if mask.sum() <= 0:
        safe = safe_moves(board, mark)
        if safe:
            preferred = int(np.random.choice(safe))
        else:
            preferred = int(valids[0])
    else:
        mask /= mask.sum()
        # Prefer safe moves with highest prob
        s_moves = safe_moves(board, mark)
        if s_moves:
            preferred = max(s_moves, key=lambda c: mask[c])
        else:
            preferred = max(valids, key=lambda c: mask[c])
    
    # 使用安全動作檢查器
    final_action = find_safe_action(board, mark, preferred)
    return int(final_action)
'''
    
    return code


if __name__ == "__main__":
    # 選最新的 checkpoint（依檔名帶時間戳或 mtime）或由參數指定
    import argparse

    parser = argparse.ArgumentParser(description="Convert PyTorch model checkpoints to submission.py files.")
    parser.add_argument("--input", type=str, default="None", help="Input checkpoint file pattern.")
    parser.add_argument("--output", type=str, default="sub/", help="Output directory for submission files.")

    args = parser.parse_args()

    ckpt_files = glob.glob(os.path.join("checkpoints", "*.pt"))

    selected = None
    if args.input:
        candidate = args.input
        # If provided is not absolute, try as-is then under checkpoints/
        tried = []
        if os.path.isabs(candidate):
            tried.append(candidate)
        else:
            tried.extend([candidate, os.path.join("checkpoints", candidate)])
        for p in tried:
            if os.path.isfile(p):
                selected = p
                break
        if selected is None:
            print(f"❌ 指定的檔案不存在: {candidate}")
            if ckpt_files:
                print("可用的檢查點有:")
                for p in sorted(ckpt_files):
                    print(" -", p)
            sys.exit(1)
    else:
        if not ckpt_files and not os.path.isfile(PREFERRED_CKPT):
            print("❌ 找不到可用的檢查點 (*.pt)。請確認 checkpoints 目錄下有有效檔案。")
            sys.exit(1)
        if ckpt_files:
            selected = choose_latest_checkpoint_by_name(ckpt_files)
        elif os.path.isfile(PREFERRED_CKPT):
            selected = PREFERRED_CKPT

    path = selected
    print(f"📦 使用檢查點: {path}")
    ckpt = try_load_checkpoint(path)

    if ckpt is None:
        # 如果使用者明確指定了 checkpoint，就直接報錯；否則嘗試 fallback 尋找
        if args.model:
            print("❌ 無法載入指定的檢查點。")
            sys.exit(1)
        path, ckpt = find_working_checkpoint()
        if ckpt is None:
            print("❌ 仍然無法載入任何檢查點。")
            sys.exit(1)

    try:
        state_dict = extract_state_dict(ckpt)
    except Exception as e:
        print(f"❌ 解析檢查點失敗: {e}")
        sys.exit(1)
   
    # 只保留張量權重並轉為 numpy
    raw_np_state = {}
    for k, v in state_dict.items():
        try:
            if isinstance(v, torch.Tensor):
                raw_np_state[k] = v.detach().cpu().numpy()
            elif isinstance(v, np.ndarray):
                raw_np_state[k] = v
            else:
                # 跳過非張量權重
                continue
        except Exception as e:
            print(f"⚠️  轉換權重失敗 {k}: {e}")

    # Normalize keys to the naming expected by the numpy-forwarder.
    # Stable-Baselines3 stores policy params under 'policy.' prefix and
    # custom feature extractors under 'policy.features_extractor.'; other
    # checkpoints may use 'module.' or 'model.' prefixes. Strip those so
    # keys like 'policy.features_extractor.stem.0.weight' -> 'stem.0.weight'.
    def _normalize_key(key: str) -> str:
        # Order matters: longer prefixes first
        prefixes = [
            'policy.features_extractor.',
            'policy.actor.features_extractor.',
            'actor.features_extractor.',
            'features_extractor.',
            'policy.',
            'actor.',
            'module.',
            'model.',
        ]
        for p in prefixes:
            if key.startswith(p):
                return key[len(p):]
        # If key contains 'policy.' in the middle (rare), remove the first occurrence
        if '.policy.' in key:
            return key.replace('.policy.', '.', 1)
        return key

    np_state = {}
    for k, v in raw_np_state.items():
        nk = _normalize_key(k)
        # Also handle double-prefix like 'policy.policy.' -> collapse
        if nk.startswith('policy.'):
            nk = nk[len('policy.'):]
        # Keep unique keys; if collision occurs, prefer the normalized one already set
        if nk in np_state:
            # prefer keeping existing; but if shapes match and names differ, keep first
            continue
        np_state[nk] = v

    out_npz = "model_weights.npz"
    np.savez_compressed(out_npz, **np_state)
    print(f"✅ 已輸出權重: {out_npz}  (包含 {len(np_state)} 個張量)")

    # 內嵌到 submission.py
    with open(out_npz, "rb") as f:
        weights_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Detect architecture type (old simple CNN vs new advanced trunk)
    has_trunk = any(k.startswith('trunk.') for k in np_state.keys())
    
    # 生成 submission 代碼
    submission_code = generate_submission_code(weights_b64, has_trunk)

    # 寫入 submission.py
    with open("submission.py", "w") as f:
        f.write(submission_code)
    
    print(f"✅ 已生成 submission.py (架構: {'高級' if has_trunk else '簡單'})")
    print(f"🔧 權重文件: {out_npz}")
    print(f"📄 提交文件: submission.py")
    
    # 驗證生成的文件
    try:
        exec(compile(open("submission.py").read(), "submission.py", "exec"))
        print("✅ submission.py 語法驗證通過")
    except Exception as e:
        print(f"⚠️ submission.py 語法驗證失敗: {e}")
        
    print("🎯 可以提交 submission.py 到 Kaggle 了！")

