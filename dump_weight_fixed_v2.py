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




if __name__ == "__main__":
    # 選最新的 checkpoint（依檔名帶時間戳或 mtime）或由參數指定
    import argparse

    parser = argparse.ArgumentParser(description="Dump a ConnectX checkpoint to submission.py with embedded weights")
    parser.add_argument(
        "-m", "--model", type=str, default=None,
        help="Path to a specific checkpoint .pt to dump (absolute or relative). If relative and not found, 'checkpoints/<name>' will be tried."
    )
    args = parser.parse_args()

    ckpt_files = glob.glob(os.path.join("checkpoints", "*.pt"))

    selected = None
    if args.model:
        candidate = args.model
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
    np_state = {}
    for k, v in state_dict.items():
        try:
            if isinstance(v, torch.Tensor):
                np_state[k] = v.detach().cpu().numpy()
            elif isinstance(v, np.ndarray):
                np_state[k] = v
            else:
                # 跳過非張量權重
                continue
        except Exception as e:
            print(f"⚠️  轉換權重失敗 {k}: {e}")

    out_npz = "model_weights.npz"
    np.savez_compressed(out_npz, **np_state)
    print(f"✅ 已輸出權重: {out_npz}  (包含 {len(np_state)} 個張量)")

    # 內嵌到 submission.py
    with open(out_npz, "rb") as f:
        weights_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Detect architecture type (old simple CNN vs new advanced trunk)
    has_trunk = any(k.startswith('trunk.') for k in np_state.keys())

    if has_trunk:
        arch_comment = "# Advanced architecture: BottleneckSE blocks + optional SpatialSelfAttention + coord embeddings"
        submission_code = f'''import numpy as np
import base64, io
{arch_comment}
WEIGHTS_B64 = "{weights_b64}"

def load_weights():
    buf = io.BytesIO(base64.b64decode(WEIGHTS_B64))
    data = np.load(buf)
    return {{k: data[k] for k in data.files}}
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
    _is_attn[idx] = f'trunk.{idx}.qkv.weight' in weights

# Infer channels / hidden
stem_w = weights['stem.0.weight']
C = stem_w.shape[0]
hidden = weights['head.1.weight'].shape[0]
half_hidden = hidden//2

# Precompute coord planes (-1..1)
_row = np.linspace(-1,1,6).reshape(6,1).repeat(7,1)
_col = np.linspace(-1,1,7).reshape(1,7).repeat(6,0)
# fix _col repeat usage: correct creation
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
            qkv = conv1x1(x, weights[f'trunk.{idx}.qkv.weight'], None)
            Bq = qkv.shape[0]//3
            q, k, v = np.split(qkv, 3, axis=0)
            # heads fixed to 4
            heads = 4
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
            out = conv1x1(out, weights[f'trunk.{idx}.proj.weight'], None)
            x = out  # no residual in original implementation
        else:
            # BottleneckSE
            c1 = conv1x1(x, weights[f'trunk.{idx}.conv1.weight'], None)
            c1 = group_norm(c1, weights[f'trunk.{idx}.gn1.weight'], weights[f'trunk.{idx}.gn1.bias'])
            c1 = relu(c1)
            c2 = conv2d(c1, weights[f'trunk.{idx}.conv2.weight'], None, pad=1)
            c2 = group_norm(c2, weights[f'trunk.{idx}.gn2.weight'], weights[f'trunk.{idx}.gn2.bias'])
            c2 = relu(c2)
            c3 = conv1x1(c2, weights[f'trunk.{idx}.conv3.weight'], None)
            c3 = group_norm(c3, weights[f'trunk.{idx}.gn3.weight'], weights[f'trunk.{idx}.gn3.bias'])
            # SE
            se_vec = c3.mean(axis=(1,2))  # (C,)
            # fc1
            w1 = weights[f'trunk.{idx}.se_fc1.weight']; b1 = weights[f'trunk.{idx}.se_fc1.bias']
            w2 = weights[f'trunk.{idx}.se_fc2.weight']; b2 = weights[f'trunk.{idx}.se_fc2.bias']
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

# -------------- Agent main --------------

def my_agent(obs, config):
    board = obs['board']; mark = obs['mark']
    win = immediate_win(board, mark)
    if win != -1:
        return int(win)
    block = immediate_block(board, mark)
    if block != -1:
        return int(block)
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
            return int(np.random.choice(safe))
        return int(valids[0])
    mask /= mask.sum()
    # Prefer safe moves with highest prob
    s_moves = safe_moves(board, mark)
    if s_moves:
        best = max(s_moves, key=lambda c: mask[c])
        return int(best)
    return int(max(valids, key=lambda c: mask[c]))

