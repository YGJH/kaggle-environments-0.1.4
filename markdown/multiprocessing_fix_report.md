# ConnectX Multiprocessing 問題分析與修復報告

## 問題發現

經過詳細檢查 `train_parallel` 和 `_worker_collect_episode` 之間的調用邏輯，發現了以下問題：

### 1. 參數不匹配問題

**問題描述：**
- `_collect_batch` 傳遞了 `'use_tactical_opponent': use_tac` 參數
- 但 `_worker_collect_episode` 函數中並沒有使用這個參數
- 這是一個無效的參數傳遞

**原代碼：**
```python
# _collect_batch 中
use_tac = use_tactical_opp and (rng.random() < tactical_ratio)
args.append({
    'config': self.config,
    'policy_state': policy_state_numpy,
    'player2_training_prob': self.player2_training_prob,
    'use_tactical_opponent': use_tac,  # ← 這個參數沒被使用
    'seed': rng.randrange(2**31 - 1),
})
```

**修復：**
移除了未使用的 `use_tactical_opponent` 參數。

### 2. Worker 進程重建問題

**問題描述：**
- 設置了 `maxtasksperchild=64`，這會導致每個 worker 進程在處理 64 個任務後自動重新啟動
- 這是你觀察到的 "跑完一次evaluation就整個重建pool" 現象的根本原因
- 頻繁的進程重建會導致性能下降和資源浪費

**原代碼：**
```python
pool = mp_ctx.Pool(processes=num_workers, maxtasksperchild=64)
```

**修復：**
移除了 `maxtasksperchild` 限制，讓 worker 進程持續運行。

### 3. 不必要的配置變數

**問題描述：**
- `use_tactical_opp` 和 `tactical_ratio` 配置變數已經不需要
- 因為 `_worker_collect_episode` 內部已經隨機選擇對手類型（'random', 'minimax', 'self'）

**修復：**
移除了相關的配置讀取。

## 修復後的改進

### 1. 簡化的參數傳遞
```python
def _collect_batch(policy_state_numpy, n_episodes: int):
    args = []
    for i in range(n_episodes):
        args.append({
            'config': self.config,
            'policy_state': policy_state_numpy,
            'player2_training_prob': self.player2_training_prob,
            'seed': rng.randrange(2**31 - 1),
        })
    results = pool.map(_worker_collect_episode, args)
    return results
```

### 2. 持久的 Worker Pool
```python
# 移除 maxtasksperchild 限制，避免不必要的進程重建
pool = mp_ctx.Pool(processes=num_workers)
```

### 3. 更清晰的對手選擇邏輯
Worker 內部隨機選擇對手：
```python
# 隨機選擇對手類型: 'random', 'minimax', 'self'
opponent_types = ['random', 'minimax', 'self']
opponent_type = random.choice(opponent_types)
```

## 性能改進預期

1. **減少進程重建開銷**：不再每 64 個任務重建一次 worker
2. **更穩定的並行處理**：worker 進程持續運行直到訓練完成
3. **減少內存分配**：避免頻繁的進程創建和銷毀
4. **更好的資源利用**：worker 進程保持熱狀態，減少初始化時間

## 驗證結果

✅ 語法檢查通過  
✅ 參數匹配正確  
✅ Pool 創建邏輯簡化  
✅ 移除無效配置項

## 結論

修復了 multiprocessing 的邏輯問題，主要是：
1. 移除了導致 worker 進程頻繁重建的 `maxtasksperchild=64` 設置
2. 清理了未使用的參數傳遞
3. 簡化了配置邏輯

現在 worker pool 會在整個訓練過程中保持穩定，不會出現評估後重建的問題。
