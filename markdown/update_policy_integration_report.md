# Update Policy 方法整合報告

## 概述

成功將 `update_policy_from_batch` 方法與現有的 `update_policy` 方法整合，提供了兩種不同的策略更新方式。

## 方法說明

### 1. update_policy_from_batch(states, actions, old_action_probs, rewards, dones, is_weights)

**用途**: 專為批次訓練和重要性加權設計（如PER - Prioritized Experience Replay）

**特點**:
- 支持重要性加權（is_weights參數）
- 返回TD error絕對值用於更新優先級
- 更適合與Ray parallel training和PER buffer配合使用
- 支持不同類型的states輸入（numpy, tensor, list）

**輸入參數**:
```python
states: list[tensor or np]      # 狀態列表，會在內部stack到device
actions: np[int64] shape [B]    # 動作數組
old_action_probs: np[float32]   # 舊策略的動作概率
rewards: np[float32] shape [B]  # 獎勵數組
dones: np[bool] shape [B]       # 結束標記
is_weights: torch.float32 [B]   # 重要性加權（PER用）
```

**返回值**:
```python
{
    "total_loss": float,           # 總損失
    "policy_loss": float,          # 策略損失
    "value_loss": float,           # 價值損失  
    "entropy": float,              # 平均熵
    "td_errors_abs": np.array[B]   # TD誤差絕對值（用於更新優先級）
}
```

### 2. update_policy(use_batch_method=False)

**用途**: 統一的更新接口，支持兩種更新方式

**特點**:
- 當 `use_batch_method=False` 時，使用原始的更新邏輯
- 當 `use_batch_method=True` 時，內部調用 `update_policy_from_batch`
- 自動處理記憶體清空
- 保持向後兼容性

## 使用場景

### 場景1: 傳統單線程訓練
```python
# 使用原始方法
agent.store_transition(state, action, prob, reward, done)
# ... 收集更多transitions
result = agent.update_policy(use_batch_method=False)
```

### 場景2: 使用新的批次方法
```python
# 使用新的批次方法
agent.store_transition(state, action, prob, reward, done)  
# ... 收集更多transitions
result = agent.update_policy(use_batch_method=True)
```

### 場景3: 與PER buffer配合使用
```python
# 從PER buffer采樣
batch, idxs, is_weights = per_buffer.sample(batch_size)

# 準備數據
states = [b['state'] for b in batch]
actions = np.array([b['action'] for b in batch])
old_probs = np.array([b['prob'] for b in batch])
rewards = np.array([b['reward'] for b in batch])
dones = np.array([b['done'] for b in batch])
is_weights_tensor = torch.tensor(is_weights, device=agent.device)

# 直接使用批次更新
result = agent.update_policy_from_batch(
    states=states,
    actions=actions,
    old_action_probs=old_probs,
    rewards=rewards,
    dones=dones,
    is_weights=is_weights_tensor
)

# 更新PER buffer的優先級
per_buffer.update_priorities(idxs, result['td_errors_abs'])
```

### 場景4: 與Ray parallel training配合
```python
# 在train_with_ray方法中
info = self.agent.update_policy_from_batch(
    states=states,
    actions=actions,
    old_action_probs=old_probs,
    rewards=rewards,
    dones=dones,
    is_weights=is_weights
)

# 獲取TD errors用於PER優先級更新
td_errors = info.get('td_errors_abs', np.abs(rewards) + 1e-3)
per.update_priorities(idxs, td_errors, is_dangerous_flags=is_danger_flags)
```

## 關鍵改進

### 1. 重要性加權支持
- 在policy loss、value loss和entropy計算中都應用了is_weights
- 支持PER等高級經驗重放技術

### 2. 更好的TD error計算
- 使用真實的TD error: |r + γV(s') - V(s)|
- 為PER buffer提供更準確的優先級更新信號

### 3. 向後兼容性
- 保持原有update_policy的所有功能
- 通過參數選擇使用哪種更新方式

### 4. 統一的熵檢測
- 兩種方法都支持低熵檢測和部分重置
- 保持一致的訓練行為

## 測試驗證

✅ 語法檢查通過
✅ PPOAgent類別導入成功
✅ 兩種更新方法都可正常調用
✅ 保持向後兼容性

## 建議使用方式

1. **傳統訓練**: 繼續使用 `update_policy()` 或 `update_policy(use_batch_method=False)`
2. **PER訓練**: 使用 `update_policy_from_batch()` 直接調用
3. **Ray並行訓練**: 在訓練循環中使用 `update_policy_from_batch()` 
4. **遷移**: 逐步從原方法遷移到新方法，設置 `use_batch_method=True`

這樣的設計讓你可以靈活選擇最適合當前訓練場景的更新方式，同時為未來的擴展留下了空間。
