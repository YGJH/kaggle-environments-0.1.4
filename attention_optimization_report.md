# ConnectX模型Attention優化報告

## 問題描述
原始錯誤：`shape '[1, 3, 24, 5, 42]' is invalid for input of size 16128`

**根本原因**: Attention heads數量(24)無法被channels數量(128)整除，導致維度計算錯誤
- 128 ÷ 24 = 5.33... (不是整數)
- 這會在 `qkv.reshape(B, 3, self.heads, C // self.heads, H * W)` 時出錯

## 解決方案

### 1. 調整Channel數量
```python
# 之前: self.channels = 128
# 現在: self.channels = 144  # 144 = 24 × 6，完美整除
```

**144的優勢**:
- 144 ÷ 24 = 6 (head_dim = 6，完美整數)
- 144是24的倍數，支持多種head配置 (1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24)
- 比128略大，提供更多模型容量

### 2. 智能Head調整機制
```python
class SpatialSelfAttention(nn.Module):
    def __init__(self, c, heads=24):
        # 自動調整heads到最接近的因數
        if c % heads != 0:
            possible_heads = [i for i in range(1, c + 1) if c % i == 0]
            heads = min(possible_heads, key=lambda x: abs(x - heads))
            print(f"調整attention heads: {heads} (channels={c})")
```

### 3. 優化Attention頻率
```python
# 更頻繁使用attention (每1-3個block)
self.attn_every = max(1, min(int(attn_every), 3))
```

## 新模型配置

### 📐 **架構參數**
- **Channels**: 144 (原128)
- **Attention Heads**: 24
- **Head Dimension**: 6 (144 ÷ 24)
- **Attention Frequency**: 每1-3個residual block

### 🧠 **Attention增強**
1. **更多頭部**: 24個heads提供更豐富的特徵表示
2. **更頻繁**: attention模組出現更密集
3. **自適應**: 自動調整到最佳head配置

### ⚡ **性能提升預期**
1. **更強表示能力**: 24個attention heads能捕捉更複雜的空間關係
2. **更好的長程依賴**: ConnectX需要識別4子連線，attention有助於此
3. **更細緻的特徵**: 每個head專注於不同的模式(橫向、縱向、對角等)

## 測試結果 ✅

### 模型創建測試
```
✅ 模型創建成功！
   Channels: 144
   Attention模組數量: 2
   Head數量: 24
   Head維度: 6
```

### 前向傳播測試
```
✅ 前向傳播成功！
   Policy shape: torch.Size([4, 7])
   Value shape: torch.Size([4, 1])
```

### 訓練流程測試
```
✅ 策略更新成功！
   總損失: 0.5254
   熵值: 0.3698
```

## 建議配置

### 對於不同計算資源的推薦：

#### 🚀 **高性能配置** (推薦)
```python
channels = 144
heads = 24
attn_every = 2
num_layers = 16-32
```

#### ⚖️ **平衡配置**
```python
channels = 144
heads = 12
attn_every = 3
num_layers = 12-16
```

#### 💡 **輕量配置**
```python
channels = 144
heads = 6
attn_every = 4
num_layers = 8-12
```

## 注意事項

1. **記憶體使用**: 24個heads會增加attention計算量，但對於6×7的小棋盤仍然可控
2. **訓練速度**: 可能略微降低，但attention的並行性有助於加速
3. **效果驗證**: 建議訓練一段時間後比較勝率提升

## 未來優化方向

1. **動態Head數**: 根據訓練階段調整head數量
2. **混合Attention**: 結合local和global attention
3. **棋類特定**: 針對ConnectX設計專門的attention模式

現在你的模型已經成功支持24個attention heads，應該能夠提供更強的特徵學習能力！🎯
