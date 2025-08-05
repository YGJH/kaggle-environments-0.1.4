# ConnectX 監督學習訓練完整指南

## 🎯 項目概述

這是一個完整的ConnectX監督學習訓練系統，使用`connectx-state-action-value.txt`數據集來訓練深度神經網路，讓AI學會玩ConnectX（四子棋）遊戲。

## 📁 文件結構

```
train_connectx_supervised.py    # 主要訓練腳本
test_supervised_simple.py       # 測試腳本
connectx-state-action-value.txt # 訓練數據集（必需）
checkpoints/                    # 模型檢查點目錄
logs/                          # 訓練日誌目錄
```

## 🚀 快速開始

### 1. 環境準備

確保你有以下文件：
- ✅ `connectx-state-action-value.txt` - 訓練數據集
- ✅ `train_connectx_supervised.py` - 訓練腳本
- ✅ Python 3.8+ 環境
- ✅ PyTorch + CUDA（推薦）

### 2. 運行測試

```bash
# 測試環境是否正確配置
python test_supervised_simple.py
```

### 3. 開始訓練

```bash
# 開始完整訓練
python train_connectx_supervised.py
```

## 📊 訓練配置

### 默認配置參數

```python
config = {
    'agent': {
        'input_size': 126,      # 輸入維度（3通道 × 42位置）
        'hidden_size': 256,     # 隱藏層大小
        'num_layers': 3,        # 隱藏層數量
        'learning_rate': 0.001, # 學習率
        'weight_decay': 0.0001  # 權重衰減
    },
    'training': {
        'epochs': 200,          # 訓練輪數
        'batch_size': 128,      # 批次大小
        'max_lines': 50000,     # 最大數據集行數
        'eval_games': 100       # 評估遊戲數量
    }
}
```

### 自定義配置

你可以修改`train_connectx_supervised.py`中的`create_config()`函數來調整參數：

```python
def create_config():
    config = {
        'training': {
            'epochs': 500,          # 增加訓練輪數
            'max_lines': 100000,    # 使用更多數據
            'batch_size': 256,      # 增加批次大小
        }
    }
    return config
```

## 📈 性能監控

### 訓練過程監控

訓練過程中會顯示：
- **Loss**: 總體損失（越小越好）  
- **Policy**: 策略損失（動作選擇準確性）
- **Value**: 價值損失（狀態評估準確性）
- **Time**: 每個epoch用時

```
Epoch [10/200] Loss: 0.629591 Policy: 0.608997 Value: 0.041189 Time: 2.43s
```

### 檢查點保存

系統會自動保存：
- `best_supervised_model.pt` - 最佳模型（每次改進時更新）
- `supervised_epoch_X.pt` - 定期檢查點（每50個epoch）
- `supervised_final_YYYYMMDD_HHMMSS.pt` - 最終模型

## 🎯 訓練結果評估

### 自動評估

訓練完成後，系統會自動進行100局對隨機對手的測試：

```
📊 評估結果:
   勝利: 85 (85.0%)
   平局: 10 (10.0%)
   失敗: 5 (5.0%)
```

### 性能指標

- **≥80%**: 🌟 優異性能，可直接用於比賽
- **60-80%**: 👍 良好性能，建議繼續訓練
- **<60%**: ⚠️ 需要改進，調整參數或增加訓練

## 🔧 進階配置

### 1. 調整網路結構

```python
'agent': {
    'hidden_size': 512,     # 增加網路容量
    'num_layers': 5,        # 增加網路深度
}
```

### 2. 調整學習參數

```python
'agent': {
    'learning_rate': 0.0005,  # 降低學習率提高穩定性
    'weight_decay': 0.0001,   # 調整正規化強度
}
```

### 3. 調整數據使用

```python
'training': {
    'max_lines': 100000,    # 使用更多訓練數據
    'batch_size': 256,      # 增加批次大小
}
```

## ⚡ 性能優化

### GPU 加速

系統會自動檢測並使用CUDA GPU：
```
🔧 使用設備: cuda
```

### 記憶體優化

- 數據分批載入避免記憶體溢出
- 自動垃圾回收清理臨時變量
- 梯度剪裁防止梯度爆炸

### 訓練速度

在RTX 3060上的典型性能：
- **數據載入**: ~1秒（50,000樣本）
- **每個epoch**: ~2.2秒  
- **完整訓練**: ~7-8分鐘（200 epochs）

## 🐛 常見問題

### 1. 找不到數據集文件

```
❌ 找不到數據集文件: connectx-state-action-value.txt
```

**解決方案**: 確保`connectx-state-action-value.txt`文件在當前目錄

### 2. CUDA 記憶體不足

```
RuntimeError: CUDA out of memory
```

**解決方案**: 
- 減小`batch_size`（如改為64）
- 減小`max_lines`（如改為25000）
- 減小網路大小`hidden_size`（如改為128）

### 3. 訓練收斂緩慢

**解決方案**:
- 調整學習率`learning_rate`
- 增加數據量`max_lines`
- 檢查數據質量

## 📝 使用示例

### 基本訓練

```bash
# 使用默認參數訓練
python train_connectx_supervised.py
```

### 快速測試

```bash
# 修改配置進行快速測試
# 在 create_config() 中設置:
# 'epochs': 20, 'max_lines': 1000
python train_connectx_supervised.py
```

### 長時間訓練

```bash
# 後台運行長時間訓練
nohup python train_connectx_supervised.py > training.log 2>&1 &
```

## 🎉 訓練完成後

### 使用訓練好的模型

1. 模型文件位於`checkpoints/`目錄
2. 可以加載`best_supervised_model.pt`用於遊戲
3. 可以與其他AI或人類玩家對戰

### 繼續改進

1. 增加訓練數據量
2. 調整網路結構
3. 結合強化學習進一步優化
4. 實現更複雜的評估策略

---

## 💡 最佳實踐

1. **從小規模開始**: 先用少量數據測試配置
2. **監控訓練過程**: 觀察損失變化確保正常收斂  
3. **保存多個檢查點**: 避免訓練意外中斷造成損失
4. **定期評估**: 每訓練一段時間評估模型性能
5. **記錄實驗**: 記錄不同配置的效果便於比較

現在你可以開始訓練你的ConnectX AI了！🚀
