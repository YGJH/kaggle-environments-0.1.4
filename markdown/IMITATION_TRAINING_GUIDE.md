# Connect4 模仿學習 + 強化學習訓練指南

## 🎯 概述

本指南將引導您完成完整的Connect4 AI訓練流程：
1. **模仿學習預訓練**: 使用C4Solver進行監督學習
2. **強化學習精調**: 基於預訓練模型進行PPO訓練

## 📋 前置要求

### 1. 確保C4Solver可執行
```bash
# 檢查c4solver是否存在且可執行
ls -la c4solver
chmod +x c4solver  # 如果需要

# 測試c4solver
echo "4" | ./c4solver
```

### 2. 安裝依賴
```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib pyyaml
pip install kaggle-environments
```

## 🚀 完整訓練流程

### 第一階段：模仿學習預訓練

#### 1. 配置模仿學習參數
編輯 `imitation_config.yaml`:
```yaml
model:
  input_size: 126        # 必須與RL訓練一致
  hidden_size: 192       # 必須與RL訓練一致  
  num_layers: 256        # 必須與RL訓練一致

training:
  learning_rate: 0.001
  weight_decay: 0.0001
  num_samples: 20000     # 訓練樣本數量
  num_epochs: 100        # 訓練輪數
  batch_size: 128
  val_split: 0.1

paths:
  input_model: null      # 設為現有模型路徑來繼續訓練
  output_model: "imitation_pretrained_model.pt"

c4solver:
  path: "./c4solver"
  timeout: 5.0
```

#### 2. 測試系統
```bash
# 運行測試套件
python test_imitation.py
```

#### 3. 開始模仿學習預訓練
```bash
# 開始預訓練（這可能需要30分鐘到幾小時）
python imitation_learning.py
```

預期輸出：
```
🎯 Connect4 模仿學習預訓練
==================================================
✅ 載入配置文件: imitation_config.yaml
✅ C4Solver initialized successfully: ./c4solver
生成 20000 個訓練樣本...
進度: 0/20000
進度: 1000/20000
...
成功生成 19856 個訓練樣本
訓練樣本: 17870, 驗證樣本: 1986
Epoch 1/100: train_loss=1.2345, train_acc=0.234, val_loss=1.1876, val_acc=0.267, lr=1.00e-03, time=45.2s
🏆 新的最佳模型! 驗證損失: 1.1876
...
✅ 模仿學習預訓練完成!
模型已保存: imitation_pretrained_model.pt
```

### 第二階段：強化學習訓練

#### 1. 修改RL訓練配置
編輯您的RL訓練配置文件，添加預訓練模型路徑：

```python
# 在config字典中添加
config = {
    'agent': {
        'input_size': 126,
        'hidden_size': 192,
        'num_layers': 256,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        # ... 其他參數 ...
        
        # 新增：預訓練模型路徑
        'pretrained_model_path': 'imitation_pretrained_model.pt'
    },
    # ... 其他配置 ...
}
```

#### 2. 開始強化學習訓練
```bash
# 使用預訓練模型開始RL訓練
python train_connectx_rl_robust.py
```

預期輸出：
```
載入預訓練模型: imitation_pretrained_model.pt (epoch=99, loss=0.5234)
✅ 成功載入預訓練模型！模型已準備好進行RL訓練
開始PPO訓練...
Episode 1: vs 隨機對手勝率: 0.650  # 預訓練模型應該有不錯的初始表現
Episode 100: vs 隨機對手勝率: 0.750
...
```

## 📊 預期結果

### 模仿學習階段
- **訓練損失**: 應該從~2.0降到~0.5
- **準確率**: 應該達到60-80%
- **訓練時間**: 20,000樣本約需要1-3小時

### 強化學習階段  
- **初始勝率**: 對隨機對手應達到60-70%（比隨機初始化的~10%好很多）
- **收斂速度**: 應該比隨機初始化快2-3倍
- **最終性能**: 對各類對手都應有顯著提升

## 🛠️ 故障排除

### 常見問題

#### 1. C4Solver無法運行
```bash
# 檢查文件權限
chmod +x c4solver

# 檢查依賴庫
ldd c4solver  # Linux
otool -L c4solver  # macOS
```

#### 2. 模仿學習樣本生成失敗
- 檢查C4Solver是否正常工作
- 減少樣本數量進行測試
- 檢查內存使用情況

#### 3. 模型載入失敗
```python
# 檢查模型文件
import torch
checkpoint = torch.load('imitation_pretrained_model.pt')
print(checkpoint.keys())
```

#### 4. 訓練過程中記憶體不足
- 減少 `batch_size`
- 減少 `num_samples` 
- 使用CPU訓練（較慢但記憶體需求低）

### 調試技巧

#### 1. 驗證預訓練效果
```python
# 測試預訓練模型對空局面的預測
from imitation_learning import ImitationLearner, load_config

config = load_config()
learner = ImitationLearner(config)
learner.load_model('imitation_pretrained_model.pt')

# 測試空局面
empty_board = [0] * 42
state = learner.dataset.encode_state_for_model(empty_board, 1)
with torch.no_grad():
    policy, value = learner.model(torch.FloatTensor(state).unsqueeze(0))
    print(f"空局面策略: {policy[0].numpy()}")
    print(f"空局面價值: {value[0].item()}")
```

#### 2. 比較訓練前後
```bash
# 保存隨機初始化的評估結果
python train_connectx_rl_robust.py --evaluate-only > before_pretraining.txt

# 使用預訓練模型評估
# 在config中設置pretrained_model_path後
python train_connectx_rl_robust.py --evaluate-only > after_pretraining.txt

# 比較結果
diff before_pretraining.txt after_pretraining.txt
```

## 📈 性能優化建議

### 1. 模仿學習階段
- **增加樣本多樣性**: 調整`max_moves`參數
- **平衡數據**: 確保不同局面階段都有足夠樣本
- **調整學習率**: 根據收斂情況微調

### 2. 強化學習階段
- **初始探索率**: 可以降低，因為預訓練模型已有良好策略
- **學習率**: 可以使用較小的學習率進行精調
- **對手課程**: 可以更早引入強對手

### 3. 硬體優化
- **GPU訓練**: 顯著加速模仿學習階段
- **並行數據生成**: 可以並行生成訓練樣本
- **記憶體管理**: 批次處理大型數據集

## 🎮 使用建議

1. **漸進式訓練**: 先用少量樣本測試，確認流程正常後再進行完整訓練
2. **定期驗證**: 在不同階段測試模型對戰表現
3. **保存檢查點**: 定期保存模型以防訓練中斷
4. **監控日誌**: 關注訓練損失和準確率趨勢
5. **性能測試**: 定期與不同對手對戰評估進步

這個完整的訓練流程將為您的Connect4 AI提供強大的基礎，預期能顯著提升訓練效率和最終性能！🚀
