# ConnectX 強化學習訓練專案

這是一個使用 PPO (Proximal Policy Optimization) 算法訓練 ConnectX AI 的完整專案，專為 Kaggle ConnectX 競賽設計。

## 專案結構

```
├── train_connectx_rl.py      # 主要訓練腳本
├── connectx_config.yaml      # 訓練配置文件
├── test_simple.py           # 簡單測試腳本
├── checkpoints/             # 模型檢查點目錄
│   └── best_model_wr_*.pt   # 最佳模型檔案
├── logs/                    # 訓練日誌目錄
└── ConnectX_RL_README.md   # 本文件
```

## 快速開始

### 1. 安裝依賴

```bash
# 使用 uv（推薦）
uv pip install PyYAML torch tqdm matplotlib tensorboard

# 或使用 pip
pip install PyYAML torch tqdm matplotlib tensorboard
```

### 2. 測試環境

```bash
# 運行簡單測試
uv run test_simple.py
```

### 3. 開始訓練

```bash
# 使用默認設置訓練
uv run train_connectx_rl.py

# 自定義訓練參數
uv run train_connectx_rl.py --config connectx_config.yaml --episodes 50000 --eval-freq 1000
```

### 4. 監控訓練進度

訓練過程會輸出以下資訊：
- 當前回合數
- 對隨機對手的勝率
- 平均獎勵
- 回合長度

最佳模型會自動保存到 `checkpoints/` 目錄。

## 訓練參數說明

### 網路架構參數
- `input_size`: 輸入大小 (126 = 6×7×3 通道)
- `hidden_size`: 隱藏層大小，增加可提高模型容量
- `num_layers`: 殘差層數量，增加可增加模型深度

### 強化學習參數
- `learning_rate`: 學習率，控制訓練速度
- `gamma`: 折扣因子，控制長期獎勵重要性
- `eps_clip`: PPO 裁剪參數，控制策略更新幅度
- `entropy_coef`: 熵係數，平衡探索和利用

### 訓練設置
- `max_episodes`: 最大訓練回合數
- `eval_frequency`: 每多少回合評估一次
- `eval_games`: 評估時進行的遊戲數量

## 模型特色

### 1. 強大的神經網路架構
- 使用殘差連接的深度網路
- 分離的策略頭和價值頭
- Dropout 防止過擬合

### 2. PPO 強化學習算法
- 穩定的策略梯度方法
- 自對弈訓練
- Generalized Advantage Estimation (GAE)

### 3. ConnectX 特化設計
- 三通道狀態編碼（己方、對方、空位）
- 動作遮罩確保有效動作
- 獎勵塑形鼓勵策略性遊戲

### 4. 完整的訓練管道
- 自動模型保存和檢查點
- 定期評估和早停機制
- 訓練統計和監控

## 性能調優建議

### 快速訓練（開發測試）
```yaml
agent:
  hidden_size: 256
  num_layers: 2
training:
  max_episodes: 5000
  eval_frequency: 250
```

### 高性能訓練（競賽用）
```yaml
agent:
  hidden_size: 1024
  num_layers: 6
training:
  max_episodes: 100000
  eval_frequency: 2000
```

## Kaggle 提交

訓練完成後，使用最佳模型創建 Kaggle 提交。

## 預期性能

- **初始性能**: 對隨機對手勝率 ~75-80%
- **短期訓練** (5000 回合): 勝率 ~85-90%
- **長期訓練** (50000+ 回合): 勝率 ~95%+

## 作者

GitHub Copilot - AI 程式設計助手
