# ConnectX 可視化功能使用指南

## 功能概述

我們已經為 ConnectX RL 訓練系統添加了遊戲可視化功能，可以在訓練過程中每 100 個回合自動展示 AI 對戰不同對手的畫面。

## 新增功能

### 1. 自動可視化
- **觸發頻率**: 每 100 個訓練回合
- **對手類型**: 根據訓練進度自動選擇
  - 回合 0-2999: 對戰隨機對手 (Random Agent)
  - 回合 3000-4999: 對戰自己 (Self-Play)
  - 回合 5000+: 對戰 Minimax 對手

### 2. 可視化內容
- **遊戲棋盤**: 實時顯示 6×7 連線棋盤狀態
- **棋子標記**: 紅色圓圈代表玩家 1，藍色圓圈代表玩家 2
- **對手信息**: 清楚標註當前對戰的對手類型
- **遊戲歷程**: 顯示每一步的棋盤變化

## 使用方法

### 自動可視化（推薦）
直接運行訓練腳本，系統會自動在每 100 回合時展示可視化：

```bash
python train_connectx_rl_robust.py
```

訓練日誌會顯示：
```
第 100 回合：展示對戰 random 對手
第 200 回合：展示對戰 random 對手
第 3000 回合：展示對戰 self_play 對手
第 5000 回合：展示對戰 minimax 對手
```

### 手動測試可視化
使用測試腳本驗證可視化功能：

```bash
python test_visualization.py
```

### 自定義可視化
您也可以在代碼中手動調用可視化功能：

```python
from train_connectx_rl_robust import ConnectXTrainer

# 創建訓練器
trainer = ConnectXTrainer(config)

# 手動展示對戰特定對手
trainer.demo_game_with_visualization("random")    # 對戰隨機對手
trainer.demo_game_with_visualization("self_play") # 自我對戰
trainer.demo_game_with_visualization("minimax")   # 對戰 Minimax
```

## 技術細節

### 依賴要求
- `matplotlib`: 用於遊戲棋盤可視化
- `numpy`: 數值計算
- `kaggle-environments`: ConnectX 遊戲環境

### 可視化組件
1. **visualize_game()**: 核心可視化方法，展示完整遊戲過程
2. **demo_game_with_visualization()**: 便捷方法，根據對手類型自動設置
3. **VISUALIZATION_AVAILABLE**: 自動檢測 matplotlib 是否可用

### 錯誤處理
- 如果 matplotlib 不可用，系統會跳過可視化但繼續訓練
- 可視化過程中的錯誤會記錄日誌但不影響主訓練流程
- 每次可視化都有異常捕獲機制

## 訓練階段說明

### 階段一：基礎學習 (0-2999 回合)
- **對手**: 隨機對手
- **目標**: 學習基本的遊戲規則和連線策略
- **可視化**: 展示 AI 如何從隨機行為學習到有意識的走棋

### 階段二：自我提升 (3000-4999 回合)
- **對手**: 自我對戰
- **目標**: 通過與自己對戰來發現更複雜的策略
- **可視化**: 展示兩個相同 AI 之間的高質量對局

### 階段三：挑戰高手 (5000+ 回合)
- **對手**: Minimax 對手
- **目標**: 對抗傳統的 AI 算法來測試策略深度
- **可視化**: 展示 RL 對戰傳統演算法的精彩對局

## 日誌信息

可視化功能會在訓練日誌中提供以下信息：
- 回合數和對手類型
- 可視化成功/失敗狀態
- 遊戲結果（勝負）
- 任何錯誤或警告信息

## 故障排除

### 常見問題
1. **matplotlib 未安裝**: 安裝 `pip install matplotlib`
2. **顯示器問題**: 在無頭環境中可能無法顯示，但不影響訓練
3. **內存使用**: 可視化會略微增加內存使用，但在合理範圍內

### 禁用可視化
如果需要禁用可視化功能，可以設置環境變量：
```bash
export DISABLE_VISUALIZATION=1
python train_connectx_rl_robust.py
```

## 性能影響

- **訓練速度**: 可視化只在每 100 回合觸發，對整體訓練速度影響極小
- **內存使用**: 每次可視化會創建臨時圖形對象，使用後自動釋放
- **磁盤空間**: 不會自動保存圖片，如需保存可手動修改代碼

## 未來擴展

計劃中的改進包括：
- 保存可視化動畫為 GIF 或視頻文件
- 添加更多統計信息展示
- 支持自定義可視化頻率
- 添加勝率趨勢圖表
