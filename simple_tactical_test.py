#!/usr/bin/env python3
"""
簡化的戰術對手測試
"""

import numpy as np
import torch
import sys
import os

# 添加當前目錄到Python路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from train_connectx_rl_robust import ConnectXTrainer
    print("✅ 成功導入 ConnectXTrainer")
    
    # 創建訓練器
    trainer = ConnectXTrainer()
    print("✅ 成功創建 ConnectXTrainer 實例")
    
    # 創建一個簡單的測試棋盤 (玩家1在第3列有3個連續棋子)
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ])
    
    print("測試棋盤:")
    print(board)
    print("玩家1可以在第3列獲勝")
    
    # 測試 if_i_can_finish 函數
    valid_actions = [0, 1, 2, 3, 4, 5, 6]
    winning_move = trainer.if_i_can_finish(board, 1, valid_actions)
    
    print(f"if_i_can_finish 返回: {winning_move}")
    
    if winning_move == 3:
        print("✅ if_i_can_finish 函數工作正常")
    else:
        print("❌ if_i_can_finish 函數可能有問題")
    
    # 測試編碼和戰術對手
    board_flat = board.flatten()
    state = trainer.agent.encode_state(board_flat, 1)
    print(f"編碼狀態維度: {len(state)}")
    
    # 創建戰術對手
    tactical_func = trainer.create_tactical_opponent()
    
    print("測試戰術對手...")
    action, prob, value = tactical_func(state, valid_actions, False)
    
    print(f"戰術對手選擇動作: {action}")
    print(f"期望動作: 3")
    
    if action == 3:
        print("🎉 戰術對手正確工作！")
    else:
        print("⚠️ 戰術對手可能需要調試")

except Exception as e:
    print(f"❌ 錯誤: {e}")
    import traceback
    traceback.print_exc()
