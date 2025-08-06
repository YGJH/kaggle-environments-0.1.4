#!/usr/bin/env python3
"""
測試反收斂停滯機制
"""

import torch
import numpy as np
from train_connectx_rl_robust import ConnectXTrainer
import logging

# 設置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_training_strategy_selection():
    """測試訓練策略選擇機制"""
    print("=" * 60)
    print("測試訓練策略選擇機制")
    print("=" * 60)
    
    trainer = ConnectXTrainer()
    
    # 模擬不同回合的策略選擇
    test_episodes = [1000, 25000, 75000, 100000]
    
    # 模擬不同的勝率情況
    trainer.win_rates = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.4, 0.3, 0.2]
    
    for episode in test_episodes:
        strategy = trainer._select_training_strategy(episode)
        print(f"Episode {episode}: 選擇策略 = {strategy}")
    
    print("\n測試完成！")

def test_convergence_stagnation_detection():
    """測試收斂停滯檢測"""
    print("=" * 60)
    print("測試收斂停滯檢測")
    print("=" * 60)
    
    trainer = ConnectXTrainer()
    
    # 測試場景1：勝率變化很小
    trainer.win_rates = [0.5] * 20  # 勝率完全不變
    trainer.last_self_play_rate = 0.3
    
    stagnation = trainer._detect_convergence_stagnation(50000, 0.5)
    print(f"場景1 - 勝率不變: 檢測結果 = {stagnation}")
    
    # 測試場景2：自對弈勝率過低
    trainer.win_rates = [0.7, 0.8, 0.75, 0.82, 0.78] * 4  # 正常變化
    trainer.last_self_play_rate = 0.25  # 自對弈勝率很低
    
    stagnation = trainer._detect_convergence_stagnation(60000, 0.78)
    print(f"場景2 - 自對弈勝率過低: 檢測結果 = {stagnation}")
    
    # 測試場景3：正常情況
    trainer.win_rates = [0.5, 0.6, 0.55, 0.65, 0.7, 0.68, 0.75, 0.72, 0.8, 0.78] * 2
    trainer.last_self_play_rate = 0.6
    
    stagnation = trainer._detect_convergence_stagnation(30000, 0.78)
    print(f"場景3 - 正常情況: 檢測結果 = {stagnation}")
    
    print("\n測試完成！")

def test_new_training_methods():
    """測試新的訓練方法"""
    print("=" * 60)
    print("測試新的訓練方法")
    print("=" * 60)
    
    trainer = ConnectXTrainer()
    
    # 測試課程化自對弈
    try:
        print("測試課程化自對弈...")
        reward, length = trainer.curriculum_self_play_episode(5000)
        print(f"課程化自對弈結果: 獎勵={reward}, 長度={length}")
    except Exception as e:
        print(f"課程化自對弈錯誤: {e}")
    
    # 測試探索增強自對弈
    try:
        print("測試探索增強自對弈...")
        reward, length = trainer.exploration_enhanced_self_play(5000)
        print(f"探索增強自對弈結果: 獎勵={reward}, 長度={length}")
    except Exception as e:
        print(f"探索增強自對弈錯誤: {e}")
    
    # 測試噪聲自對弈
    try:
        print("測試噪聲自對弈...")
        reward, length = trainer.noisy_self_play_episode(5000)
        print(f"噪聲自對弈結果: 獎勵={reward}, 長度={length}")
    except Exception as e:
        print(f"噪聲自對弈錯誤: {e}")
    
    print("\n測試完成！")

def test_stagnation_handling():
    """測試收斂停滯處理"""
    print("=" * 60)
    print("測試收斂停滯處理")
    print("=" * 60)
    
    trainer = ConnectXTrainer()
    
    # 記錄處理前的狀態
    initial_exploration = trainer.current_exploration
    initial_lr = trainer.agent.optimizer.param_groups[0]['lr']
    initial_entropy_coef = getattr(trainer.agent, 'entropy_coef', 0.01)
    
    print(f"處理前 - 探索率: {initial_exploration:.3f}, 學習率: {initial_lr:.2e}, 熵係數: {initial_entropy_coef:.4f}")
    
    # 執行收斂停滯處理
    trainer._handle_convergence_stagnation(75000)
    
    # 檢查處理後的狀態
    after_exploration = trainer.current_exploration
    after_lr = trainer.agent.optimizer.param_groups[0]['lr']
    after_entropy_coef = getattr(trainer.agent, 'entropy_coef', 0.01)
    
    print(f"處理後 - 探索率: {after_exploration:.3f}, 學習率: {after_lr:.2e}, 熵係數: {after_entropy_coef:.4f}")
    
    print("\n測試完成！")

def run_comprehensive_test():
    """運行全面測試"""
    print("\n" + "=" * 80)
    print("ConnectX 反收斂停滯機制綜合測試")
    print("=" * 80)
    
    test_training_strategy_selection()
    test_convergence_stagnation_detection()
    test_new_training_methods()
    test_stagnation_handling()
    
    print("\n" + "=" * 80)
    print("所有測試完成！")
    print("=" * 80)
    print("\n建議:")
    print("1. 新的訓練策略可以有效增加多樣性")
    print("2. 收斂停滯檢測可以及時發現問題")
    print("3. 自適應處理機制可以動態調整訓練參數")
    print("4. 課程化學習可以利用歷史模型增加挑戰性")
    print("5. 部分權重重置可以打破局部最優")

if __name__ == "__main__":
    run_comprehensive_test()
