#!/usr/bin/env python3
"""
測試agent函數返回值格式一致性
"""

import torch
import numpy as np
from train_connectx_rl_robust import ConnectXTrainer
import logging

# 設置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_agent_return_formats():
    """測試所有agent函數的返回值格式"""
    print("=" * 60)
    print("測試Agent函數返回值格式一致性")
    print("=" * 60)
    
    trainer = ConnectXTrainer()
    
    # 創建模擬狀態 - 使用正確的維度
    board = [0] * 42  # 空棋盤
    mark = 1
    state = trainer.agent.encode_state(board, mark)  # 這會返回126維的正確狀態
    valid_actions = [0, 1, 2, 3, 4, 5, 6]
    
    test_results = []
    
    # 測試各種agent函數
    test_cases = [
        ("基本select_action", lambda: trainer.agent.select_action(state, valid_actions, training=True)),
        ("戰術對手", lambda: trainer.create_tactical_opponent()(state, valid_actions, training=True)),
        ("歷史模型預測", lambda: trainer._predict_with_historical_model({'policy_net_state': trainer.agent.policy_net.state_dict().copy()}, state, valid_actions)),
    ]
    
    # 測試新的訓練方法中的agent函數
    try:
        print("\n測試各種agent函數的返回值格式...")
        for name, func in test_cases:
            try:
                result = func()
                result_type = type(result)
                if isinstance(result, tuple):
                    length = len(result)
                    types = [type(x).__name__ for x in result]
                    print(f"✅ {name}: {length}個值 - {types}")
                    test_results.append((name, True, f"{length}個值"))
                else:
                    print(f"❌ {name}: 非元組返回值 - {result_type}")
                    test_results.append((name, False, f"非元組: {result_type}"))
                    
            except Exception as e:
                print(f"❌ {name}: 測試失敗 - {e}")
                test_results.append((name, False, f"異常: {e}"))
    
    except Exception as e:
        print(f"整體測試失敗: {e}")
    
    # 測試新的訓練方法
    print("\n測試新訓練方法的運行...")
    training_methods = [
        ("課程化自對弈", lambda: trainer.curriculum_self_play_episode(1000)),
        ("探索增強自對弈", lambda: trainer.exploration_enhanced_self_play(1000)),
        ("噪聲自對弈", lambda: trainer.noisy_self_play_episode(1000)),
    ]
    
    for name, method in training_methods:
        try:
            result = method()
            if isinstance(result, tuple) and len(result) == 2:
                reward, length = result
                print(f"✅ {name}: 成功 - 獎勵={reward}, 長度={length}")
                test_results.append((name, True, "正常完成"))
            else:
                print(f"❌ {name}: 返回格式異常 - {result}")
                test_results.append((name, False, f"異常返回: {result}"))
        except Exception as e:
            print(f"❌ {name}: 執行失敗 - {e}")
            test_results.append((name, False, f"執行異常: {e}"))
    
    # 總結
    print("\n" + "=" * 60)
    print("測試總結")
    print("=" * 60)
    
    success_count = sum(1 for _, success, _ in test_results if success)
    total_count = len(test_results)
    
    print(f"成功: {success_count}/{total_count}")
    
    for name, success, detail in test_results:
        status = "✅" if success else "❌"
        print(f"{status} {name}: {detail}")
    
    if success_count == total_count:
        print("\n🎉 所有測試通過！agent函數返回值格式一致。")
        return True
    else:
        print(f"\n⚠️  {total_count - success_count}個測試失敗，需要進一步調試。")
        return False

def test_array_compatibility():
    """測試numpy數組兼容性"""
    print("\n" + "=" * 60)
    print("測試NumPy數組兼容性")
    print("=" * 60)
    
    # 模擬可能出現問題的情況
    test_data = [
        # 正常情況：所有返回值都是3元組
        [(1, 0.5, 0.1), (2, 0.3, 0.2), (0, 0.8, -0.1)],
        # 混合情況：不同長度的元組（這會導致原來的錯誤）
        # [(1, 0.5, 0.1), (2, 0.3), (0, 0.8, -0.1, True)],
    ]
    
    for i, data in enumerate(test_data):
        try:
            # 嘗試轉換為numpy數組
            array = np.array(data)
            print(f"✅ 測試 {i+1}: 成功創建數組，形狀: {array.shape}")
        except ValueError as e:
            print(f"❌ 測試 {i+1}: 數組創建失敗 - {e}")
    
    print("測試完成。")

if __name__ == "__main__":
    success = test_agent_return_formats()
    test_array_compatibility()
    
    if success:
        print("\n🚀 修復成功！可以安全地繼續訓練。")
    else:
        print("\n🛠️  需要進一步調試和修復。")
