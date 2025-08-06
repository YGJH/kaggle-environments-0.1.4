#!/usr/bin/env python3
"""
測試戰術對手訓練功能
"""

import sys
sys.path.append('.')

from train_connectx_rl_robust import ConnectXTrainer
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tactical_training():
    """測試戰術對手訓練"""
    try:
        # 創建訓練器
        trainer = ConnectXTrainer()
        print("✅ 成功創建 ConnectXTrainer")
        
        # 測試戰術對手訓練
        print("🔄 測試戰術對手訓練...")
        reward, episode_length = trainer.play_against_tactical_opponent()
        print(f"✅ 戰術對手訓練成功！回合長度: {episode_length}, 獎勵: {reward}")
        
        # 測試隨機對手訓練
        print("🔄 測試隨機對手訓練...")  
        reward2, episode_length2 = trainer.play_against_random_agent()
        print(f"✅ 隨機對手訓練成功！回合長度: {episode_length2}, 獎勵: {reward2}")
        
        print("🎉 所有測試通過！")
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tactical_training()
