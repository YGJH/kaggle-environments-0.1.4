#!/usr/bin/env python3
"""
快速測試訓練系統是否正常工作
"""
import sys
import os
import yaml

# 添加項目路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_connectx_rl_robust import ConnectXTrainer

def quick_training_test():
    """快速訓練測試"""
    print("🚀 快速訓練系統測試")
    print("=" * 40)
    
    # 載入並修改配置進行快速測試
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 修改配置以進行快速測試
    config['training']['max_episodes'] = 5  # 只訓練5個回合
    config['training']['eval_frequency'] = 2  # 每2回合評估一次
    config['evaluation']['num_games'] = 3  # 評估只進行3局
    config['evaluation']['mode'] = 'random'  # 使用最快的評估模式
    
    print("🔧 測試配置:")
    print(f"  最大回合: {config['training']['max_episodes']}")
    print(f"  評估頻率: {config['training']['eval_frequency']}")
    print(f"  評估模式: {config['evaluation']['mode']}")
    print(f"  評估局數: {config['evaluation']['num_games']}")
    
    try:
        trainer = ConnectXTrainer(config)
        print("\n✅ 訓練器初始化成功")
        
        # 測試單次自對弈
        print("🎮 測試自對弈...")
        reward, length = trainer.self_play_episode()
        print(f"  遊戲結果: 獎勵={reward}, 長度={length}")
        
        # 測試評估
        print("📊 測試評估...")
        win_rate = trainer.evaluate_model()
        print(f"  勝率: {win_rate:.3f}")
        
        # 簡短訓練測試
        print("🏋️ 開始簡短訓練...")
        trainer.train()
        
        print("\n🎉 所有測試通過！系統運行正常。")
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_training_test()
