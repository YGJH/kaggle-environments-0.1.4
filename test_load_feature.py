#!/usr/bin/env python3
"""
測試模型載入功能
"""
import os
import sys
import yaml

# 添加項目路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_connectx_rl_robust import ConnectXTrainer

def test_load_functionality():
    """測試模型載入功能"""
    print("🧪 測試模型載入功能")
    print("=" * 50)
    
    # 載入配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 修改配置以進行快速測試
    config['training']['max_episodes'] = 3
    config['training']['eval_frequency'] = 1
    config['evaluation']['num_games'] = 3
    
    print("🔧 第一階段：創建並保存模型")
    
    # 創建訓練器並訓練幾個回合
    trainer1 = ConnectXTrainer(config)
    
    # 模擬一些訓練歷史
    for i in range(3):
        reward, length = trainer1.self_play_episode()
        trainer1.episode_rewards.append(reward)
        print(f"  回合 {i}: 獎勵={reward}, 長度={length}")
    
    # 模擬一些勝率歷史
    trainer1.win_rates = [0.5, 0.6, 0.7]
    
    # 保存檢查點
    test_checkpoint = "test_checkpoint.pt"
    trainer1.save_checkpoint(test_checkpoint)
    print(f"✅ 模型已保存到: checkpoints/{test_checkpoint}")
    
    print("\n🔄 第二階段：載入模型並繼續訓練")
    
    # 創建新的訓練器並載入檢查點
    trainer2 = ConnectXTrainer(config)
    
    # 檢查載入前的狀態
    print(f"  載入前 - 回合歷史: {len(trainer2.episode_rewards)}")
    print(f"  載入前 - 勝率歷史: {len(trainer2.win_rates)}")
    
    # 載入檢查點
    checkpoint_path = f"checkpoints/{test_checkpoint}"
    if trainer2.load_checkpoint(checkpoint_path):
        print("✅ 檢查點載入成功")
        
        # 檢查載入後的狀態
        print(f"  載入後 - 回合歷史: {len(trainer2.episode_rewards)}")
        print(f"  載入後 - 勝率歷史: {len(trainer2.win_rates)}")
        print(f"  載入後 - 最佳勝率: {max(trainer2.win_rates) if trainer2.win_rates else 0:.3f}")
        
        # 驗證數據一致性
        if (len(trainer2.episode_rewards) == len(trainer1.episode_rewards) and
            len(trainer2.win_rates) == len(trainer1.win_rates)):
            print("✅ 訓練歷史載入正確")
        else:
            print("❌ 訓練歷史載入不正確")
    else:
        print("❌ 檢查點載入失敗")
    
    print("\n🧹 清理測試文件")
    # 清理測試文件
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"  已刪除: {checkpoint_path}")
    
    print("\n🎉 測試完成！")

def test_command_line_usage():
    """展示命令行使用方法"""
    print("\n📖 命令行使用說明")
    print("=" * 50)
    
    usage_examples = [
        "# 從頭開始訓練",
        "uv run train_connectx_rl_robust.py",
        "",
        "# 從最佳模型繼續訓練",
        "uv run train_connectx_rl_robust.py --load checkpoints/best_model_wr_0.880.pt",
        "",
        "# 從特定檢查點繼續訓練",
        "uv run train_connectx_rl_robust.py --load checkpoints/checkpoint_episode_2000.pt",
        "",
        "# 指定訓練回合數和載入模型",
        "uv run train_connectx_rl_robust.py --load checkpoints/best_model.pt --episodes 10000",
        "",
        "# 使用自定義配置和載入模型",
        "uv run train_connectx_rl_robust.py --config my_config.yaml --load my_model.pt"
    ]
    
    for line in usage_examples:
        print(line)

if __name__ == "__main__":
    test_load_functionality()
    test_command_line_usage()
