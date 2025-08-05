#!/usr/bin/env python3
"""
測試新的評估系統
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_connectx_rl_robust import ConnectXTrainer
import yaml

def test_evaluation_modes():
    """測試不同的評估模式"""
    
    print("🧪 測試ConnectX評估系統")
    print("=" * 50)
    
    # 載入預訓練模型
    if not os.path.exists("config.yaml"):
        print("❌ 請先確保config.yaml存在")
        return
    
    if not os.path.exists("checkpoints"):
        print("❌ 沒有找到預訓練模型，請先訓練模型")
        return
    
    # 創建訓練器
    trainer = ConnectXTrainer("config.yaml")
    
    # 載入最佳模型
    checkpoints = [f for f in os.listdir("checkpoints") if f.startswith("best_model")]
    if checkpoints:
        import torch
        best_checkpoint = f"checkpoints/{sorted(checkpoints)[-1]}"
        print(f"📥 載入模型: {best_checkpoint}")
        
        checkpoint = torch.load(best_checkpoint, map_location=trainer.agent.device)
        trainer.agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
        print("✅ 模型載入成功")
    else:
        print("⚠️ 沒有找到最佳模型，使用隨機初始化的模型")
    
    # 測試不同評估模式
    test_games = 20  # 測試用較少遊戲數
    
    print("\n🎯 評估模式 1: 對隨機對手")
    random_win_rate = trainer.evaluate_against_random(test_games)
    print(f"勝率: {random_win_rate:.3f}")
    
    print("\n🤖 評估模式 2: 對Minimax對手")
    try:
        minimax_win_rate = trainer.evaluate_against_minimax(test_games // 2)
        print(f"勝率: {minimax_win_rate:.3f}")
    except Exception as e:
        print(f"錯誤: {e}")
    
    print("\n🔄 評估模式 3: 自對弈")
    try:
        self_play_score = trainer.evaluate_self_play(test_games // 2)
        print(f"平衡分數: {self_play_score:.3f}")
    except Exception as e:
        print(f"錯誤: {e}")
    
    print("\n📊 評估模式 4: 詳細指標")
    try:
        metrics = trainer.evaluate_with_metrics(test_games)
        print(f"勝率: {metrics['win_rate']:.3f}")
        print(f"平均步數: {metrics['avg_game_length']:.1f}")
        print(f"快速獲勝: {metrics['quick_wins']}")
        print(f"長遊戲獲勝: {metrics['comeback_wins']}")
    except Exception as e:
        print(f"錯誤: {e}")
    
    print("\n🏆 評估模式 5: 綜合評估")
    try:
        comprehensive = trainer.evaluate_comprehensive(test_games)
        print(f"綜合分數: {comprehensive['comprehensive_score']:.3f}")
        print(f"  vs 隨機: {comprehensive['vs_random']:.3f}")
        print(f"  vs Minimax: {comprehensive['vs_minimax']:.3f}")
        print(f"  自對弈: {comprehensive['self_play']:.3f}")
    except Exception as e:
        print(f"錯誤: {e}")
    
    print("\n✅ 評估測試完成！")

def show_evaluation_options():
    """顯示評估選項說明"""
    print("\n📖 評估模式說明")
    print("=" * 50)
    
    print("🎲 random: 對隨機對手評估")
    print("   - 最基礎的評估方式")
    print("   - 適合初期訓練監控")
    
    print("\n🤖 minimax: 對Minimax算法評估")
    print("   - 測試對策略性對手的表現")
    print("   - 更有挑戰性的評估")
    
    print("\n📊 detailed: 詳細指標評估")
    print("   - 提供豐富的統計信息")
    print("   - 包括遊戲長度、快速獲勝等")
    
    print("\n🏆 comprehensive: 綜合評估")
    print("   - 結合多種對手的評估")
    print("   - 最全面的性能評估")
    
    print("\n⚙️ 配置方式:")
    print("在config.yaml中設置:")
    print("evaluation:")
    print("  mode: comprehensive  # 選擇評估模式")
    print("  weights:")
    print("    vs_random: 0.4")
    print("    vs_minimax: 0.4") 
    print("    self_play: 0.2")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="測試評估系統")
    parser.add_argument("--test", action="store_true", help="運行評估測試")
    parser.add_argument("--help-eval", action="store_true", help="顯示評估選項說明")
    
    args = parser.parse_args()
    
    if args.help_eval:
        show_evaluation_options()
    elif args.test:
        test_evaluation_modes()
    else:
        print("使用 --test 運行評估測試")
        print("使用 --help-eval 查看評估選項說明")
