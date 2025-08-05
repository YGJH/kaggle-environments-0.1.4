#!/usr/bin/env python3
"""
測試評估系統功能
"""
import sys
import os
import yaml
import logging

# 添加項目路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_connectx_rl_robust import ConnectXTrainer

def test_evaluation_modes():
    """測試所有評估模式"""
    print("🧪 測試評估系統")
    print("=" * 50)
    
    # 載入配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 測試各種評估模式
    evaluation_modes = ['random', 'comprehensive', 'detailed', 'minimax']
    
    for mode in evaluation_modes:
        print(f"\n🎯 測試評估模式: {mode}")
        print("-" * 30)
        
        try:
            # 更新配置
            config['evaluation']['mode'] = mode
            config['evaluation']['num_games'] = 5  # 減少遊戲數量以加快測試
            
            # 創建新的訓練器實例
            trainer = ConnectXTrainer(config)
            
            # 運行評估（沒有模型時會使用隨機智能體）
            win_rate = trainer.evaluate_model(model=None, episode=0)
            
            print(f"✅ {mode} 模式評估完成，勝率: {win_rate:.3f}")
            
        except Exception as e:
            print(f"❌ {mode} 模式測試失敗: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🎉 評估系統測試完成!")

def show_evaluation_help():
    """顯示評估選項說明"""
    help_text = """
📖 評估模式說明
===============

🎲 random: 隨機對手評估
   - 對手使用完全隨機策略
   - 快速評估，適合初期訓練
   - 勝率通常較高

🔄 comprehensive: 綜合評估
   - 多種不同強度的對手
   - 平衡速度和準確性
   - 推薦用於一般訓練

🔍 detailed: 詳細評估
   - 包含統計分析
   - 提供詳細的性能指標
   - 適合深入分析

🧠 minimax: Minimax對手評估
   - 對抗強化的傳統AI
   - 最具挑戰性
   - 適合評估模型上限

配置方法:
在 config.yaml 中設置:
evaluation:
  mode: "comprehensive"  # 選擇評估模式
  num_games: 100        # 評估遊戲數量
  
使用方法:
python train_connectx_rl_robust.py --eval-mode detailed
"""
    print(help_text)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            test_evaluation_modes()
        elif sys.argv[1] == "--help-eval":
            show_evaluation_help()
        else:
            print("使用 --test 運行評估測試")
            print("使用 --help-eval 查看評估選項說明")
    else:
        test_evaluation_modes()
