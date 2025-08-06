#!/usr/bin/env python3
"""
測試可視化功能的腳本
"""

import sys
import os

# 添加當前目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from train_connectx_rl_robust import ConnectXTrainer
    import logging
    
    # 設置日誌
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    def test_visualization():
        """測試可視化功能"""
        try:
            # 創建一個簡單的配置
            config = {
                'model': {
                    'hidden_size': 256,
                    'num_layers': 3,
                    'dropout': 0.1
                },
                'training': {
                    'learning_rate': 0.001,
                    'batch_size': 256,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'epsilon': 0.2,
                    'value_coef': 0.5,
                    'entropy_coef': 0.01,
                    'max_grad_norm': 0.5,
                    'ppo_epochs': 4,
                    'rollout_length': 256,
                    'eval_frequency': 100,
                    'eval_games': 20,
                    'checkpoint_frequency': 1000,
                    'early_stopping_patience': 5000,
                    'lr_scheduler_patience': 500,
                    'lr_scheduler_factor': 0.8,
                    'min_lr': 1e-6,
                    'curriculum_start_episode': 1000,
                    'minimax_start_episode': 3000,
                    'random_action_prob': 0.1,
                    'random_action_decay': 0.99,
                    'min_random_action_prob': 0.01
                }
            }
            
            # 創建訓練器
            logger.info("創建訓練器...")
            trainer = ConnectXTrainer(config)
            
            # 測試不同對手類型的可視化
            opponent_types = ["random", "self_play", "minimax"]
            
            for opponent_type in opponent_types:
                logger.info(f"測試對戰 {opponent_type} 對手的可視化...")
                try:
                    trainer.demo_game_with_visualization(opponent_type)
                    logger.info(f"✓ {opponent_type} 對手可視化成功")
                except Exception as e:
                    logger.error(f"✗ {opponent_type} 對手可視化失敗: {e}")
            
            logger.info("可視化測試完成！")
            
        except ImportError as e:
            logger.error(f"導入模塊失敗: {e}")
            logger.info("請確保已安裝必要的依賴包：matplotlib")
            return False
        except Exception as e:
            logger.error(f"測試過程中出現錯誤: {e}")
            return False
        
        return True
    
    if __name__ == "__main__":
        logger.info("開始測試可視化功能...")
        success = test_visualization()
        
        if success:
            logger.info("🎉 所有測試通過！")
        else:
            logger.error("❌ 測試失敗")
            sys.exit(1)

except ImportError as e:
    print(f"無法導入訓練模塊: {e}")
    print("請確保 train_connectx_rl_robust.py 文件存在且可以正常導入")
