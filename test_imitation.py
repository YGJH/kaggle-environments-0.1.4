#!/usr/bin/env python3
"""
測試模仿學習系統
"""

import sys
import os
import torch
import numpy as np
import logging

# 添加當前目錄到路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from c4solver_wrapper import get_c4solver
from imitation_learning import ImitationLearner, load_config

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_c4solver():
    """測試C4Solver"""
    logger.info("🔧 測試C4Solver...")
    
    try:
        solver = get_c4solver()
        if solver is None:
            logger.error("❌ C4Solver初始化失敗")
            return False
        
        # 測試空局面
        empty_board = [0] * 42
        result = solver.evaluate_board(empty_board, analyze=True)
        logger.info(f"空局面分析: {result}")
        
        if result['valid']:
            logger.info("✅ C4Solver測試通過")
            return True
        else:
            logger.error("❌ C4Solver返回無效結果")
            return False
            
    except Exception as e:
        logger.error(f"❌ C4Solver測試失敗: {e}")
        return False

def test_model_loading():
    """測試模型載入"""
    logger.info("🧪 測試模型載入...")
    
    try:
        config = load_config()
        
        # 創建學習器
        learner = ImitationLearner(config)
        
        # 測試模型前向傳播
        dummy_input = torch.randn(1, 126).to(learner.device)
        
        with torch.no_grad():
            policy_probs, value = learner.model(dummy_input)
        
        logger.info(f"模型輸出形狀: policy={policy_probs.shape}, value={value.shape}")
        logger.info(f"策略概率: {policy_probs[0].cpu().numpy()}")
        logger.info(f"價值估計: {value[0].cpu().numpy()}")
        
        if policy_probs.shape == (1, 7) and value.shape == (1, 1):
            logger.info("✅ 模型測試通過")
            return True
        else:
            logger.error("❌ 模型輸出形狀不正確")
            return False
            
    except Exception as e:
        logger.error(f"❌ 模型測試失敗: {e}")
        return False

def test_sample_generation():
    """測試樣本生成"""
    logger.info("📊 測試樣本生成...")
    
    try:
        config = load_config()
        learner = ImitationLearner(config)
        
        # 生成小量樣本測試
        samples = learner.dataset.generate_training_samples(10)
        
        if len(samples) > 0:
            sample = samples[0]
            logger.info(f"樣本數量: {len(samples)}")
            logger.info(f"樣本結構: {list(sample.keys())}")
            logger.info(f"狀態形狀: {sample['state'].shape}")
            logger.info(f"動作分佈: {sample['action_dist']}")
            logger.info("✅ 樣本生成測試通過")
            return True
        else:
            logger.error("❌ 無法生成樣本")
            return False
            
    except Exception as e:
        logger.error(f"❌ 樣本生成測試失敗: {e}")
        return False

def test_training_step():
    """測試訓練步驟"""
    logger.info("🏋️ 測試訓練步驟...")
    
    try:
        config = load_config()
        # 減少樣本數量以加快測試
        config['training']['num_samples'] = 100
        config['training']['num_epochs'] = 2
        config['training']['batch_size'] = 32
        
        learner = ImitationLearner(config)
        
        # 生成少量樣本
        samples = learner.dataset.generate_training_samples(100)
        
        if len(samples) == 0:
            logger.error("❌ 無法生成訓練樣本")
            return False
        
        # 執行一個訓練步驟
        train_loss, train_acc = learner.train_epoch(samples, 32)
        
        logger.info(f"訓練損失: {train_loss:.4f}")
        logger.info(f"訓練準確率: {train_acc:.3f}")
        
        if train_loss > 0 and 0 <= train_acc <= 1:
            logger.info("✅ 訓練步驟測試通過")
            return True
        else:
            logger.error("❌ 訓練指標異常")
            return False
            
    except Exception as e:
        logger.error(f"❌ 訓練步驟測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    logger.info("🎯 模仿學習系統測試套件")
    logger.info("=" * 50)
    
    tests = [
        ("C4Solver", test_c4solver),
        ("模型載入", test_model_loading),
        ("樣本生成", test_sample_generation),
        ("訓練步驟", test_training_step)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n>>> 執行測試: {test_name}")
        try:
            if test_func():
                passed += 1
            logger.info("-" * 30)
        except Exception as e:
            logger.error(f"❌ 測試 {test_name} 崩潰: {e}")
            logger.info("-" * 30)
    
    logger.info(f"\n📊 測試結果: {passed}/{total} 通過")
    
    if passed == total:
        logger.info("🎉 所有測試通過! 模仿學習系統準備就緒。")
        logger.info("\n🚀 可以開始模仿學習預訓練:")
        logger.info("   python imitation_learning.py")
    else:
        logger.info("⚠️ 部分測試失敗，請檢查系統配置。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
