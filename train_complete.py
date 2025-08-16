#!/usr/bin/env python3
"""
Connect4 模仿學習 + 強化學習完整訓練流程
一鍵啟動腳本
"""

import sys
import os
import subprocess
import time
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_requirements():
    """檢查運行要求"""
    logger.info("🔍 檢查系統要求...")
    
    # 檢查C4Solver
    if not os.path.exists('./c4solver'):
        logger.error("❌ 找不到 c4solver 可執行檔")
        logger.info("請確保 c4solver 在當前目錄下且可執行")
        return False
    
    # 檢查Python模塊
    try:
        import torch
        import numpy
        import yaml
        from kaggle_environments import make
        logger.info("✅ 所有依賴模塊都已安裝")
    except ImportError as e:
        logger.error(f"❌ 缺少依賴模塊: {e}")
        logger.info("請運行: pip install torch numpy pyyaml kaggle-environments")
        return False
    
    return True

def run_imitation_pretraining():
    """運行模仿學習預訓練"""
    logger.info("🎯 開始模仿學習預訓練...")
    
    try:
        # 先運行測試
        logger.info("運行系統測試...")
        result = subprocess.run([sys.executable, 'test_imitation.py'], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning("系統測試有問題，但繼續進行訓練...")
            logger.info(f"測試輸出: {result.stdout}")
            logger.info(f"測試錯誤: {result.stderr}")
        
        # 運行模仿學習
        logger.info("開始模仿學習訓練...")
        start_time = time.time()
        
        result = subprocess.run([sys.executable, 'imitation_learning.py'], 
                              capture_output=False, text=True)
        
        training_time = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ 模仿學習完成! 用時: {training_time/60:.1f} 分鐘")
            return True
        else:
            logger.error("❌ 模仿學習失敗")
            return False
            
    except Exception as e:
        logger.error(f"❌ 模仿學習出錯: {e}")
        return False

def setup_rl_config():
    """設置強化學習配置以使用預訓練模型"""
    logger.info("⚙️ 配置強化學習使用預訓練模型...")
    
    config_lines = []
    
    # 檢查是否存在預訓練模型
    pretrained_models = [
        'imitation_pretrained_model_best.pt',
        'imitation_pretrained_model.pt'
    ]
    
    model_path = None
    for path in pretrained_models:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        logger.warning("⚠️ 找不到預訓練模型文件")
        logger.info("強化學習將使用隨機初始化")
        return False
    
    logger.info(f"✅ 找到預訓練模型: {model_path}")
    
    # 創建RL配置更新腳本
    update_script = f"""
# 自動生成的配置更新
# 在 train_connectx_rl_robust.py 的config中添加以下行：
# 'pretrained_model_path': '{model_path}'

import os
import re

config_file = 'train_connectx_rl_robust.py'
if os.path.exists(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找config定義
    if "'input_size':" in content and "'pretrained_model_path'" not in content:
        # 添加預訓練模型路徑
        content = content.replace(
            "'input_size': 126,",
            "'input_size': 126,\\n        'pretrained_model_path': '{model_path}',"
        )
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ 已自動配置強化學習使用預訓練模型")
    else:
        print("⚠️ 無法自動配置，請手動添加 'pretrained_model_path': '{model_path}' 到config中")
else:
    print("❌ 找不到 train_connectx_rl_robust.py")
"""
    
    with open('temp_config_update.py', 'w') as f:
        f.write(update_script)
    
    try:
        subprocess.run([sys.executable, 'temp_config_update.py'])
        os.remove('temp_config_update.py')
    except Exception:
        pass
    
    return True

def run_rl_training():
    """運行強化學習訓練"""
    logger.info("🏋️ 開始強化學習訓練...")
    
    try:
        # 運行強化學習
        logger.info("啟動PPO強化學習訓練...")
        logger.info("(這個過程會持續很長時間，請耐心等待)")
        
        result = subprocess.run([sys.executable, 'train_connectx_rl_robust.py'])
        
        if result.returncode == 0:
            logger.info("✅ 強化學習訓練完成!")
            return True
        else:
            logger.error("❌ 強化學習訓練失敗")
            return False
            
    except KeyboardInterrupt:
        logger.info("⏹️ 用戶中斷訓練")
        return False
    except Exception as e:
        logger.error(f"❌ 強化學習出錯: {e}")
        return False

def main():
    """主函數"""
    logger.info("🚀 Connect4 AI 完整訓練流程")
    logger.info("=" * 60)
    logger.info("此腳本將依次執行:")
    logger.info("1. 檢查系統要求")
    logger.info("2. 模仿學習預訓練 (使用C4Solver)")
    logger.info("3. 配置強化學習")
    logger.info("4. PPO強化學習訓練")
    logger.info("=" * 60)
    
    # 詢問用戶確認
    response = input("是否開始完整訓練流程? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        logger.info("❌ 用戶取消訓練")
        return 1
    
    # 步驟1: 檢查要求
    if not check_requirements():
        logger.error("❌ 系統要求不滿足，請先解決依賴問題")
        return 1
    
    # 步驟2: 模仿學習預訓練
    logger.info("\n" + "="*40)
    logger.info("第一階段：模仿學習預訓練")
    logger.info("="*40)
    
    if not run_imitation_pretraining():
        logger.error("❌ 模仿學習失敗，無法繼續")
        return 1
    
    # 步驟3: 配置強化學習
    logger.info("\n" + "="*40)
    logger.info("第二階段：配置強化學習")
    logger.info("="*40)
    
    setup_rl_config()
    
    # 詢問是否繼續RL訓練
    response = input("\n模仿學習完成！是否繼續強化學習訓練? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        logger.info("✅ 模仿學習階段完成。您可以稍後手動運行強化學習。")
        logger.info("運行強化學習: python train_connectx_rl_robust.py")
        return 0
    
    # 步驟4: 強化學習訓練
    logger.info("\n" + "="*40)
    logger.info("第三階段：強化學習訓練")
    logger.info("="*40)
    
    if run_rl_training():
        logger.info("\n🎉 完整訓練流程成功完成!")
        logger.info("您的Connect4 AI已經準備就緒!")
    else:
        logger.error("\n❌ 強化學習失敗")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\n⏹️ 訓練被用戶中斷")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n💥 意外錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
