#!/usr/bin/env python3
"""
ConnectX 預訓練腳本
修正數據載入格式問題
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from train_connectx_rl_robust import ConnectXTrainer, create_default_config
import yaml

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_state_action_dataset_fixed(file_path="connectx-state-action-value.txt"):
    """修正版本：載入狀態-動作價值數據集"""
    states = []
    action_values = []
    
    try:
        logger.info(f"載入訓練數據集: {file_path}")
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            try:
                # 找到第一個逗號的位置，分割棋盤狀態和動作價值
                comma_idx = line.find(',')
                if comma_idx == -1:
                    logger.warning(f"第 {line_idx + 1} 行格式錯誤，找不到逗號分隔符")
                    continue
                
                # 解析棋盤狀態 (逗號前的42個數字)
                board_part = line[:comma_idx]
                if len(board_part) != 42:
                    logger.warning(f"第 {line_idx + 1} 行棋盤狀態長度錯誤，期望42個字符，得到{len(board_part)}個")
                    continue
                
                board_state = [int(c) for c in board_part]
                
                # 解析動作價值 (逗號後的7個數字)
                action_part = line[comma_idx+1:]
                action_parts = action_part.split(',')
                if len(action_parts) != 7:
                    logger.warning(f"第 {line_idx + 1} 行動作價值數量錯誤，期望7個值，得到{len(action_parts)}個")
                    continue
                
                action_vals = [float(x) for x in action_parts]
                
                # 創建虛擬trainer來使用編碼功能
                from train_connectx_rl_robust import PPOAgent
                config = create_default_config()
                agent = PPOAgent(config['agent'])
                
                # 將棋盤狀態轉換為我們的編碼格式 (假設玩家1的視角)
                encoded_state = agent.encode_state(board_state, 1)
                
                states.append(encoded_state)
                action_values.append(action_vals)
                
                # 每10000行報告一次進度
                if (line_idx + 1) % 10000 == 0:
                    logger.info(f"已處理 {line_idx + 1} 行數據")
                
            except (ValueError, IndexError) as e:
                logger.warning(f"第 {line_idx + 1} 行解析錯誤: {e}")
                continue
        
        logger.info(f"成功載入 {len(states)} 個訓練樣本")
        return np.array(states), np.array(action_values)
        
    except FileNotFoundError:
        logger.error(f"找不到數據集文件: {file_path}")
        return None, None
    except Exception as e:
        logger.error(f"載入數據集時出錯: {e}")
        return None, None

def main():
    # 確保配置文件存在
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        logger.info(f"創建默認配置文件: {config_path}")
        config = create_default_config()
        config['agent']['supervised_learning_rate'] = 1e-4
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # 創建訓練器
    trainer = ConnectXTrainer(config_path)
    
    # 使用修正版本的數據載入函數
    logger.info("開始載入數據集...")
    states, action_values = load_state_action_dataset_fixed("connectx-state-action-value.txt")
    
    if states is not None and action_values is not None:
        logger.info("數據載入成功，開始預訓練...")
        success = trainer.supervised_pretrain(
            states, action_values, 
            epochs=50,
            batch_size=128
        )
        
        if success:
            logger.info("預訓練成功完成！")
            # 評估預訓練後的模型
            eval_score = trainer.evaluate_model()
            logger.info(f"預訓練後評估分數: {eval_score:.3f}")
            
            # 保存預訓練模型
            trainer.save_checkpoint("pretrained_final.pt")
            logger.info("預訓練模型已保存為 pretrained_final.pt")
        else:
            logger.error("預訓練失敗！")
    else:
        logger.error("無法載入預訓練數據集！")

if __name__ == "__main__":
    main()
