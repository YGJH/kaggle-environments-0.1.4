#!/usr/bin/env python3

import sys
import os
import torch
import numpy as np
import random
from train_connectx_rl_robust import ConnectXTrainer
from kaggle_environments import make

def diagnose_model_issues():
    """診斷模型的問題"""
    trainer = ConnectXTrainer("config.yaml")
    
    # 加載檢查點
    import glob
    checkpoints = glob.glob("checkpoints/*.pt")
    if checkpoints:
        latest = sorted(checkpoints)[-1]
        print(f"Loading checkpoint: {latest}")
        trainer.load_checkpoint(latest)
    
    print("=== 模型診斷 ===")
    
    # 1. 檢查不同局面的價值評估
    test_boards = [
        # 空棋盤
        [0] * 42,
        # 只有一個子在中央
        [0]*38 + [1] + [0]*3,
        # 一個簡單的三連線威脅
        [0]*35 + [1, 1, 1, 0] + [0]*3,
    ]
    
    for i, board in enumerate(test_boards):
        state = trainer.agent.encode_state(board, 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(trainer.agent.device)
        
        with torch.no_grad():
            trainer.agent.policy_net.eval()
            policy, value = trainer.agent.policy_net(state_tensor)
            
        print(f"\n局面 {i+1}:")
        print(f"棋盤: {board[:7]}")  # 只顯示最底行
        print(f"價值評估: {value.item():.3f}")
        print(f"策略分佈: {policy.cpu().numpy()[0]}")
    
    # 2. 檢查模型參數的分佈
    print("\n=== 參數診斷 ===")
    total_params = 0
    for name, param in trainer.agent.policy_net.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            print(f"{name}: {param.shape}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")
    
    print(f"總參數量: {total_params:,}")
    
    # 3. 檢查梯度流動
    print("\n=== 訓練一個小批次來檢查梯度 ===")
    
    # 創建一個簡單的訓練批次
    trainer.agent.memory.clear()
    
    # 添加一些正向獎勵的經驗
    state = trainer.agent.encode_state([0] * 42, 1)
    trainer.agent.store_transition(state, 3, 0.5, 1.0, True)  # 中央移動，正獎勵
    trainer.agent.store_transition(state, 0, 0.1, 0.0, False)  # 邊緣移動，零獎勵
    
    print(f"記憶體大小: {len(trainer.agent.memory)}")
    
    if len(trainer.agent.memory) >= 2:
        # 嘗試更新
        trainer.agent.update_policy()
        print("策略更新完成")

def create_better_config():
    """創建改進的配置"""
    config = """
agent:
  input_size: 126
  hidden_size: 1560
  num_layers: 32
  learning_rate: 0.0003  # 提高學習率
  weight_decay: 1e-5
  gamma: 0.99           # 降低折扣因子
  eps_clip: 0.15        # 略微降低裁剪
  k_epochs: 6
  entropy_coef: 0.03    # 增加熵係數促進探索
  value_coef: 0.3       # 大幅降低價值權重
  buffer_size: 50000
  min_batch_size: 512
  gae_lambda: 0.95

training:
  num_workers: 8
  episodes_per_update: 8
  max_episodes: 800000
  eval_frequency: 50    # 降低評估頻率
  eval_games: 10
  checkpoint_frequency: 1000
  use_tactical_opponent_in_rollout: true
  tactical_rollout_ratio: 0.3
  win_reward_scaling: 1.2    # 降低獎勵縮放
  loss_penalty_scaling: 0.8  # 降低懲罰
  player2_training_probability: 0.5
  parallel_rollout: true
  visualize_every: 200

evaluation:
  minimax_depth: 7
  weights:
    self_play: 0.4
    vs_minimax: 0.4
    vs_random: 0.2
"""
    
    with open("config_fixed.yaml", "w") as f:
        f.write(config)
    print("創建了修復的配置: config_fixed.yaml")

if __name__ == "__main__":
    diagnose_model_issues()
    print("\n" + "="*50)
    create_better_config()
