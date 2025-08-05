#!/usr/bin/env python3
"""
簡化的 ConnectX RL 訓練腳本測試
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import logging
from datetime import datetime

try:
    import yaml
except ImportError:
    print("PyYAML not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyYAML"])
    import yaml

from kaggle_environments import make, evaluate

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_environment():
    """測試 ConnectX 環境"""
    logger.info("Testing ConnectX environment...")
    
    env = make("connectx", debug=False)
    env.reset()
    
    logger.info(f"Initial state: {env.state}")
    logger.info(f"Done: {env.done}")
    
    # 隨機遊戲
    moves = 0
    while not env.done and moves < 20:
        actions = []
        for player_idx in range(2):
            if env.state[player_idx]['status'] == 'ACTIVE':
                board = env.state[player_idx]['observation']['board']
                valid_actions = [col for col in range(7) if board[col] == 0]
                action = random.choice(valid_actions)
                actions.append(action)
            else:
                actions.append(0)
        
        env.step(actions)
        moves += 1
        logger.info(f"Move {moves}: Actions {actions}, Done: {env.done}")
    
    logger.info(f"Final state: {env.state}")
    return True

def test_neural_network():
    """測試神經網路"""
    logger.info("Testing neural network...")
    
    class SimpleConnectXNet(nn.Module):
        def __init__(self, input_size=126, hidden_size=256):
            super(SimpleConnectXNet, self).__init__()
            
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 7)  # 7 columns
            )
            
        def forward(self, x):
            return self.network(x)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    net = SimpleConnectXNet().to(device)
    
    # 測試輸入
    dummy_input = torch.randn(1, 126).to(device)
    output = net(dummy_input)
    
    logger.info(f"Network output shape: {output.shape}")
    logger.info(f"Network output: {output}")
    
    return True

def test_training_loop():
    """測試訓練循環"""
    logger.info("Testing training loop...")
    
    env = make("connectx", debug=False)
    
    # 簡單的隨機智能體
    def random_agent():
        def agent_func(obs, config):
            valid_actions = [col for col in range(7) if obs.board[col] == 0]
            return random.choice(valid_actions)
        return agent_func
    
    # 運行幾局遊戲
    for game in range(3):
        env.reset()
        moves = 0
        logger.info(f"Starting game {game + 1}")
        
        while not env.done and moves < 42:
            actions = []
            for player_idx in range(2):
                if env.state[player_idx]['status'] == 'ACTIVE':
                    board = env.state[player_idx]['observation']['board']
                    valid_actions = [col for col in range(7) if board[col] == 0]
                    action = random.choice(valid_actions)
                    actions.append(action)
                else:
                    actions.append(0)
            
            env.step(actions)
            moves += 1
        
        logger.info(f"Game {game + 1} finished in {moves} moves. Winner: {env.state[0]['reward']}")
    
    return True

if __name__ == "__main__":
    logger.info("Starting ConnectX RL training script tests...")
    
    try:
        # 測試環境
        test_environment()
        
        # 測試神經網路
        test_neural_network()
        
        # 測試訓練循環
        test_training_loop()
        
        logger.info("All tests passed! Ready for full training.")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
