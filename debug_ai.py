#!/usr/bin/env python3
"""調試AI智能體問題"""

import os
from kaggle_environments import make

def load_ai():
    """載入AI智能體"""
    with open("submission.py", "r") as f:
        submission_code = f.read()
    
    namespace = {}
    exec(submission_code, namespace)
    
    return namespace['my_agent']

def debug_agent(obs, config):
    """調試智能體，顯示觀察結構"""
    print(f"觀察結構: {type(obs)}")
    print(f"觀察內容: {obs}")
    print(f"配置: {config}")
    
    # 嘗試訪問board
    if hasattr(obs, 'board'):
        print(f"board: {obs.board}")
    if hasattr(obs, 'mark'):
        print(f"mark: {obs.mark}")
    
    # 返回隨機有效動作
    import random
    valid_actions = [col for col in range(7) if obs.board[col] == 0]
    return random.choice(valid_actions)

print("🔍 調試AI智能體")

# 創建環境
env = make("connectx", debug=True)

# 使用調試智能體
env.run([debug_agent, debug_agent])

print("調試完成")
