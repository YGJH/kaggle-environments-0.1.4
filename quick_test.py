#!/usr/bin/env python3
"""快速測試AI智能體"""

import os
from kaggle_environments import make

def load_ai():
    """載入AI智能體"""
    with open("submission.py", "r") as f:
        submission_code = f.read()
    
    namespace = {}
    exec(submission_code, namespace)
    
    # 尋找智能體函數
    for name, obj in namespace.items():
        if callable(obj) and name != 'my_agent' and not name.startswith('_'):
            if hasattr(obj, '__code__') and obj.__code__.co_argcount >= 2:
                return obj
    
    if 'my_agent' in namespace:
        return namespace['my_agent']
    
    raise Exception("找不到AI智能體")

def random_agent(obs, config):
    import random
    valid_actions = [col for col in range(7) if obs.board[col] == 0]
    return random.choice(valid_actions)

print("🎯 快速測試AI vs 隨機對手")

# 載入AI
ai_agent = load_ai()
print("✅ AI載入成功")

# 創建環境並運行
env = make("connectx", debug=False)
env.run([ai_agent, random_agent])

# 檢查結果
if env.state[0].status == "DONE":
    if env.state[0].reward == 1:
        print("🎉 AI 獲勝！")
    elif env.state[0].reward == -1:
        print("😮 隨機對手獲勝！")
    else:
        print("🤝 平局！")
else:
    print(f"⚠️ 遊戲狀態: {env.state[0].status}")

print(f"遊戲步數: {len(env.steps)}")
