#!/usr/bin/env python3
"""
測試戰術開局對手是否正確實現：
1. 強制先手 (player 1)
2. 開局序列 3->4->2
3. 然後使用戰術邏輯 (win -> block -> safe -> random)
"""

import train_connectx_rl_robust
from kaggle_environments import make
import torch

def test_tactical_opening():
    trainer = train_connectx_rl_robust.ConnectXTrainer('config.yaml')
    
    print("🎯 測試戰術開局對手")
    print("=" * 50)
    
    # 測試多局遊戲，觀察開局模式
    opening_moves = []
    for game_num in range(5):
        print(f"\n遊戲 {game_num + 1}:")
        
        env = make('connectx', debug=False)
        env.reset()
        move_count = 0
        
        with torch.no_grad():
            trainer.agent.policy_net.eval()
            while not env.done and move_count < 10:  # 只看前10步
                actions = []
                
                for p in range(2):
                    if env.state[p]['status'] == 'ACTIVE':
                        board, mark = trainer.agent.extract_board_and_mark(env.state, p)
                        valid = trainer.agent.get_valid_actions(board)
                        
                        if p == 0:  # 戰術開局對手 (強制先手)
                            action = trainer._tactical_random_opening_agent(board, mark, valid)
                            if move_count < 6:  # 記錄前3步對手動作
                                opening_moves.append(action)
                            print(f"  對手 (Player {p+1}, mark={mark}) 選擇: {action}")
                        else:  # 訓練Agent (後手)
                            state = trainer.agent.encode_state(board, mark)
                            action, _, _ = trainer.agent.select_action(state, valid, training=False)
                            action = int(action)
                            print(f"  Agent (Player {p+1}, mark={mark}) 選擇: {action}")
                        
                        actions.append(action)
                    else:
                        actions.append(0)
                
                try:
                    env.step(actions)
                except Exception:
                    break
                
                move_count += 1
    
    print(f"\n📊 開局動作統計 (對手前幾步):")
    for i, move in enumerate(opening_moves[:15]):  # 顯示前5局的前3步
        game_num = i // 3 + 1
        step_num = i % 3 + 1
        print(f"  遊戲{game_num} 第{step_num}步: {move}")
    
    # 分析開局模式
    if len(opening_moves) >= 3:
        pattern_count = {}
        for i in range(0, len(opening_moves), 3):
            if i + 2 < len(opening_moves):
                pattern = tuple(opening_moves[i:i+3])
                pattern_count[pattern] = pattern_count.get(pattern, 0) + 1
        
        print(f"\n🎯 開局模式分析:")
        for pattern, count in pattern_count.items():
            print(f"  {pattern}: {count} 次")
        
        expected_pattern = (3, 4, 2)
        if expected_pattern in pattern_count:
            print(f"✅ 發現期望的開局模式 {expected_pattern}!")
        else:
            print(f"⚠️  未發現期望的開局模式 {expected_pattern}")

if __name__ == "__main__":
    test_tactical_opening()
