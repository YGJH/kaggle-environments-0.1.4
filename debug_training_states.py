#!/usr/bin/env python3
"""
調試訓練時的環境狀態問題
"""
import sys
import os
import yaml

# 添加項目路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_connectx_rl_robust import ConnectXTrainer
from kaggle_environments import make

def debug_training_states():
    """調試訓練過程中的環境狀態"""
    print("🔍 調試訓練時的環境狀態變化")
    print("=" * 50)
    
    # 載入配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    trainer = ConnectXTrainer(config)
    
    # 創建環境
    env = make("connectx", debug=True)
    env.reset()
    
    print(f"🎮 初始環境狀態:")
    print(f"  環境完成狀態: {env.done}")
    print(f"  狀態長度: {len(env.state)}")
    
    for i, player_state in enumerate(env.state):
        print(f"\n  玩家 {i} 狀態:")
        print(f"    鍵: {list(player_state.keys())}")
        print(f"    狀態: {player_state['status']}")
        
        if 'observation' in player_state:
            obs = player_state['observation']
            print(f"    觀察類型: {type(obs)}")
            
            if hasattr(obs, 'keys'):
                obs_keys = list(obs.keys())
            else:
                obs_keys = [attr for attr in dir(obs) if not attr.startswith('_')]
            
            print(f"    觀察鍵/屬性: {obs_keys}")
            
            # 嘗試提取棋盤和標記
            try:
                board, mark = trainer.agent.extract_board_and_mark(env.state, i)
                print(f"    ✅ 提取成功: 棋盤長度={len(board)}, 標記={mark}")
            except Exception as e:
                print(f"    ❌ 提取失敗: {e}")
    
    # 模擬幾步遊戲
    print(f"\n🎯 模擬遊戲步驟:")
    move_count = 0
    
    while not env.done and move_count < 5:
        print(f"\n  === 第 {move_count + 1} 步 ===")
        
        # 為活躍玩家生成動作
        actions = [3, 3]  # 都選擇中間列
        
        print(f"  執行動作: {actions}")
        env.step(actions)
        
        print(f"  遊戲完成: {env.done}")
        
        # 檢查每個玩家的狀態
        for i, player_state in enumerate(env.state):
            print(f"    玩家 {i}:")
            print(f"      狀態: {player_state['status']}")
            print(f"      獎勵: {player_state.get('reward', 'N/A')}")
            
            if 'observation' in player_state:
                obs = player_state['observation']
                
                if hasattr(obs, 'keys'):
                    obs_keys = list(obs.keys())
                else:
                    obs_keys = [attr for attr in dir(obs) if not attr.startswith('_')]
                
                print(f"      觀察鍵: {obs_keys}")
                
                # 嘗試提取狀態
                try:
                    board, mark = trainer.agent.extract_board_and_mark(env.state, i)
                    print(f"      ✅ 狀態提取: 棋盤長度={len(board)}, 標記={mark}")
                    
                    # 顯示棋盤狀態
                    if len(board) == 42:
                        non_zero = [(idx, val) for idx, val in enumerate(board) if val != 0]
                        if non_zero:
                            print(f"      非零位置: {non_zero[:5]}...")  # 只顯示前5個
                        
                except Exception as e:
                    print(f"      ❌ 狀態提取失敗: {e}")
        
        move_count += 1
    
    print(f"\n🏁 模擬完成")

if __name__ == "__main__":
    debug_training_states()
