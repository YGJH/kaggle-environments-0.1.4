#!/usr/bin/env python3
"""
調試ConnectX環境狀態結構
"""

from kaggle_environments import make
import json

def debug_env_structure():
    """調試環境狀態結構"""
    print("🔍 調試ConnectX環境狀態結構")
    print("=" * 50)
    
    # 創建環境
    env = make("connectx", debug=True)
    env.reset()
    
    print("📊 初始環境狀態:")
    print(f"環境類型: {type(env)}")
    print(f"環境狀態類型: {type(env.state)}")
    print(f"環境狀態長度: {len(env.state)}")
    
    # 檢查環境狀態結構
    for i, player_state in enumerate(env.state):
        print(f"\n🎮 玩家 {i+1} 狀態:")
        print(f"  類型: {type(player_state)}")
        print(f"  鍵: {list(player_state.keys()) if isinstance(player_state, dict) else 'Not a dict'}")
        
        if 'observation' in player_state:
            obs = player_state['observation']
            print(f"  觀察類型: {type(obs)}")
            print(f"  觀察鍵: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dict'}")
            
            # 檢查每個鍵的內容
            for key, value in obs.items():
                if key == 'board' and isinstance(value, (list, tuple)):
                    print(f"    {key}: 長度={len(value)}, 前10元素={value[:10]}")
                else:
                    print(f"    {key}: {value}")
    
    # 檢查環境是否有其他屬性
    print(f"\n🔧 環境屬性:")
    for attr in ['observation', 'board', 'state', 'done']:
        if hasattr(env, attr):
            value = getattr(env, attr)
            print(f"  {attr}: {type(value)}")
            if attr == 'observation' and hasattr(value, 'keys'):
                print(f"    觀察鍵: {list(value.keys())}")
    
    # 執行一步動作看看狀態變化
    print(f"\n⚡ 執行動作後的狀態:")
    try:
        env.step([3, 2])  # 玩家1選擇列3，玩家2選擇列2
        
        for i, player_state in enumerate(env.state):
            if 'observation' in player_state:
                obs = player_state['observation']
                print(f"  玩家 {i+1} 觀察鍵: {list(obs.keys())}")
                if 'board' in obs:
                    board = obs['board']
                    print(f"    棋盤長度: {len(board)}")
                    # 顯示非零位置
                    non_zero = [(idx, val) for idx, val in enumerate(board) if val != 0]
                    print(f"    非零位置: {non_zero}")
                    
                    # 顯示為6x7格式
                    print("    棋盤狀態:")
                    for row in range(6):
                        row_data = board[row*7:(row+1)*7]
                        print(f"      {row_data}")
    except Exception as e:
        print(f"執行動作時出錯: {e}")

def test_alternative_access():
    """測試其他方式訪問棋盤"""
    print("\n🧪 測試其他訪問方式")
    print("=" * 30)
    
    env = make("connectx", debug=True)
    
    # 嘗試不同的方式創建智能體來獲取觀察
    def dummy_agent(obs, config):
        print(f"智能體接收到的觀察:")
        print(f"  類型: {type(obs)}")
        if hasattr(obs, '__dict__'):
            print(f"  屬性: {list(obs.__dict__.keys())}")
        if hasattr(obs, 'keys'):
            print(f"  鍵: {list(obs.keys())}")
        
        # 檢查觀察內容
        for attr in ['board', 'mark']:
            if hasattr(obs, attr):
                value = getattr(obs, attr)
                print(f"  {attr}: {value}")
        
        return 3  # 選擇中間列
    
    try:
        # 運行一個智能體來看看觀察格式
        env.run([dummy_agent, dummy_agent], num_episodes=1)
    except Exception as e:
        print(f"運行智能體時出錯: {e}")

if __name__ == "__main__":
    debug_env_structure()
    test_alternative_access()
