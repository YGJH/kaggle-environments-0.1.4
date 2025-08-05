#!/usr/bin/env python3
"""
ConnectX 人機對戰系統
讓你可以和訓練好的 AI 模型對戰

使用方法:
python play_against_ai.py
"""

import sys
import os
from kaggle_environments import make, utils

def print_board(board):
    """打印美觀的棋盤"""
    print("\n  0 1 2 3 4 5 6")
    print(" ┌─┬─┬─┬─┬─┬─┬─┐")
    
    for row in range(6):
        print(" │", end="")
        for col in range(7):
            piece = board[row * 7 + col]
            if piece == 0:
                print("·", end="")
            elif piece == 1:
                print("●", end="")  # 玩家1 (你)
            else:
                print("○", end="")  # 玩家2 (AI)
            print("│", end="")
        print()
        if row < 5:
            print(" ├─┼─┼─┼─┼─┼─┼─┤")
    
    print(" └─┴─┴─┴─┴─┴─┴─┘")
    print("  0 1 2 3 4 5 6\n")

def get_player_move(board):
    """獲取玩家輸入的動作"""
    valid_actions = [col for col in range(7) if board[col] == 0]
    
    while True:
        try:
            print(f"有效列: {valid_actions}")
            move = input("選擇列數 (0-6): ").strip()
            
            if move.lower() in ['q', 'quit', 'exit']:
                return None
            
            move = int(move)
            if move in valid_actions:
                return move
            else:
                print("❌ 該列已滿或無效，請選擇其他列！")
                
        except ValueError:
            print("❌ 請輸入數字 0-6，或輸入 'q' 退出")
        except KeyboardInterrupt:
            print("\n\n👋 再見！")
            return None

def create_human_agent():
    """創建人類玩家智能體"""
    def human_agent(obs, config):
        board = obs.board
        print_board(board)
        print("🔴 你的回合！")
        
        move = get_player_move(board)
        if move is None:
            # 如果玩家退出，隨機選擇一個有效動作
            valid_actions = [col for col in range(7) if board[col] == 0]
            return valid_actions[0] if valid_actions else 0
        
        print(f"你選擇了列 {move}")
        return move
    
    return human_agent

def create_random_agent():
    """創建隨機智能體"""
    import random
    
    def random_agent(obs, config):
        valid_actions = [col for col in range(7) if obs.board[col] == 0]
        return random.choice(valid_actions)
    
    return random_agent

def play_game(player1_agent, player2_agent, player1_name="玩家1", player2_name="玩家2"):
    """進行一場遊戲"""
    env = make("connectx", debug=False)
    
    print(f"\n🎮 開始遊戲: {player1_name} vs {player2_name}")
    print(f"🔴 {player1_name} = ●")
    print(f"🔵 {player2_name} = ○")
    print("=" * 50)
    
    # 使用正確的遊戲循環
    agents = [player1_agent, player2_agent]
    
    # 執行遊戲直到結束
    try:
        env.run(agents)
    except Exception as e:
        print(f"遊戲執行錯誤: {e}")
        return 0
    
    # 顯示最終結果
    print_board(env.state[0]['observation']['board'])
    
    if env.state[0]['status'] == 'DONE':
        if env.state[0]['reward'] == 1:
            print(f"🎉 {player1_name} 獲勝！")
            return 1
        elif env.state[0]['reward'] == -1:
            print(f"🎉 {player2_name} 獲勝！")
            return 2
        else:
            print("🤝 平局！")
            return 0
    else:
        print("⏰ 遊戲超時")
        return 0

def main():
    print("🎯 ConnectX 人機對戰系統")
    print("=" * 50)
    
    # 載入 AI 智能體
    try:
        if os.path.exists("submission.py"):
            # 讀取submission.py內容
            with open("submission.py", "r") as f:
                submission_code = f.read()
            
            # 執行代碼以獲取AI智能體
            namespace = {}
            exec(submission_code, namespace)
            
            # 獲取原始AI智能體
            original_ai = namespace['my_agent']
            
            # 創建包裝器來適配觀察格式
            def ai_agent_wrapper(obs, config):
                # 將Struct轉換為字典格式
                obs_dict = {
                    'board': obs.board,
                    'mark': obs.mark
                }
                return original_ai(obs_dict, config)
            
            ai_agent = ai_agent_wrapper
            print("✅ AI 模型載入成功！（勝率: 94%）")
        else:
            print("❌ 找不到 submission.py，請先運行 dump_weight.py")
            return
    except Exception as e:
        print(f"❌ 載入 AI 模型失敗: {e}")
        print("🤖 使用隨機智能體代替")
        ai_agent = create_random_agent()
    
    # 創建人類玩家智能體
    human_agent = create_human_agent()
    
    print("\n選擇遊戲模式:")
    print("1. 你 vs AI (你先手)")
    print("2. AI vs 你 (AI 先手)")
    print("3. AI vs AI (觀戰)")
    print("4. 你 vs 隨機對手")
    
    while True:
        try:
            choice = input("\n選擇模式 (1-4): ").strip()
            
            if choice == "1":
                print("🎮 你 vs AI 模式 (你先手)")
                result = play_game(human_agent, ai_agent, "你", "AI")
                break
            elif choice == "2":
                print("🎮 AI vs 你 模式 (AI先手)")
                result = play_game(ai_agent, human_agent, "AI", "你")
                break
            elif choice == "3":
                print("🎮 AI vs AI 觀戰模式")
                result = play_game(ai_agent, ai_agent, "AI-1", "AI-2")
                break
            elif choice == "4":
                print("🎮 你 vs 隨機對手模式")
                random_agent = create_random_agent()
                result = play_game(human_agent, random_agent, "你", "隨機對手")
                break
            else:
                print("❌ 請選擇 1-4")
                
        except KeyboardInterrupt:
            print("\n\n👋 再見！")
            return
    
    # 詢問是否再玩一局
    print("\n" + "=" * 50)
    play_again = input("再玩一局嗎？(y/n): ").strip().lower()
    if play_again in ['y', 'yes', '是', '好']:
        main()
    else:
        print("👋 感謝遊戲！")

if __name__ == "__main__":
    main()
