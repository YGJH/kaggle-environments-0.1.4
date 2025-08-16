#!/usr/bin/env python3
"""
ConnectX Human vs AI Game - Terminal Version
"""

import sys
import numpy as np
from kaggle_environments import make, utils
import os
import time

# 導入AI模型
try:
    submission = utils.read_file("submission.py")
    agent = utils.get_last_callable(submission)
except Exception as e:
    print(f"無法載入 AI 模型: {e}")
    print("請確認 submission.py 文件存在")
    sys.exit(1)

class ConnectXTerminal:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1  # 1 = 人類, 2 = AI
        self.game_over = False
        
    def clear_screen(self):
        """清除螢幕"""
        pass
        # os.system('cls' if os.name == 'nt' else 'clear')
        
    def print_board(self):
        """顯示棋盤"""
        print("\n" + "="*50)
        print("🎮 ConnectX - 人類 vs AI 對戰")
        print("="*50)
        
        # 顯示列編號
        print("   ", end="")
        for col in range(self.cols):
            print(f"  {col} ", end="")
        print("\n")
        
        # 顯示棋盤
        for row in range(self.rows):
            print(f"{row}  ", end="")
            for col in range(self.cols):
                if self.board[row][col] == 0:
                    print("| ⚪ ", end="")
                elif self.board[row][col] == 1:
                    print("| 🔴 ", end="")
                else:
                    print("| 🟡 ", end="")
            print("|")
        
        # 底部邊框
        print("   " + "+" + "--------"*self.cols + "+")
        print("   ", end="")
        for col in range(self.cols):
            print(f"  {col} ", end="")
        print()
        

    def print_status(self):
        """顯示遊戲狀態"""
        if self.game_over:
            return
            
        if self.current_player == 1:
            print("\n🔴 你的回合！請選擇列 (0-6): ", end="")
        else:
            print("\n🟡 AI 思考中...")
            
    def is_valid_move(self, col):
        """檢查移動是否有效"""
        return 0 <= col < self.cols and self.board[0][col] == 0
        
    def make_move(self, col, player):
        """在指定列放置棋子"""
        if not self.is_valid_move(col):
            return False
            
        for row in range(self.rows - 1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = player
                return True
        return False
        
    def check_win(self, player):
        """檢查是否有玩家獲勝"""
        # 檢查水平方向
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if all(self.board[row][col + i] == player for i in range(4)):
                    return True
        
        # 檢查垂直方向  
        for row in range(self.rows - 3):
            for col in range(self.cols):
                if all(self.board[row + i][col] == player for i in range(4)):
                    return True
        
        # 檢查對角線（左上到右下）
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if all(self.board[row + i][col + i] == player for i in range(4)):
                    return True
        
        # 檢查對角線（右上到左下）
        for row in range(self.rows - 3):
            for col in range(3, self.cols):
                if all(self.board[row + i][col - i] == player for i in range(4)):
                    return True
        
        return False
        
    def is_board_full(self):
        """檢查棋盤是否已滿"""
        return all(self.board[0][col] != 0 for col in range(self.cols))
        
    def human_move(self):
        """處理人類玩家移動"""
        while True:
            try:
                self.print_board()
                self.print_status()
                
                user_input = input().strip()
                
                # 檢查是否要退出
                if user_input.lower() in ['q', 'quit', 'exit']:
                    print("\n👋 感謝遊玩！")
                    return False
                    
                col = int(user_input)
                
                if not self.is_valid_move(col):
                    print(f"❌ 列 {col} 無效或已滿，請重新選擇！")
                    input("按 Enter 繼續...")
                    continue
                    
                if self.make_move(col, 1):
                    return True
                    
            except ValueError:
                print("❌ 請輸入有效的數字 (0-6) 或 'q' 退出")
                input("按 Enter 繼續...")
            except KeyboardInterrupt:
                print("\n\n👋 感謝遊玩！")
                return False
                
    def ai_move(self):
        """AI移動"""
        try:
            # 顯示思考狀態
            self.clear_screen()
            self.print_board()
            print("\n🟡 AI 思考中", end="")
            
            # 動畫效果
            for i in range(3):
                print(".", end="", flush=True)
                time.sleep(0.5)
            print()
            
            # 獲取AI移動
            obs = {
                'board': self.board.flatten().tolist(),
                'mark': 2
            }
            config = {'rows': self.rows, 'columns': self.cols, 'inarow': 4}
            
            move = agent(obs, config)
            
            # 驗證移動
            if not self.is_valid_move(move):
                # 如果AI返回無效移動，選擇第一個有效的列
                for col in range(self.cols):
                    if self.is_valid_move(col):
                        move = col
                        break
                else:
                    return False  # 無有效移動
            
            if self.make_move(move, 2):
                print(f"🟡 AI 選擇了列 {move}")
                time.sleep(1)
                return True
                
        except Exception as e:
            print(f"AI 出錯: {e}")
            # 回退到隨機有效移動
            valid_cols = [col for col in range(self.cols) if self.is_valid_move(col)]
            if valid_cols:
                move = np.random.choice(valid_cols)
                self.make_move(move, 2)
                print(f"🟡 AI 隨機選擇了列 {move}")
                time.sleep(1)
                return True
                
        return False
        
    def play_game(self):
        """主遊戲循環"""
        self.clear_screen()
        print("🎮 歡迎來到 ConnectX！")
        print("\n遊戲規則：")
        print("• 目標：連續四個棋子（水平、垂直或對角線）")
        print("• 🔴 你是紅色棋子")
        print("• 🟡 AI是黃色棋子")
        print("• 輸入列號 (0-6) 放置棋子")
        print("• 輸入 'q' 退出遊戲")
        print("\n按 Enter 開始遊戲...")
        input()
        
        while not self.game_over:
            self.clear_screen()
            
            if self.current_player == 1:
                # 人類回合
                if not self.human_move():
                    break
                    
                # 檢查人類是否獲勝
                if self.check_win(1):
                    self.clear_screen()
                    self.print_board()
                    print("\n🎉 恭喜！你贏了！")
                    self.game_over = True
                    break
                    
            else:
                # AI回合
                if not self.ai_move():
                    print("AI 無法移動")
                    break
                    
                # 檢查AI是否獲勝
                if self.check_win(2):
                    self.clear_screen()
                    self.print_board()
                    print("\n🤖 AI 獲勝！再接再厲！")
                    self.game_over = True
                    break
            
            # 檢查平局
            if self.is_board_full():
                self.clear_screen()
                self.print_board()
                print("\n🤝 平局！棋盤已滿。")
                self.game_over = True
                break
                
            # 切換玩家
            self.current_player = 3 - self.current_player
            
        # 詢問是否再玩一局
        if self.game_over:
            print("\n想再玩一局嗎？(y/n): ", end="")
            try:
                if input().strip().lower() in ['y', 'yes']:
                    self.restart_game()
                    self.play_game()
            except KeyboardInterrupt:
                pass
                
        print("\n👋 感謝遊玩！")
        
    def restart_game(self):
        """重新開始遊戲"""
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        self.game_over = False

def main():
    """主函數"""
    try:
        game = ConnectXTerminal()
        game.play_game()
    except Exception as e:
        print(f"遊戲出現錯誤: {e}")
        print("請確認 submission.py 文件存在且正確。")

if __name__ == "__main__":
    main()