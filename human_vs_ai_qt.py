#!/usr/bin/env python3
"""
ConnectX Human vs AI Game - PyQt5 GUI Version

使用方法：
- 點擊列按鈕放置棋子
- 人類是紅色棋子，AI是黃色棋子
- 目標是連續四個棋子（水平、垂直或對角線）
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QPushButton, QLabel, 
                            QMessageBox, QFrame, QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor
from kaggle_environments import make, utils
import time

# 導入AI模型
submission = utils.read_file("submission.py")
agent = utils.get_last_callable(submission)

class AIThread(QThread):
    """AI思考線程"""
    move_calculated = pyqtSignal(int)
    
    def __init__(self, board, rows, cols):
        super().__init__()
        self.board = board
        self.rows = rows
        self.cols = cols
    
    def run(self):
        try:
            # 模擬思考時間
            time.sleep(1)
            
            # 獲取AI移動
            obs = {
                'board': self.board.flatten().tolist(),
                'mark': 2
            }
            config = {'rows': self.rows, 'columns': self.cols, 'inarow': 4}
            
            move = agent(obs, config)
            
            # 驗證移動
            if not (0 <= move < self.cols and self.board[0][move] == 0):
                # 如果AI返回無效移動，選擇第一個有效的列
                for col in range(self.cols):
                    if self.board[0][col] == 0:
                        move = col
                        break
                else:
                    move = -1  # 無效移動
            
            self.move_calculated.emit(move)
            
        except Exception as e:
            print(f"AI決策出錯: {e}")
            # 回退到第一個有效移動
            for col in range(self.cols):
                if self.board[0][col] == 0:
                    move = col
                    break
            else:
                move = -1
            
            self.move_calculated.emit(move)

class ConnectXGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1  # 1 = 人類, 2 = AI
        self.game_over = False
        self.ai_thinking = False
        self.ai_thread = None
        # 先手選擇：預設人類先手
        self.ai_starts = False
        
        self.init_ui()
        
    def init_ui(self):
        """初始化用戶界面"""
        self.setWindowTitle("ConnectX - 人類 vs AI")
        self.setFixedSize(800, 740)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2c3e50;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                font-weight: bold;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        
        # 中央Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主佈局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(30, 20, 30, 20)
        
        # 標題
        title_label = QLabel("🎮 ConnectX - 人類 vs AI 對戰")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setStyleSheet("color: white; margin: 10px;")
        main_layout.addWidget(title_label)
        
        # 狀態標籤
        self.status_label = QLabel("🔴 你的回合！")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.status_label.setStyleSheet("color: white; margin: 5px;")
        main_layout.addWidget(self.status_label)

        # 先手選擇
        starter_layout = QHBoxLayout()
        starter_layout.setAlignment(Qt.AlignCenter)
        starter_label = QLabel("先手：")
        starter_label.setFont(QFont("Arial", 11))
        self.rb_human_first = QRadioButton("我先手")
        self.rb_ai_first = QRadioButton("AI先手")
        self.rb_human_first.setChecked(True)
        self.rb_human_first.setObjectName('human_first')
        self.rb_ai_first.setObjectName('ai_first')
        self.starter_group = QButtonGroup(self)
        self.starter_group.addButton(self.rb_human_first)
        self.starter_group.addButton(self.rb_ai_first)
        self.rb_human_first.toggled.connect(self._on_starter_changed)
        self.rb_ai_first.toggled.connect(self._on_starter_changed)
        starter_layout.addWidget(starter_label)
        starter_layout.addWidget(self.rb_human_first)
        starter_layout.addWidget(self.rb_ai_first)
        main_layout.addLayout(starter_layout)
        
        # 遊戲棋盤框架
        board_frame = QFrame()
        board_frame.setStyleSheet("""
            QFrame {
                background-color: #34495e;
                border: 3px solid #34495e;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        main_layout.addWidget(board_frame)
        
        # 棋盤佈局
        board_layout = QVBoxLayout(board_frame)
        board_layout.setSpacing(5)
        
        # 列按鈕佈局
        button_layout = QHBoxLayout()
        button_layout.setSpacing(5)
        
        # 創建列按鈕
        self.column_buttons = []
        for col in range(self.cols):
            btn = QPushButton(f"⬇️ {col}")
            btn.setFont(QFont("Arial", 12, QFont.Bold))
            btn.setFixedSize(80, 40)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border: 2px solid #2980b9;
                }
                QPushButton:hover {
                    background-color: #5dade2;
                }
                QPushButton:pressed {
                    background-color: #2471a3;
                }
                QPushButton:disabled {
                    background-color: #7f8c8d;
                    border: 2px solid #5d6d7e;
                }
            """)
            btn.clicked.connect(lambda checked, c=col: self.human_move(c))
            button_layout.addWidget(btn)
            self.column_buttons.append(btn)
        
        board_layout.addLayout(button_layout)
        
        # 棋盤格子佈局
        grid_layout = QGridLayout()
        grid_layout.setSpacing(2)
        
        # 創建棋盤格子
        self.cells = []
        for row in range(self.rows):
            cell_row = []
            for col in range(self.cols):
                cell = QLabel("⚪")
                cell.setAlignment(Qt.AlignCenter)
                cell.setFont(QFont("Arial", 24, QFont.Bold))
                cell.setFixedSize(80, 80)
                cell.setStyleSheet("""
                    QLabel {
                        background-color: #ecf0f1;
                        border: 2px solid #bdc3c7;
                        border-radius: 5px;
                    }
                """)
                grid_layout.addWidget(cell, row, col)
                cell_row.append(cell)
            self.cells.append(cell_row)
        
        board_layout.addLayout(grid_layout)
        
        # 控制按鈕佈局
        control_layout = QHBoxLayout()
        control_layout.setSpacing(20)
        control_layout.setAlignment(Qt.AlignCenter)
        
        # 重新開始按鈕
        self.restart_button = QPushButton("🔄 重新開始")
        self.restart_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.restart_button.setFixedSize(120, 40)
        self.restart_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: 2px solid #c0392b;
            }
            QPushButton:hover {
                background-color: #ec7063;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        self.restart_button.clicked.connect(self.restart_game)
        control_layout.addWidget(self.restart_button)
        
        # 退出按鈕
        self.quit_button = QPushButton("❌ 退出")
        self.quit_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.quit_button.setFixedSize(120, 40)
        self.quit_button.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: 2px solid #7f8c8d;
            }
            QPushButton:hover {
                background-color: #a6acaf;
            }
            QPushButton:pressed {
                background-color: #6c7b7f;
            }
        """)
        self.quit_button.clicked.connect(self.quit_game)
        control_layout.addWidget(self.quit_button)
        
        main_layout.addLayout(control_layout)
        
        # 遊戲說明
        info_label = QLabel("目標：連續四個棋子（水平、垂直或對角線）\n🔴 你是紅色  🟡 AI是黃色")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setFont(QFont("Arial", 10))
        info_label.setStyleSheet("color: #bdc3c7; margin: 10px;")
        main_layout.addWidget(info_label)
        
        # 顯示歡迎消息
        QTimer.singleShot(100, self.show_welcome_message)

    def _on_starter_changed(self, checked: bool):
        if not checked:
            return
        # 根據選項更新狀態，並重開局以套用
        self.ai_starts = self.rb_ai_first.isChecked()
        self.restart_game()

    def show_welcome_message(self):
        """顯示歡迎消息"""
        msg = QMessageBox()
        msg.setWindowTitle("歡迎")
        msg.setText("🎮 歡迎來到 ConnectX！")
        msg.setInformativeText(
            "遊戲規則：\n"
            "• 目標：連續四個棋子（水平、垂直或對角線）\n"
            "• 🔴 你是紅色棋子\n"
            "• 🟡 AI是黃色棋子\n"
            "• 點擊列按鈕放置棋子\n"
            "• 人類先手\n\n"
            "祝你好運！"
        )
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        
    def update_board_display(self):
        """更新棋盤顯示"""
        for row in range(self.rows):
            for col in range(self.cols):
                if self.board[row][col] == 0:
                    self.cells[row][col].setText("⚪")
                    self.cells[row][col].setStyleSheet("""
                        QLabel {
                            background-color: #ecf0f1;
                            border: 2px solid #bdc3c7;
                            border-radius: 5px;
                        }
                    """)
                elif self.board[row][col] == 1:
                    self.cells[row][col].setText("🔴")
                    self.cells[row][col].setStyleSheet("""
                        QLabel {
                            background-color: #ffe6e6;
                            border: 2px solid #ff9999;
                            border-radius: 5px;
                        }
                    """)
                else:
                    self.cells[row][col].setText("🟡")
                    self.cells[row][col].setStyleSheet("""
                        QLabel {
                            background-color: #fff3cd;
                            border: 2px solid #ffcc66;
                            border-radius: 5px;
                        }
                    """)
    
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
    
    def human_move(self, col):
        """處理人類玩家移動"""
        if self.game_over or self.ai_thinking or self.current_player != 1:
            return
            
        if not self.is_valid_move(col):
            msg = QMessageBox()
            msg.setWindowTitle("無效移動")
            msg.setText("該列已滿，請選擇其他列！")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        
        # 執行移動
        if self.make_move(col, 1):
            self.update_board_display()
            
            # 檢查是否獲勝
            if self.check_win(1):
                self.game_over = True
                self.status_label.setText("🎉 恭喜！你贏了！")
                self.status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
                self.disable_buttons()
                
                msg = QMessageBox()
                msg.setWindowTitle("遊戲結束")
                msg.setText("🎉 恭喜！你贏了！")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return
            
            # 檢查平局
            if self.is_board_full():
                self.game_over = True
                self.status_label.setText("🤝 平局！")
                self.status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
                self.disable_buttons()
                
                msg = QMessageBox()
                msg.setWindowTitle("遊戲結束")
                msg.setText("🤝 平局！棋盤已滿。")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return
            
            # 切換到AI回合
            self.current_player = 2
            self.ai_turn()
    
    def ai_turn(self):
        """AI回合"""
        if self.game_over:
            return
            
        self.ai_thinking = True
        self.status_label.setText("🟡 AI思考中...")
        self.status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
        self.disable_buttons()
        
        # 創建AI線程
        self.ai_thread = AIThread(self.board.copy(), self.rows, self.cols)
        self.ai_thread.move_calculated.connect(self.execute_ai_move)
        self.ai_thread.start()
    
    def execute_ai_move(self, move):
        """執行AI移動"""
        self.ai_thinking = False
        
        if move == -1 or self.game_over:
            self.status_label.setText("❌ AI無法移動")
            self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
            return
        
        # 執行AI移動
        if self.make_move(move, 2):
            self.update_board_display()
            
            # 檢查AI是否獲勝
            if self.check_win(2):
                self.game_over = True
                self.status_label.setText("🤖 AI獲勝！")
                self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
                self.disable_buttons()
                
                msg = QMessageBox()
                msg.setWindowTitle("遊戲結束")
                msg.setText("🤖 AI獲勝！再接再厲！")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return
            
            # 檢查平局
            if self.is_board_full():
                self.game_over = True
                self.status_label.setText("🤝 平局！")
                self.status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
                self.disable_buttons()
                
                msg = QMessageBox()
                msg.setWindowTitle("遊戲結束")
                msg.setText("🤝 平局！棋盤已滿。")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return
            
            # 切換回人類回合
            self.current_player = 1
            self.status_label.setText("🔴 你的回合！")
            self.status_label.setStyleSheet("color: white; font-weight: bold;")
            self.enable_buttons()
    
    def disable_buttons(self):
        """禁用列按鈕"""
        for btn in self.column_buttons:
            btn.setEnabled(False)
    
    def enable_buttons(self):
        """啟用列按鈕"""
        if not self.game_over and not self.ai_thinking:
            for btn in self.column_buttons:
                btn.setEnabled(True)
    
    def restart_game(self):
        """重新開始遊戲"""
        # 停止AI線程
        if self.ai_thread and self.ai_thread.isRunning():
            self.ai_thread.terminate()
            self.ai_thread.wait()
        
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.game_over = False
        self.ai_thinking = False
        self.update_board_display()
        
        # 根據先手選擇設定當前玩家與狀態
        if self.ai_starts:
            self.current_player = 2  # AI 行動
            self.status_label.setText("🟡 AI思考中...")
            self.status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
            self.disable_buttons()
            QTimer.singleShot(300, self.ai_turn)
        else:
            self.current_player = 1  # 人類行動
            self.status_label.setText("🔴 你的回合！")
            self.status_label.setStyleSheet("color: white; font-weight: bold;")
            self.enable_buttons()
    
    def quit_game(self):
        """退出遊戲"""
        reply = QMessageBox.question(
            self, 
            "退出", 
            "確定要退出遊戲嗎？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 停止AI線程
            if self.ai_thread and self.ai_thread.isRunning():
                self.ai_thread.terminate()
                self.ai_thread.wait()
            self.close()
    
    def closeEvent(self, event):
        """窗口關閉事件"""
        # 停止AI線程
        if self.ai_thread and self.ai_thread.isRunning():
            self.ai_thread.terminate()
            self.ai_thread.wait()
        event.accept()

def main():
    """主函數"""
    app = QApplication(sys.argv)
    
    # 設置應用程序樣式
    app.setStyle('Fusion')
    
    # 設置深色主題
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(44, 62, 80))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(52, 73, 94))
    palette.setColor(QPalette.AlternateBase, QColor(66, 84, 103))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(52, 73, 94))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    try:
        game = ConnectXGUI()
        game.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"遊戲出現錯誤: {e}")
        msg = QMessageBox()
        msg.setWindowTitle("錯誤")
        msg.setText(f"遊戲出現錯誤: {e}\n請確認 submission.py 文件存在且正確。")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

if __name__ == "__main__":
    main()
