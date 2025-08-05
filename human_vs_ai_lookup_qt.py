#!/usr/bin/env python3
"""
ConnectX Human vs AI Game - PyQt5 GUI Version with Lookup Table

使用方法：
- 點擊列按鈕放置棋子
- 人類是紅色棋子，AI是黃色棋子
- 目標是連續四個棋子（水平、垂直或對角線）
- AI 使用查表策略，根據 connectx-state-action-value.txt 做出最佳移動
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QPushButton, QLabel, 
                            QMessageBox, QFrame, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor
import time
import os

class LookupTableChecker(QThread):
    """查表文件檢查線程"""
    check_finished = pyqtSignal(bool, str)
    
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        
    def run(self):
        """檢查查表文件是否存在並可讀取"""
        try:
            if not os.path.exists(self.filename):
                self.check_finished.emit(False, f"錯誤：找不到文件 {self.filename}")
                return
            
            # 檢查文件是否可讀
            if not os.access(self.filename, os.R_OK):
                self.check_finished.emit(False, f"錯誤：無法讀取文件 {self.filename}")
                return
            
            # 簡單檢查文件格式（只讀第一行）
            with open(self.filename, 'r') as f:
                first_line = f.readline().strip()
                if not first_line:
                    self.check_finished.emit(False, "錯誤：文件為空")
                    return
                
                parts = first_line.split(',')
                if len(parts) != 8:
                    self.check_finished.emit(False, "錯誤：文件格式不正確")
                    return
            
            file_size_mb = os.path.getsize(self.filename) / (1024 * 1024)
            self.check_finished.emit(True, f"查表文件就緒 ({file_size_mb:.1f} MB)")
            
        except Exception as e:
            self.check_finished.emit(False, f"檢查錯誤：{str(e)}")

class DynamicLookupTable:
    """動態查表類"""
    def __init__(self, filename):
        self.filename = filename
        self.cache = {}  # 小型緩存，存儲最近查詢的狀態
        self.max_cache_size = 1000  # 最大緩存大小
        
    def lookup_state(self, state_str):
        """查找特定狀態的動作值"""
        # 先檢查緩存
        if state_str in self.cache:
            return self.cache[state_str]
        print(state_str)
        # 如果緩存中沒有，則動態查找
        try:
            with open(self.filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split(',')
                    if len(parts) != 8:
                        continue
                    
                    if parts[0] == state_str:
                        # 找到了！解析動作值
                        action_values = []
                        for i in range(1, 8):
                            if parts[i].strip() == '':
                                action_values.append(None)
                            else:
                                try:
                                    action_values.append(int(parts[i]))
                                except ValueError:
                                    action_values.append(None)
                        
                        # 加入緩存
                        self._add_to_cache(state_str, action_values)
                        return action_values
            
            # 沒找到
            return None
            
        except Exception as e:
            print(f"查表錯誤: {e}")
            return None
    
    def _add_to_cache(self, state_str, action_values):
        """將狀態加入緩存"""
        # 如果緩存滿了，移除最舊的條目
        if len(self.cache) >= self.max_cache_size:
            # 移除第一個條目（FIFO）
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[state_str] = action_values
    
    def get_cache_info(self):
        """獲取緩存信息"""
        return f"緩存: {len(self.cache)}/{self.max_cache_size} 個狀態"

class AIThread(QThread):
    """AI思考線程"""
    move_calculated = pyqtSignal(int, str)  # 移動和解釋
    
    def __init__(self, board, rows, cols, lookup_table, current_player):
        super().__init__()
        self.board = board
        self.rows = rows
        self.cols = cols
        self.lookup_table = lookup_table  # 現在是 DynamicLookupTable 實例
        self.current_player = current_player
    
    def board_to_state_string(self, board, current_player):
        """將棋盤轉換為狀態字符串（Kaggle格式）"""
        # 將numpy array轉換為平面列表，從左到右，從上到下讀取
        flat_board = board.flatten()
        
        # 在查表中，1總是先手，2總是後手
        # 由於AI是1（先手），人類是2（後手），直接使用棋盤狀態即可
        adjusted_board = flat_board.tolist()
        
        return ''.join(map(str, adjusted_board))
    
    def get_valid_columns(self, board):
        """獲取所有有效的列"""
        valid_cols = []
        for col in range(self.cols):
            if board[0][col] == 0:
                valid_cols.append(col)
        return valid_cols
    
    def run(self):
        try:
            # 模擬思考時間
            time.sleep(0.5)
            
            # 轉換棋盤狀態
            state_str = self.board_to_state_string(self.board, self.current_player)
            
            explanation = ""
            move = -1
            
            # 動態查表
            action_values = self.lookup_table.lookup_state(state_str)
            
            if action_values is not None:
                explanation = "📖 查表決策"
                
                # 找到最佳移動
                best_value = None
                best_moves = []
                
                valid_cols = self.get_valid_columns(self.board)
                
                for col in valid_cols:
                    if action_values[col] is not None:
                        if best_value is None or action_values[col] > best_value:
                            best_value = action_values[col]
                            best_moves = [col]
                        elif action_values[col] == best_value:
                            best_moves.append(col)
                
                if best_moves:
                    # 如果有多個相同最佳值，按優先順序選擇 [3, 2, 4, 1, 5, 0, 6]
                    priority_order = [3, 2, 4, 1, 5, 0, 6]
                    for preferred_col in priority_order:
                        if preferred_col in best_moves:
                            move = preferred_col
                            break
                    
                    if move == -1:
                        move = best_moves[0]
                    
                    # 解釋移動值
                    if best_value > 0:
                        explanation += f" (勝利於 {best_value} 步內)"
                    elif best_value == 0:
                        explanation += f" (平局)"
                    else:
                        explanation += f" (失敗於 {abs(best_value)} 步內)"
                else:
                    # 沒有找到有效移動，選擇第一個有效列
                    move = valid_cols[0] if valid_cols else -1
                    explanation = "⚠️ 查表無有效值，選擇第一個可用列"
            
            else:
                # 狀態不在表中，使用簡單策略
                explanation = "❓ 狀態不在查表中，使用啟發式"
                
                valid_cols = self.get_valid_columns(self.board)
                if valid_cols:
                    # 優先選擇中間列
                    priority_order = [3, 2, 4, 1, 5, 0, 6]
                    for preferred_col in priority_order:
                        if preferred_col in valid_cols:
                            move = preferred_col
                            break
                    
                    if move == -1:
                        move = valid_cols[0]
                else:
                    move = -1
            
            # 最終驗證移動有效性
            if not (0 <= move < self.cols and self.board[0][move] == 0):
                valid_cols = self.get_valid_columns(self.board)
                move = valid_cols[0] if valid_cols else -1
                explanation = "🔧 移動修正"
            
            # 添加緩存信息
            cache_info = self.lookup_table.get_cache_info()
            explanation += f" | {cache_info}"
            
            self.move_calculated.emit(move, explanation)
            
        except Exception as e:
            print(f"AI決策出錯: {e}")
            # 回退到第一個有效移動
            valid_cols = self.get_valid_columns(self.board)
            move = valid_cols[0] if valid_cols else -1
            explanation = f"❌ 錯誤回退: {str(e)}"
            self.move_calculated.emit(move, explanation)

class ConnectXGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1  # 1 = AI (先手), 2 = 人類 (後手)
        self.game_over = False
        self.ai_thinking = False
        self.ai_thread = None
        self.lookup_table = None  # 改為 DynamicLookupTable 實例
        self.table_ready = False
        
        self.init_ui()
        self.check_lookup_table()
        
    def init_ui(self):
        """初始化用戶界面"""
        self.setWindowTitle("ConnectX - 人類 vs AI (查表版)")
        self.setFixedSize(800, 750)
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
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # 標題
        title_label = QLabel("🎮 ConnectX - 人類 vs AI 對戰 (動態查表版)")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setStyleSheet("color: white; margin: 10px;")
        main_layout.addWidget(title_label)
        
        # 檢查狀態標籤
        self.check_label = QLabel("正在檢查查表文件...")
        self.check_label.setAlignment(Qt.AlignCenter)
        self.check_label.setFont(QFont("Arial", 12))
        self.check_label.setStyleSheet("color: #3498db; margin: 5px;")
        main_layout.addWidget(self.check_label)
        
        # 狀態標籤
        self.status_label = QLabel("⏳ 正在檢查...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.status_label.setStyleSheet("color: white; margin: 5px;")
        main_layout.addWidget(self.status_label)
        
        # AI 解釋標籤
        self.ai_explanation_label = QLabel("")
        self.ai_explanation_label.setAlignment(Qt.AlignCenter)
        self.ai_explanation_label.setFont(QFont("Arial", 11))
        self.ai_explanation_label.setStyleSheet("color: #f39c12; margin: 5px;")
        main_layout.addWidget(self.ai_explanation_label)
        
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
            btn.setEnabled(False)  # 初始禁用，等待查表加載
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
        self.restart_button.setEnabled(False)
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
        info_label = QLabel("目標：連續四個棋子（水平、垂直或對角線）\n� AI是黃色（先手）  � 你是紅色（後手）（動態查表決策）")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setFont(QFont("Arial", 10))
        info_label.setStyleSheet("color: #bdc3c7; margin: 10px;")
        main_layout.addWidget(info_label)
        
    def check_lookup_table(self):
        """檢查查表文件"""
        self.checker_thread = LookupTableChecker("connectx-state-action-value.txt")
        self.checker_thread.check_finished.connect(self.on_check_finished)
        self.checker_thread.start()
    
    def on_check_finished(self, success, message):
        """查表文件檢查完成"""
        self.check_label.setText(message)
        
        if success:
            # 初始化動態查表
            self.lookup_table = DynamicLookupTable("connectx-state-action-value.txt")
            self.table_ready = True
            
            # 隱藏檢查標籤
            self.check_label.hide()
            
            self.status_label.setText("� AI的回合！")
            print('\a')
            self.status_label.setStyleSheet("color: white; font-weight: bold;")
            self.enable_buttons()
            self.restart_button.setEnabled(True)
            
            # 顯示歡迎消息，然後AI先行
            QTimer.singleShot(100, self.show_welcome_message)
        else:
            self.status_label.setText("❌ 查表文件不可用")
            self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
            self.check_label.setStyleSheet("color: #e74c3c; margin: 5px;")
            
            msg = QMessageBox()
            msg.setWindowTitle("錯誤")
            msg.setText("無法使用查表文件！\n請確認 connectx-state-action-value.txt 文件存在且可讀取。")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        
    def show_welcome_message(self):
        """顯示歡迎消息"""
        msg = QMessageBox()
        msg.setWindowTitle("歡迎")
        msg.setText("🎮 歡迎來到 ConnectX 動態查表版！")
        msg.setInformativeText(
            "動態查表系統已就緒！\n\n"
            "遊戲規則：\n"
            "• 目標：連續四個棋子（水平、垂直或對角線）\n"
            "• � AI是黃色棋子（先手，玩家1）\n"
            "• � 你是紅色棋子（後手，玩家2）\n"
            "• 點擊列按鈕放置棋子\n"
            "• AI先手\n\n"
            "AI會即時查表並顯示決策說明，挑戰完美策略吧！\n"
            "動態查表不會占用大量記憶體，只在需要時查找！"
        )
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        
        # 歡迎消息後，AI先行
        QTimer.singleShot(500, self.ai_turn)
        
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
                    self.cells[row][col].setText("�")
                    self.cells[row][col].setStyleSheet("""
                        QLabel {
                            background-color: #fff3cd;
                            border: 2px solid #ffcc66;
                            border-radius: 5px;
                        }
                    """)
                else:
                    self.cells[row][col].setText("�")
                    self.cells[row][col].setStyleSheet("""
                        QLabel {
                            background-color: #ffe6e6;
                            border: 2px solid #ff9999;
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
        if self.game_over or self.ai_thinking or self.current_player != 2 or not self.table_ready:
            return
            
        if not self.is_valid_move(col):
            msg = QMessageBox()
            msg.setWindowTitle("無效移動")
            msg.setText("該列已滿，請選擇其他列！")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return
        
        # 清除AI解釋
        self.ai_explanation_label.setText("")
        
        # 執行移動
        if self.make_move(col, 2):
            self.update_board_display()
            
            # 檢查是否獲勝
            if self.check_win(2):
                self.game_over = True
                self.status_label.setText("🎉 恭喜！你贏了！")
                self.status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
                self.ai_explanation_label.setText("🎊 你擊敗了完美AI！")
                self.disable_buttons()
                
                msg = QMessageBox()
                msg.setWindowTitle("遊戲結束")
                msg.setText("🎉 不可思議！你擊敗了完美AI！")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return
            
            # 檢查平局
            if self.is_board_full():
                self.game_over = True
                self.status_label.setText("🤝 平局！")
                self.status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
                self.ai_explanation_label.setText("⚖️ 雙方平手！")
                self.disable_buttons()
                
                msg = QMessageBox()
                msg.setWindowTitle("遊戲結束")
                msg.setText("🤝 平局！棋盤已滿。")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return
            
            # 切換到AI回合
            self.current_player = 1
            self.ai_turn()
    
    def ai_turn(self):
        """AI回合"""
        if self.game_over or not self.table_ready:
            return
            
        self.ai_thinking = True
        self.status_label.setText("🟡 AI思考中...")
        self.status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
        self.ai_explanation_label.setText("🤔 正在分析最佳移動...")
        self.disable_buttons()
        
        # 創建AI線程
        self.ai_thread = AIThread(self.board.copy(), self.rows, self.cols, self.lookup_table, self.current_player)
        self.ai_thread.move_calculated.connect(self.execute_ai_move)
        self.ai_thread.start()
    
    def execute_ai_move(self, move, explanation):
        """執行AI移動"""
        self.ai_thinking = False
        self.ai_explanation_label.setText(explanation)
        
        if move == -1 or self.game_over:
            self.status_label.setText("❌ AI無法移動")
            self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
            return
        
        # 執行AI移動
        if self.make_move(move, 1):
            self.update_board_display()
            
            # 檢查AI是否獲勝
            if self.check_win(1):
                self.game_over = True
                self.status_label.setText("🤖 AI獲勝！")
                self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
                self.ai_explanation_label.setText("🎯 完美策略勝利！")
                self.disable_buttons()
                
                msg = QMessageBox()
                msg.setWindowTitle("遊戲結束")
                msg.setText("🤖 AI獲勝！完美策略不可戰勝！")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return
            
            # 檢查平局
            if self.is_board_full():
                self.game_over = True
                self.status_label.setText("🤝 平局！")
                self.status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
                self.ai_explanation_label.setText("⚖️ 雙方平手！")
                self.disable_buttons()
                
                msg = QMessageBox()
                msg.setWindowTitle("遊戲結束")
                msg.setText("🤝 平局！棋盤已滿。")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return
            
            # 切換回人類回合
            self.current_player = 2
            self.status_label.setText("🔴 你的回合！")
            self.status_label.setStyleSheet("color: white; font-weight: bold;")
            self.enable_buttons()
    
    def disable_buttons(self):
        """禁用列按鈕"""
        for btn in self.column_buttons:
            btn.setEnabled(False)
    
    def enable_buttons(self):
        """啟用列按鈕"""
        if not self.game_over and not self.ai_thinking and self.table_ready:
            for btn in self.column_buttons:
                btn.setEnabled(True)
    
    def restart_game(self):
        """重新開始遊戲"""
        # 停止AI線程
        if self.ai_thread and self.ai_thread.isRunning():
            self.ai_thread.terminate()
            self.ai_thread.wait()
        
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1  # AI先手
        self.game_over = False
        self.ai_thinking = False
        
        self.update_board_display()
        self.status_label.setText("� AI的回合！")
        self.status_label.setStyleSheet("color: white; font-weight: bold;")
        self.ai_explanation_label.setText("")
        self.enable_buttons()
        
        # 重新開始後，AI先行
        QTimer.singleShot(500, self.ai_turn)
    
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
            # 停止所有線程
            if hasattr(self, 'checker_thread') and self.checker_thread.isRunning():
                self.checker_thread.terminate()
                self.checker_thread.wait()
            if self.ai_thread and self.ai_thread.isRunning():
                self.ai_thread.terminate()
                self.ai_thread.wait()
            self.close()
    
    def closeEvent(self, event):
        """窗口關閉事件"""
        # 停止所有線程
        if hasattr(self, 'checker_thread') and self.checker_thread.isRunning():
            self.checker_thread.terminate()
            self.checker_thread.wait()
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
        msg.setText(f"遊戲出現錯誤: {e}\n請確認 connectx-state-action-value.txt 文件存在且可讀取。")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

if __name__ == "__main__":
    main()
