#!/usr/bin/env python3
"""
Connect4 人機對戰 GUI
支援多種AI對手：訓練的RL模型、C4Solver、隨機對手等
"""

import os
import sys
import torch
import numpy as np
import yaml
from typing import Optional, List, Tuple
import time
import random

# 導入必要的組件（不引入 PPOAgent，直接使用 ConnectXNet 推論）
from train_connectx_rl_robust import ConnectXNet, flat_to_2d, is_win_from, find_drop_row
from c4solver_wrapper import get_c4solver, C4SolverWrapper
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QPushButton, QLabel, QMessageBox, QFileDialog, QComboBox, QCheckBox
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont, QColor, QPalette
    _QT_OK = True
except Exception:
    _QT_OK = False

class Connect4Game:
    """Connect4遊戲核心"""
    
    def __init__(self):
        self.board = [0] * 42  # 6x7棋盤，扁平化
        self.current_player = 1  # 1=人類, 2=AI
        self.game_over = False
        self.winner = 0
        self.move_history = []
        
    def reset(self):
        """重置遊戲"""
        self.board = [0] * 42
        self.current_player = 1
        self.game_over = False
        self.winner = 0
        self.move_history = []
        
    def get_valid_actions(self) -> List[int]:
        """獲取有效動作(可下棋的列)"""
        return [col for col in range(7) if self.board[col] == 0]
    
    def make_move(self, col: int) -> bool:
        """執行移動
        
        Args:
            col: 列號(0-6)
            
        Returns:
            是否成功執行移動
        """
        if col < 0 or col > 6:
            return False
        
        # 找到該列最底部的空位
        grid = flat_to_2d(self.board)
        row = find_drop_row(grid, col)
        
        if row is None:
            return False  # 該列已滿
            
        # 執行移動
        self.board[row * 7 + col] = self.current_player
        self.move_history.append(col)
        
        # 檢查是否獲勝（需在grid上暫時放入新子再檢查）
        try:
            grid[row][col] = self.current_player
        except Exception:
            pass
        if is_win_from(grid, row, col, self.current_player):
            self.game_over = True
            self.winner = self.current_player
        # 檢查是否平局
        elif len(self.get_valid_actions()) == 0:
            self.game_over = True
            self.winner = 0  # 平局
        else:
            # 切換玩家
            self.current_player = 3 - self.current_player
            
        return True
    
    def display_board(self):
        """顯示棋盤"""
        print("\n  " + " ".join([str(i) for i in range(7)]))
        print("  " + "-" * 13)
        
        grid = flat_to_2d(self.board)
        symbols = {0: ".", 1: "🔴", 2: "🔵"}
        
        for row in range(6):
            print(f"{row}|", end="")
            for col in range(7):
                print(f"{symbols[grid[row][col]]} ", end="")
            print("|")
        print("  " + "-" * 13)
        print("  " + " ".join([str(i) for i in range(7)]))
    
    def get_board_analysis(self) -> str:
        """獲取棋盤分析信息"""
        valid_moves = self.get_valid_actions()
        analysis = f"\n📊 局面分析:\n"
        analysis += f"  當前玩家: {'🔴 人類' if self.current_player == 1 else '🔵 AI'}\n"
        analysis += f"  可下位置: {valid_moves}\n"
        analysis += f"  移動歷史: {' -> '.join(map(str, self.move_history)) if self.move_history else '無'}\n"
        
        # 檢查威脅
        threats = []
        for col in valid_moves:
            # 檢查是否是獲勝移動
            if self.is_winning_move(col, self.current_player):
                threats.append(f"🎯 第{col}列可獲勝")
            # 檢查是否需要阻擋對手
            elif self.is_winning_move(col, 3 - self.current_player):
                threats.append(f"🛡️ 第{col}列需阻擋對手")
        
        if threats:
            analysis += f"  戰術提示: {', '.join(threats)}\n"
            
        return analysis
    
    def is_winning_move(self, col: int, player: int) -> bool:
        """檢查是否是獲勝移動"""
        grid = flat_to_2d(self.board)
        row = find_drop_row(grid, col)
        if row is None:
            return False
        return is_win_from(grid, row, col, player)


class AIOpponent:
    """AI對手基類"""
    
    def __init__(self, name: str):
        self.name = name
        
    def get_move(self, game: Connect4Game) -> int:
        """獲取AI的移動"""
        raise NotImplementedError
    
    def get_confidence(self) -> float:
        """獲取信心分數(0-1)"""
        return 0.5


class RandomAI(AIOpponent):
    """隨機AI對手"""
    
    def __init__(self):
        super().__init__("隨機AI")
        
    def get_move(self, game: Connect4Game) -> int:
        valid_actions = game.get_valid_actions()
        return random.choice(valid_actions) if valid_actions else 0
    
    def get_confidence(self) -> float:
        return 0.2


class CenterPreferenceAI(AIOpponent):
    """偏好中央的AI"""
    
    def __init__(self):
        super().__init__("中央偏好AI")
        
    def get_move(self, game: Connect4Game) -> int:
        valid_actions = game.get_valid_actions()
        if not valid_actions:
            return 0
            
        # 偏好順序：中央 -> 兩側
        preference = [3, 4, 2, 5, 1, 6, 0]
        for col in preference:
            if col in valid_actions:
                return col
        return valid_actions[0]
    
    def get_confidence(self) -> float:
        return 0.4


class C4SolverAI(AIOpponent):
    """使用C4Solver的完美AI"""
    
    def __init__(self):
        super().__init__("C4Solver (完美)")
        self.solver = get_c4solver()
        self.last_confidence = 0.8
        
    def get_move(self, game: Connect4Game) -> int:
        if self.solver is None:
            raise NoSolverError()
            # Fallback到中央偏好
        
        try:
            action, confidence = self.solver.get_best_move(game.board, game.get_valid_actions())
            self.last_confidence = confidence
            return action
        except Exception as e:
            print(f"⚠️ C4Solver錯誤: {e}")
            return CenterPreferenceAI().get_move(game)
    
    def get_confidence(self) -> float:
        return self.last_confidence


class RLModelAI(AIOpponent):
    """使用訓練的強化學習模型"""
    
    def __init__(self, model_path: str):
        super().__init__("RL模型")
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.last_confidence = 0.6
        
        try:
            self.load_model()
        except Exception as e:
            print(f"⚠️ 無法載入RL模型: {e}")
            
    def load_model(self):
        """載入訓練好的模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 創建模型
        self.model = ConnectXNet(
            input_size=126,  # 6*7*2 + 7*2 = 98
            hidden_size=512,
            num_layers=3
        ).to(self.device)
        
        # 載入權重
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        
        print(f"✅ 成功載入RL模型: {self.model_path}")
        
    def encode_state(self, game: Connect4Game) -> torch.Tensor:
        """編碼遊戲狀態"""
        try:
            # 基本棋盤編碼 (6*7*2 = 84)
            encoded = np.zeros((6, 7, 2), dtype=np.float32)
            grid = flat_to_2d(game.board)
            
            for r in range(6):
                for c in range(7):
                    if grid[r][c] == 1:
                        encoded[r, c, 0] = 1
                    elif grid[r][c] == 2:
                        encoded[r, c, 1] = 1
            
            # 當前玩家編碼 (1)
            current_player_encoded = np.array([game.current_player - 1], dtype=np.float32)
            
            # 有效動作編碼 (7)
            valid_actions = game.get_valid_actions()
            valid_mask = np.zeros(7, dtype=np.float32)
            for action in valid_actions:
                valid_mask[action] = 1
                
            # 移動計數編碼 (1)
            move_count = np.array([len(game.move_history) / 42.0], dtype=np.float32)
            
            # 組合所有特徵
            combined = np.concatenate([
                encoded.flatten(),        # 84
                current_player_encoded,   # 1
                valid_mask,              # 7
                move_count              # 1
            ])  # 總共 93 維
            
            # 如果模型期望126維，補充額外特徵
            if len(combined) < 126:
                # 添加位置特徵
                position_features = np.zeros(126 - len(combined), dtype=np.float32)
                combined = np.concatenate([combined, position_features])
            
            return torch.FloatTensor(combined).unsqueeze(0).to(self.device)
            
        except Exception as e:
            print(f"⚠️ 狀態編碼錯誤: {e}")
            # 返回零向量作為fallback
            return torch.zeros(1, 126).to(self.device)
    
    def get_move(self, game: Connect4Game) -> int:
        if self.model is None:
            return CenterPreferenceAI().get_move(game)
        
        try:
            with torch.no_grad():
                state = self.encode_state(game)
                action_probs, value = self.model(state)
                
                # 只考慮有效動作
                valid_actions = game.get_valid_actions()
                if not valid_actions:
                    return 0
                
                # 獲取有效動作的概率
                probs = torch.softmax(action_probs[0], dim=0).cpu().numpy()
                valid_probs = [probs[a] for a in valid_actions]
                
                # 選擇概率最高的動作
                best_idx = np.argmax(valid_probs)
                action = valid_actions[best_idx]
                
                # 計算信心分數
                self.last_confidence = min(1.0, max(valid_probs) * 2)
                
                return action
                
        except Exception as e:
            print(f"⚠️ RL模型預測錯誤: {e}")
            return CenterPreferenceAI().get_move(game)
    
    def get_confidence(self) -> float:
        return self.last_confidence


class GameManager:
    """（舊）CLI 遊戲管理器（已被 GUI 取代）"""
    pass


# ===== PyQt5 GUI 實作 =====
class AIWorker(QThread):
    move_ready = pyqtSignal(int, float)  # (col, think_time)

    def __init__(self, game_snapshot: Connect4Game, ai: AIOpponent):
        super().__init__()
        self._game = game_snapshot
        self._ai = ai

    def run(self):
        start = time.time()
        try:
            move = int(self._ai.get_move(self._game))
        except Exception:
            # fallback: first valid
            valid = self._game.get_valid_actions()
            move = valid[0] if valid else -1
        dt = time.time() - start
        self.move_ready.emit(move, dt)


class Connect4GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        if not _QT_OK:
            raise RuntimeError("PyQt5 not installed. Please install PyQt5 to use the GUI.")
        self.setWindowTitle("Connect4 人機對戰 (GUI)")
        self.rows, self.cols = 6, 7
        self.game = Connect4Game()
        self.ai: Optional[AIOpponent] = None
        self.ai_thread: Optional[AIWorker] = None
        self._build_ui()
        self._reset_board(start_with_ai=False)

    def _build_ui(self):
        # global palette: dark-ish
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(44, 62, 80))
        pal.setColor(QPalette.WindowText, Qt.white)
        self.setPalette(pal)

        cw = QWidget(); self.setCentralWidget(cw)
        main = QVBoxLayout(cw)
        main.setContentsMargins(16, 16, 16, 16)
        main.setSpacing(10)

        # Controls
        top = QHBoxLayout(); top.setSpacing(10)
        self.ai_combo = QComboBox(); self.ai_combo.addItems([
            "隨機AI", "中央偏好AI", "C4Solver (完美)", "RL 模型"
        ])
        self.ai_combo.currentIndexChanged.connect(self._on_ai_changed)
        self.load_btn = QPushButton("載入RL模型…")
        self.load_btn.clicked.connect(self._on_load_model)
        self.load_btn.setEnabled(False)
        self.ai_first_chk = QCheckBox("AI先手")
        self.ai_first_chk.setChecked(False)
        self.start_btn = QPushButton("開始/重開")
        self.start_btn.clicked.connect(self._on_restart)
        self.status_lbl = QLabel("🔴 你的回合！")
        self.status_lbl.setFont(QFont("Arial", 12, QFont.Bold))
        top.addWidget(QLabel("對手:")); top.addWidget(self.ai_combo)
        top.addWidget(self.load_btn)
        top.addWidget(self.ai_first_chk)
        top.addWidget(self.start_btn)
        top.addStretch(1)
        top.addWidget(self.status_lbl)
        main.addLayout(top)

        # Column buttons
        btn_row = QHBoxLayout(); btn_row.setSpacing(6)
        self.col_buttons: List[QPushButton] = []
        for c in range(self.cols):
            b = QPushButton(f"⬇ {c}"); b.setFixedHeight(36)
            b.clicked.connect(lambda _=False, col=c: self._on_human_move(col))
            self.col_buttons.append(b)
            btn_row.addWidget(b)
        main.addLayout(btn_row)

        # Board grid
        grid = QGridLayout(); grid.setSpacing(4)
        self.cells: List[List[QLabel]] = []
        for r in range(self.rows):
            row_cells = []
            for c in range(self.cols):
                lab = QLabel("⚪"); lab.setAlignment(Qt.AlignCenter)
                lab.setFont(QFont("Arial", 24, QFont.Bold))
                lab.setFixedSize(56, 56)
                lab.setStyleSheet("QLabel { background:#ecf0f1; border:2px solid #bdc3c7; border-radius:6px; }")
                grid.addWidget(lab, r, c)
                row_cells.append(lab)
            self.cells.append(row_cells)
        main.addLayout(grid)

        self.resize(640, 520)

    def _on_ai_changed(self, idx: int):
        self.load_btn.setEnabled(idx == 3)
        # instantiate AI instance
        if idx == 0:
            self.ai = RandomAI()
        elif idx == 1:
            self.ai = CenterPreferenceAI()
        elif idx == 2:
            self.ai = C4SolverAI()
        else:
            # RL will be created when model is chosen
            self.ai = None

    def _on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇RL模型", "checkpoints", "Torch Model (*.pt *.pth)")
        if not path:
            return
        try:
            self.ai = RLModelAI(path)
            QMessageBox.information(self, "模型", f"已載入模型\n{os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, "模型載入失敗", f"{e}")
            self.ai = CenterPreferenceAI()

    def _on_restart(self):
        self._reset_board(start_with_ai=self.ai_first_chk.isChecked())

    def _reset_board(self, start_with_ai: bool):
        self.game.reset()
        self._update_board()
        self._enable_columns(True)
        if start_with_ai:
            self.game.current_player = 2
            self.status_lbl.setText("🟡 AI思考中…")
            self._enable_columns(False)
            self._start_ai_turn()
        else:
            self.game.current_player = 1
            self.status_lbl.setText("🔴 你的回合！")

    def _update_board(self):
        grid = flat_to_2d(self.game.board)
        for r in range(self.rows):
            for c in range(self.cols):
                v = grid[r][c]
                if v == 1:
                    self.cells[r][c].setText("🔴")
                elif v == 2:
                    self.cells[r][c].setText("🟡")
                else:
                    self.cells[r][c].setText("⚪")

    def _enable_columns(self, on: bool):
        for b in self.col_buttons:
            b.setEnabled(on)

    def _on_human_move(self, col: int):
        if self.game.game_over or self.game.current_player != 1:
            return
        if col not in self.game.get_valid_actions():
            QMessageBox.information(self, "無效移動", "該列已滿，請選擇其他列。")
            return
        ok = self.game.make_move(col)
        self._update_board()
        if not ok:
            return
        if self.game.game_over:
            self._show_result()
            return
        # Switch to AI
        self.game.current_player = 2
        self.status_lbl.setText("🟡 AI思考中…")
        self._enable_columns(False)
        self._start_ai_turn()

    def _start_ai_turn(self):
        # Ensure AI instance
        if self.ai is None:
            idx = self.ai_combo.currentIndex()
            self._on_ai_changed(idx)
            if self.ai is None:
                self.ai = CenterPreferenceAI()
        # snapshot game for worker
        snap = Connect4Game()
        snap.board = self.game.board.copy()
        snap.current_player = self.game.current_player
        snap.move_history = self.game.move_history.copy()
        self.ai_thread = AIWorker(snap, self.ai)
        self.ai_thread.move_ready.connect(self._on_ai_move_decided)
        self.ai_thread.start()

    def _on_ai_move_decided(self, col: int, think_time: float):
        # Apply move
        if col == -1 or col not in self.game.get_valid_actions():
            # fallback first valid
            valid = self.game.get_valid_actions()
            col = valid[0] if valid else -1
        if col != -1:
            self.game.make_move(col)
        self._update_board()
        if self.game.game_over:
            self._show_result()
            return
        # back to human
        conf = 0.0
        try:
            conf = float(self.ai.get_confidence()) if self.ai else 0.0
        except Exception:
            conf = 0.0
        emoji = "�" if conf > 0.8 else ("🤔" if conf > 0.5 else "🎲")
        self.status_lbl.setText(f"🔴 你的回合！  (AI思考 {think_time:.2f}s {emoji} 信心 {conf:.0%})")
        self.game.current_player = 1
        self._enable_columns(True)

    def _show_result(self):
        self._enable_columns(False)
        if self.game.winner == 1:
            self.status_lbl.setText("🎉 你贏了！")
            QMessageBox.information(self, "遊戲結果", "� 恭喜！你贏了！")
        elif self.game.winner == 2:
            self.status_lbl.setText("🤖 AI獲勝！")
            QMessageBox.information(self, "遊戲結果", f"🤖 {self.ai.name if self.ai else 'AI'} 獲勝！")
        else:
            self.status_lbl.setText("🤝 平局！")
            QMessageBox.information(self, "遊戲結果", "🤝 平局！棋盤已滿。")


def main():
    """GUI 主函式"""
    if not _QT_OK:
        print("PyQt5 未安裝。請先安裝: pip install PyQt5")
        return
    app = QApplication(sys.argv)
    # 美化主題
    app.setStyle('Fusion')
    win = Connect4GUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
