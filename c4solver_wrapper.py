#!/usr/bin/env python3
"""
Connect4 C++ Solver Python包裝器 - 修復版本
"""

import subprocess
import os
import time
import logging
import random
from typing import List, Tuple, Optional, Dict
import numpy as np

logger = logging.getLogger(__name__)

class C4SolverError(Exception):
    """C4Solver相關錯誤"""
    pass

class C4SolverWrapper:
    """Connect4 C++ Solver Python包裝器"""
    
    def __init__(self, solver_path: str = "connect4/c4solver", timeout: float = 5.0):
        """
        初始化C4Solver包裝器
        
        Args:
            solver_path: c4solver可執行檔路徑
            timeout: 求解超時時間(秒)
        """
        self.solver_path = solver_path
        self.timeout = timeout
        
        # 檢查solver是否存在且可執行
        if not os.path.exists(solver_path):
            raise C4SolverError(f"C4Solver not found at: {solver_path}")
        
        # 測試solver是否正常工作
        try:
            self._test_solver()
            logger.info(f"✅ C4Solver initialized successfully: {solver_path}")
        except Exception as e:
            raise C4SolverError(f"C4Solver test failed: {e}")
    
    def _test_solver(self):
        """測試solver是否正常工作"""
        try:
            # 測試簡單局面
            result = subprocess.run(
                [self.solver_path, '-a'],
                input="\n",  # 空序列測試
                text=True,
                capture_output=True,
                timeout=2.0
            )
            if result.returncode != 0:
                raise C4SolverError(f"Solver returned error: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise C4SolverError("Solver test timeout")
    
    def board_to_move_sequence(self, board: List[int]) -> str:
        """
        將42長度的board轉換為移動序列字符串
        
        Args:
            board: 42長度的扁平化棋盤 [row0_col0, row0_col1, ..., row5_col6]
        
        Returns:
            移動序列字符串，例如 "4433221"
        """
        try:
            # 轉換為6x7格式
            grid = np.array(board).reshape(6, 7)
            
            # 重建移動序列
            moves = []
            temp_grid = np.zeros((6, 7), dtype=int)
            
            # 按時間順序重建每一步
            for move_num in range(1, 43):  # 最多42步
                found = False
                for col in range(7):
                    for row in range(5, -1, -1):  # 從底部往上找
                        if grid[row][col] != 0 and temp_grid[row][col] == 0:
                            # 檢查這個位置是否是當前應該放置的位置
                            if row == 5 or temp_grid[row + 1][col] != 0:
                                moves.append(str(col + 1))  # C4Solver使用1-7而不是0-6
                                temp_grid[row][col] = grid[row][col]
                                found = True
                                break
                    if found:
                        break
                if not found:
                    break
            
            return ''.join(moves)
        except Exception as e:
            logger.warning(f"Board to sequence conversion failed: {e}")
            return ""
    
    def solve_position(self, move_sequence: str, weak: bool = False, analyze: bool = False) -> Dict:
        """
        求解Connect4局面
        
        Args:
            move_sequence: 移動序列，例如 "4433221"
            weak: 是否使用弱求解模式(只判斷勝負)
            analyze: 是否分析所有可能移動
        
        Returns:
            dict: {
                'score': int,           # 局面評分(analyze=False時)
                'scores': List[int],    # 所有動作評分(analyze=True時)
                'move_sequence': str,   # 輸入的移動序列
                'solve_time': float,    # 求解時間(秒)
                'valid': bool          # 是否為有效局面
            }
        """
        # 構建命令參數
        cmd = [self.solver_path]
        if weak:
            cmd.append('-w')
        if analyze:
            cmd.append('-a')
        
        try:
            start_time = time.time()
            
            # 執行solver
            result = subprocess.run(
                cmd,
                input=move_sequence + '\n',
                text=True,
                capture_output=True,
                timeout=self.timeout
            )
            
            solve_time = time.time() - start_time
            
            if result.returncode != 0:
                logger.warning(f"C4Solver error: {result.stderr.strip()}")
                return {
                    'score': 0,
                    'scores': [0] * 7,
                    'move_sequence': move_sequence,
                    'solve_time': solve_time,
                    'valid': False
                }
            
            # 解析輸出 - 跳過 "Loading opening book" 行
            output_lines = result.stdout.strip().split('\n')
            output_line = None
            for line in output_lines:
                if not line.startswith('Loading') and line.strip():
                    output_line = line.strip()
                    break
            
            if not output_line:
                return {
                    'score': 0,
                    'scores': [0] * 7,
                    'move_sequence': move_sequence,
                    'solve_time': solve_time,
                    'valid': False
                }
            
            parts = output_line.split()
            
            if analyze:
                # 分析模式：期望格式 "序列 分數1 分數2 ... 分數7"
                # 對於空序列，格式為 " 分數1 分數2 ... 分數7"
                if len(parts) >= 7:  # 至少7個評分
                    scores = []
                    # 如果第一個部分是序列或空白，從適當位置開始解析
                    start_idx = 1 if (len(parts) > 7 or (parts[0] == move_sequence and move_sequence)) else 0
                    
                    for i in range(start_idx, start_idx + 7):
                        try:
                            if i < len(parts):
                                scores.append(int(parts[i]))
                            else:
                                scores.append(0)
                        except (ValueError, IndexError):
                            scores.append(0)
                    
                    return {
                        'score': max(scores) if scores else 0,
                        'scores': scores,
                        'move_sequence': move_sequence,
                        'solve_time': solve_time,
                        'valid': True
                    }
            else:
                # 標準模式：返回局面評分
                if len(parts) >= 2:
                    try:
                        score = int(parts[1])
                        return {
                            'score': score,
                            'scores': [score] * 7,  # 為了一致性
                            'move_sequence': move_sequence,
                            'solve_time': solve_time,
                            'valid': True
                        }
                    except ValueError:
                        pass
            
            return {
                'score': 0,
                'scores': [0] * 7,
                'move_sequence': move_sequence,
                'solve_time': solve_time,
                'valid': False
            }
            
        except subprocess.TimeoutExpired:
            logger.warning(f"C4Solver timeout for sequence: {move_sequence}")
            return {
                'score': 0,
                'scores': [0] * 7,
                'move_sequence': move_sequence,
                'solve_time': self.timeout,
                'valid': False
            }
        except Exception as e:
            logger.error(f"C4Solver execution error: {e}")
            return {
                'score': 0,
                'scores': [0] * 7,
                'move_sequence': move_sequence,
                'solve_time': 0.0,
                'valid': False
            }
    
    def evaluate_board(self, board: List[int], analyze: bool = False) -> Dict:
        """
        評估棋盤局面
        
        Args:
            board: 42長度的扁平化棋盤
            analyze: 是否分析所有可能移動
        
        Returns:
            求解結果字典
        """
        move_sequence = self.board_to_move_sequence(board)
        return self.solve_position(move_sequence, weak=False, analyze=analyze)
    
    def get_best_move(self, board: List[int], valid_actions: List[int]) -> Tuple[int, float]:
        """
        獲取最佳移動
        
        Args:
            board: 42長度的扁平化棋盤
            valid_actions: 可用動作列表(0-6)
        
        Returns:
            (best_action, confidence_score)
        """
        try:
            result = self.evaluate_board(board, analyze=True)
            
            if not result['valid'] or len(result['scores']) != 7:
                # Fallback策略
                center_preference = [3, 4, 2, 5, 1, 6, 0]
                for col in center_preference:
                    if col in valid_actions:
                        return col, 0.1
                return valid_actions[0] if valid_actions else 0, 0.0
            
            # 找出有效動作中的最佳評分
            best_score = float('-inf')
            best_action = valid_actions[0] if valid_actions else 0
            
            for action in valid_actions:
                if action < len(result['scores']):
                    score = result['scores'][action]
                    if score > best_score:
                        best_score = score
                        best_action = action
            
            # 計算信心分數(基於評分差異)
            scores_array = np.array([result['scores'][a] for a in valid_actions if a < len(result['scores'])])
            if len(scores_array) > 1:
                confidence = min(1.0, abs(best_score - np.mean(scores_array)) / 10.0)
            else:
                confidence = 0.5
            
            return best_action, confidence
            
        except Exception as e:
            logger.error(f"Get best move failed: {e}")
            # Fallback策略
            center_preference = [3, 4, 2, 5, 1, 6, 0]
            for col in center_preference:
                if col in valid_actions:
                    return col, 0.1
            return valid_actions[0] if valid_actions else 0, 0.0


# 全局C4Solver實例
_c4solver = None

def get_c4solver() -> Optional[C4SolverWrapper]:
    """獲取全局C4Solver實例"""
    global _c4solver
    if _c4solver is None:
        try:
            # 嘗試不同的路徑
            possible_paths = [
                "connect4/c4solver",
                "./connect4/c4solver", 
                "c4solver",
                "./c4solver"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    _c4solver = C4SolverWrapper(path)
                    break
            
            if _c4solver is None:
                logger.warning("C4Solver not found in any of the expected paths")
                
        except Exception as e:
            logger.warning(f"C4Solver initialization failed: {e}")
            _c4solver = None
    
    return _c4solver
