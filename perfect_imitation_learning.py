#!/usr/bin/env python3
"""
Connect4 強化模仿學習系統 - 完全重寫版本
確保模型100%學會C4Solver的完整策略
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import yaml
import logging
from collections import deque, defaultdict
from datetime import datetime
from typing import List, Dict, Tuple, Set
import time
import json

# 導入必要的組件
from train_connectx_rl_robust import ConnectXNet, PPOAgent, flat_to_2d, is_winning_move, find_drop_row, is_win_from
from c4solver_wrapper import get_c4solver, C4SolverWrapper
from kaggle_environments import make

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_imitation_learning.log')
    ]
)
logger = logging.getLogger(__name__)

class SystematicPositionGenerator:
    """系統化局面生成器 - 確保覆蓋所有類型的局面"""
    
    def __init__(self, solver: C4SolverWrapper):
        self.solver = solver
        self.generated_positions = set()  # 避免重複局面
        self.position_stats = defaultdict(int)  # 統計不同深度的局面數量
        
    def _position_to_key(self, board: List[int]) -> str:
        """將局面轉換為唯一鍵值"""
        return ''.join(map(str, board))
    
    def generate_systematic_positions(self, target_per_depth: int = 500) -> List[Tuple[List[int], int]]:
        """系統化生成不同深度的局面"""
        positions = []
        
        logger.info("🎯 系統化生成訓練局面...")
        
        # 0. 空局面 (重要!)
        empty_board = [0] * 42
        positions.append((empty_board.copy(), 1))
        positions.append((empty_board.copy(), 2))
        logger.info("✅ 添加空局面")
        
        # 1. 開局局面 (1-6步)
        logger.info("生成開局局面 (1-6步)...")
        opening_positions = self._generate_opening_positions(target_per_depth // 3)
        positions.extend(opening_positions)
        logger.info(f"✅ 生成 {len(opening_positions)} 個開局局面")
        
        # 2. 中局局面 (7-20步)
        logger.info("生成中局局面 (7-20步)...")
        midgame_positions = self._generate_midgame_positions(target_per_depth // 2)
        positions.extend(midgame_positions)
        logger.info(f"✅ 生成 {len(midgame_positions)} 個中局局面")
        
        # 3. 終局局面 (21+步)
        logger.info("生成終局局面 (21+步)...")
        endgame_positions = self._generate_endgame_positions(target_per_depth // 3)
        positions.extend(endgame_positions)
        logger.info(f"✅ 生成 {len(endgame_positions)} 個終局局面")
        
        # 4. 戰術局面 (有威脅的局面)
        logger.info("生成戰術局面...")
        tactical_positions = self._generate_tactical_positions(target_per_depth // 4)
        positions.extend(tactical_positions)
        logger.info(f"✅ 生成 {len(tactical_positions)} 個戰術局面")
        
        # 去重
        unique_positions = []
        seen_keys = set()
        for board, player in positions:
            key = self._position_to_key(board) + f"_p{player}"
            if key not in seen_keys:
                seen_keys.add(key)
                unique_positions.append((board, player))
        
        logger.info(f"🎯 總共生成 {len(unique_positions)} 個獨特局面")
        return unique_positions
    
    def _generate_opening_positions(self, target_count: int) -> List[Tuple[List[int], int]]:
        """生成開局局面 (1-6步)"""
        positions = []
        
        # 重要的開局模式
        important_openings = [
            "4",      # 中央開局
            "43",     # 中央 + 偏移
            "44",     # 中央疊加
            "443",    # 經典開局
            "4433",   # 對稱開局
            "4343",   # 交替中央
            "3",      # 左中開局
            "34",     # 左中轉中央
            "343",    # 戰術性開局
            "3423",   # 複雜開局
        ]
        
        for opening in important_openings:
            try:
                board = self._apply_move_sequence(opening)
                if board:
                    # 添加兩個玩家的視角
                    current_player = (len(opening) % 2) + 1
                    positions.append((board, current_player))
                    positions.append((board, 3 - current_player))
            except Exception as e:
                logger.warning(f"開局序列 {opening} 失敗: {e}")
                continue
        
        # 隨機開局變化
        for _ in range(target_count - len(positions)):
            try:
                depth = random.randint(1, 6)
                board = self._generate_random_game(max_moves=depth)
                if board and sum(1 for x in board if x != 0) == depth:
                    current_player = (depth % 2) + 1
                    positions.append((board, current_player))
            except Exception:
                continue
                
        return positions[:target_count]
    
    def _generate_midgame_positions(self, target_count: int) -> List[Tuple[List[int], int]]:
        """生成中局局面 (7-20步)"""
        positions = []
        
        for _ in range(target_count):
            try:
                depth = random.randint(7, 20)
                board = self._generate_random_game(max_moves=depth)
                if board and 7 <= sum(1 for x in board if x != 0) <= 20:
                    current_player = (depth % 2) + 1
                    positions.append((board, current_player))
            except Exception:
                continue
                
        return positions
    
    def _generate_endgame_positions(self, target_count: int) -> List[Tuple[List[int], int]]:
        """生成終局局面 (21+步)"""
        positions = []
        
        for _ in range(target_count):
            try:
                depth = random.randint(21, 35)
                board = self._generate_random_game(max_moves=depth)
                if board and sum(1 for x in board if x != 0) >= 21:
                    current_player = (depth % 2) + 1
                    positions.append((board, current_player))
            except Exception:
                continue
                
        return positions
    
    def _generate_tactical_positions(self, target_count: int) -> List[Tuple[List[int], int]]:
        """生成有戰術意義的局面 (有威脅、需要防守等)"""
        positions = []
        attempts = 0
        max_attempts = target_count * 10
        
        while len(positions) < target_count and attempts < max_attempts:
            attempts += 1
            try:
                # 生成隨機局面
                depth = random.randint(8, 25)
                board = self._generate_random_game(max_moves=depth)
                if not board:
                    continue
                
                # 檢查是否有戰術意義
                player1_can_win = self._has_immediate_threat(board, 1)
                player2_can_win = self._has_immediate_threat(board, 2)
                has_multiple_threats = self._count_threats(board) >= 2
                
                if player1_can_win or player2_can_win or has_multiple_threats:
                    current_player = (depth % 2) + 1
                    positions.append((board, current_player))
                    
            except Exception:
                continue
        
        return positions
    
    def _apply_move_sequence(self, sequence: str) -> List[int]:
        """應用移動序列生成局面"""
        board = [0] * 42
        current_player = 1
        
        for move_char in sequence:
            col = int(move_char) - 1  # C4Solver使用1-7，我們使用0-6
            if col < 0 or col > 6:
                continue
                
            # 找到該列的落點
            grid = flat_to_2d(board)
            row = find_drop_row(grid, col)
            if row is None:
                return None  # 該列已滿
            
            board[row * 7 + col] = current_player
            current_player = 3 - current_player
        
        return board
    
    def _generate_random_game(self, max_moves: int) -> List[int]:
        """生成隨機對局到指定步數"""
        board = [0] * 42
        current_player = 1
        moves_made = 0
        
        while moves_made < max_moves:
            valid_actions = [c for c in range(7) if board[c] == 0]
            if not valid_actions:
                break
            
            col = random.choice(valid_actions)
            grid = flat_to_2d(board)
            row = find_drop_row(grid, col)
            if row is None:
                break
            
            board[row * 7 + col] = current_player
            
            # 檢查是否遊戲結束
            if is_win_from(grid, row, col, current_player):
                break
                
            current_player = 3 - current_player
            moves_made += 1
        
        return board
    
    def _has_immediate_threat(self, board: List[int], player: int) -> bool:
        """檢查玩家是否有立即獲勝威脅"""
        valid_actions = [c for c in range(7) if board[c] == 0]
        for col in valid_actions:
            if is_winning_move(board, col, player):
                return True
        return False
    
    def _count_threats(self, board: List[int]) -> int:
        """計算局面中的威脅數量"""
        threats = 0
        for player in [1, 2]:
            if self._has_immediate_threat(board, player):
                threats += 1
        return threats

class PerfectExpertPolicy:
    """完美專家策略 - 直接使用C4Solver的最優決策"""
    
    def __init__(self, solver: C4SolverWrapper):
        self.solver = solver
        self.cache = {}  # 緩存C4Solver結果
        
    def get_expert_policy(self, board: List[int], valid_actions: List[int]) -> np.ndarray:
        """獲取C4Solver的完美策略分佈
        
        重要：不使用softmax！直接使用C4Solver的最優動作作為one-hot分佈
        """
        # 快速緩存查找
        board_key = ''.join(map(str, board))
        if board_key in self.cache:
            return self.cache[board_key]
        
        try:
            # 獲取C4Solver分析
            result = self.solver.evaluate_board(board, analyze=True)
            
            if not result['valid'] or len(result['scores']) != 7:
                # Fallback: 均勻分佈
                policy = np.zeros(7)
                for action in valid_actions:
                    policy[action] = 1.0 / len(valid_actions)
                self.cache[board_key] = policy
                return policy
            
            scores = np.array(result['scores'])
            
            # 找出有效動作中的最高分數
            best_score = float('-inf')
            best_actions = []
            
            for action in valid_actions:
                if action < len(scores):
                    score = scores[action]
                    if score > best_score:
                        best_score = score
                        best_actions = [action]
                    elif score == best_score:
                        best_actions.append(action)
            
            # 創建策略分佈：最優動作獲得100%概率，其他為0
            policy = np.zeros(7)
            for action in best_actions:
                policy[action] = 1.0 / len(best_actions)
            
            self.cache[board_key] = policy
            return policy
            
        except Exception as e:
            logger.warning(f"Expert policy failed: {e}")
            # Fallback策略
            policy = np.zeros(7)
            if valid_actions:
                # 偏好中央
                center_preference = [3, 4, 2, 5, 1, 6, 0]
                for col in center_preference:
                    if col in valid_actions:
                        policy[col] = 1.0
                        break
            
            self.cache[board_key] = policy
            return policy

class EnhancedImitationDataset:
    """增強的模仿學習數據集"""
    
    def __init__(self, solver: C4SolverWrapper):
        self.solver = solver
        self.position_generator = SystematicPositionGenerator(solver)
        self.expert_policy = PerfectExpertPolicy(solver)
        self.training_samples = []
        
    def generate_comprehensive_dataset(self, positions_per_depth: int = 1000) -> List[Dict]:
        """生成完整的訓練數據集"""
        logger.info("🎯 開始生成完整訓練數據集...")
        
        # 第一步：系統化生成局面
        positions = self.position_generator.generate_systematic_positions(positions_per_depth)
        logger.info(f"✅ 生成了 {len(positions)} 個訓練局面")
        
        # 第二步：為每個局面生成專家策略
        samples = []
        failed_count = 0
        
        for i, (board, current_player) in enumerate(positions):
            if i % 500 == 0:
                logger.info(f"處理進度: {i}/{len(positions)} ({100*i/len(positions):.1f}%)")
            
            try:
                # 檢查遊戲是否已結束
                if self._is_game_over(board):
                    continue
                
                # 獲取有效動作
                valid_actions = [c for c in range(7) if board[c] == 0]
                if not valid_actions:
                    continue
                
                # 編碼狀態
                encoded_state = self._encode_state(board, current_player)
                
                # 獲取專家策略
                expert_policy = self.expert_policy.get_expert_policy(board, valid_actions)
                
                # 驗證策略合理性
                if np.sum(expert_policy) == 0:
                    failed_count += 1
                    continue
                
                sample = {
                    'state': encoded_state,
                    'policy': expert_policy,
                    'board': board.copy(),
                    'player': current_player,
                    'valid_actions': valid_actions.copy(),
                    'move_count': sum(1 for x in board if x != 0)
                }
                
                samples.append(sample)
                
            except Exception as e:
                failed_count += 1
                if failed_count < 10:  # 只記錄前幾個錯誤
                    logger.warning(f"樣本生成失敗: {e}")
                continue
        
        logger.info(f"✅ 成功生成 {len(samples)} 個訓練樣本")
        logger.info(f"⚠️ 失敗 {failed_count} 個樣本")
        
        # 分析數據集統計
        self._analyze_dataset(samples)
        
        return samples
    
    def _is_game_over(self, board: List[int]) -> bool:
        """檢查遊戲是否結束"""
        grid = flat_to_2d(board)
        
        # 檢查是否有人獲勝
        for r in range(6):
            for c in range(7):
                if board[r*7 + c] != 0:
                    if is_win_from(grid, r, c, board[r*7 + c]):
                        return True
        
        # 檢查是否棋盤已滿
        return all(board[i] != 0 for i in range(7))
    
    def _encode_state(self, board: List[int], current_player: int) -> np.ndarray:
        """編碼狀態為模型輸入"""
        try:
            # 基本棋盤編碼 (6*7*2 = 84維)
            encoded = np.zeros((6, 7, 2), dtype=np.float32)
            grid = flat_to_2d(board)
            
            for r in range(6):
                for c in range(7):
                    if grid[r][c] == 1:
                        encoded[r, c, 0] = 1
                    elif grid[r][c] == 2:
                        encoded[r, c, 1] = 1
            
            # 當前玩家編碼 (1維)
            current_player_encoded = np.array([current_player - 1], dtype=np.float32)
            
            # 有效動作編碼 (7維)
            valid_actions = [c for c in range(7) if board[c] == 0]
            valid_mask = np.zeros(7, dtype=np.float32)
            for action in valid_actions:
                valid_mask[action] = 1
            
            # 位置特徵 (34維)
            position_features = np.zeros(34, dtype=np.float32)
            
            # 組合所有特徵 (84 + 1 + 7 + 34 = 126維)
            combined = np.concatenate([
                encoded.flatten(),      # 84
                current_player_encoded, # 1  
                valid_mask,            # 7
                position_features      # 34
            ])
            
            return combined
            
        except Exception as e:
            logger.error(f"狀態編碼失敗: {e}")
            return np.zeros(126, dtype=np.float32)
    
    def _analyze_dataset(self, samples: List[Dict]):
        """分析數據集統計信息"""
        if not samples:
            return
        
        move_counts = [s['move_count'] for s in samples]
        player_counts = [s['player'] for s in samples]
        
        logger.info("📊 數據集統計:")
        logger.info(f"  總樣本數: {len(samples)}")
        logger.info(f"  移動步數分佈: 最小={min(move_counts)}, 最大={max(move_counts)}, 平均={np.mean(move_counts):.1f}")
        logger.info(f"  玩家分佈: P1={player_counts.count(1)}, P2={player_counts.count(2)}")
        
        # 分析策略分佈
        policy_entropies = []
        max_probs = []
        for sample in samples[:100]:  # 分析前100個樣本
            policy = sample['policy']
            entropy = -np.sum(policy * np.log(policy + 1e-8))
            max_prob = np.max(policy)
            policy_entropies.append(entropy)
            max_probs.append(max_prob)
        
        logger.info(f"  策略熵值範圍: {np.min(policy_entropies):.3f} - {np.max(policy_entropies):.3f}")
        logger.info(f"  最大概率範圍: {np.min(max_probs):.3f} - {np.max(max_probs):.3f}")

class EnhancedImitationLearner:
    """增強的模仿學習訓練器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🖥️ 使用設備: {self.device}")
        
        # 初始化C4Solver
        self.solver = get_c4solver()
        if self.solver is None:
            raise RuntimeError("無法初始化C4Solver")
        
        # 初始化數據集
        self.dataset = EnhancedImitationDataset(self.solver)
        
        # 初始化模型
        self.model = ConnectXNet(
            input_size=config['model']['input_size'],
            hidden_size=config['model']['hidden_size'], 
            num_layers=config['model']['num_layers']
        ).to(self.device)
        
        # 優化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 學習率調度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=8
        )
        
        # 訓練統計
        self.training_stats = {
            'losses': [],
            'accuracies': [],
            'learning_rates': [],
            'policy_kl_divs': []
        }
        
    def train(self, positions_per_depth: int = 1000, num_epochs: int = 200, 
              batch_size: int = 64, val_split: float = 0.15, 
              save_path: str = "perfect_imitation_model.pt"):
        """主訓練循環"""
        logger.info("🚀 開始完美模仿學習訓練...")
        
        # 生成訓練數據
        all_samples = self.dataset.generate_comprehensive_dataset(positions_per_depth)
        
        if len(all_samples) < 100:
            raise RuntimeError(f"訓練樣本太少: {len(all_samples)}")
        
        # 分割數據集
        random.shuffle(all_samples)
        val_size = int(len(all_samples) * val_split)
        train_samples = all_samples[:-val_size] if val_size > 0 else all_samples
        val_samples = all_samples[-val_size:] if val_size > 0 else []
        
        logger.info(f"📊 數據集分割: 訓練={len(train_samples)}, 驗證={len(val_samples)}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 15
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 訓練一個epoch
            train_loss, train_acc, train_kl = self._train_epoch(train_samples, batch_size)
            
            # 驗證
            val_loss, val_acc, val_kl = self._evaluate(val_samples, batch_size) if val_samples else (0, 0, 0)
            
            # 更新學習率
            self.scheduler.step(val_loss if val_samples else train_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 記錄統計
            self.training_stats['losses'].append(train_loss)
            self.training_stats['accuracies'].append(train_acc)
            self.training_stats['learning_rates'].append(current_lr)
            self.training_stats['policy_kl_divs'].append(train_kl)
            
            epoch_time = time.time() - start_time
            
            # 日誌輸出
            log_msg = (f"Epoch {epoch+1:3d}/{num_epochs}: "
                      f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
                      f"train_kl={train_kl:.4f}")
            
            if val_samples:
                log_msg += f", val_loss={val_loss:.4f}, val_acc={val_acc:.3f}, val_kl={val_kl:.4f}"
            
            log_msg += f", lr={current_lr:.2e}, time={epoch_time:.1f}s"
            logger.info(log_msg)
            
            # 保存最佳模型
            current_loss = val_loss if val_samples else train_loss
            if current_loss < best_val_loss:
                best_val_loss = current_loss
                patience_counter = 0
                self._save_model(save_path.replace('.pt', '_best.pt'), epoch, current_loss)
                logger.info(f"🏆 新的最佳模型! 損失: {current_loss:.4f}")
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= max_patience:
                logger.info(f"⏰ 早停觸發 (耐心={max_patience})")
                break
            
            # 定期保存
            if (epoch + 1) % 20 == 0:
                self._save_model(save_path.replace('.pt', f'_epoch_{epoch+1}.pt'), epoch, current_loss)
        
        # 保存最終模型
        self._save_model(save_path, epoch, train_loss)
        logger.info("✅ 完美模仿學習訓練完成!")
        
        # 評估最終性能
        self._final_evaluation()
    
    def _train_epoch(self, samples: List[Dict], batch_size: int) -> Tuple[float, float, float]:
        """訓練一個epoch"""
        self.model.train()
        random.shuffle(samples)
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_kl_div = 0.0
        num_batches = 0
        
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            
            # 準備批次數據 - 優化tensor創建
            states_array = np.array([s['state'] for s in batch], dtype=np.float32)
            policies_array = np.array([s['policy'] for s in batch], dtype=np.float32)
            
            states = torch.from_numpy(states_array).to(self.device)
            target_policies = torch.from_numpy(policies_array).to(self.device)
            
            # 前向傳播
            self.optimizer.zero_grad()
            pred_policies, pred_values = self.model(states)
            
            # 計算損失 - 使用交叉熵而不是MSE
            policy_loss = self._policy_loss(pred_policies, target_policies)
            
            # 反向傳播
            policy_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 統計
            with torch.no_grad():
                accuracy = self._calculate_accuracy(pred_policies, target_policies)
                kl_div = self._calculate_kl_divergence(pred_policies, target_policies)
                
                total_loss += policy_loss.item()
                total_accuracy += accuracy
                total_kl_div += kl_div
                num_batches += 1
        
        return (total_loss / max(1, num_batches), 
                total_accuracy / max(1, num_batches),
                total_kl_div / max(1, num_batches))
    
    def _evaluate(self, samples: List[Dict], batch_size: int) -> Tuple[float, float, float]:
        """評估模型"""
        if not samples:
            return 0.0, 0.0, 0.0
            
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_kl_div = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i + batch_size]
                
                # 優化tensor創建
                states_array = np.array([s['state'] for s in batch], dtype=np.float32)
                policies_array = np.array([s['policy'] for s in batch], dtype=np.float32)
                
                states = torch.from_numpy(states_array).to(self.device)
                target_policies = torch.from_numpy(policies_array).to(self.device)
                
                pred_policies, pred_values = self.model(states)
                
                loss = self._policy_loss(pred_policies, target_policies)
                accuracy = self._calculate_accuracy(pred_policies, target_policies)
                kl_div = self._calculate_kl_divergence(pred_policies, target_policies)
                
                total_loss += loss.item()
                total_accuracy += accuracy
                total_kl_div += kl_div
                num_batches += 1
        
        return (total_loss / max(1, num_batches),
                total_accuracy / max(1, num_batches),
                total_kl_div / max(1, num_batches))
    
    def _policy_loss(self, pred_policies: torch.Tensor, target_policies: torch.Tensor) -> torch.Tensor:
        """計算策略損失 - 使用KL散度"""
        # 添加小值避免log(0)
        pred_policies = torch.softmax(pred_policies, dim=1) + 1e-8
        target_policies = target_policies + 1e-8
        
        # KL散度: sum(target * log(target / pred))
        kl_loss = torch.sum(target_policies * torch.log(target_policies / pred_policies), dim=1)
        return torch.mean(kl_loss)
    
    def _calculate_accuracy(self, pred_policies: torch.Tensor, target_policies: torch.Tensor) -> float:
        """計算預測準確率"""
        pred_actions = torch.argmax(pred_policies, dim=1)
        target_actions = torch.argmax(target_policies, dim=1)
        correct = (pred_actions == target_actions).float()
        return torch.mean(correct).item()
    
    def _calculate_kl_divergence(self, pred_policies: torch.Tensor, target_policies: torch.Tensor) -> float:
        """計算KL散度"""
        pred_policies = torch.softmax(pred_policies, dim=1) + 1e-8
        target_policies = target_policies + 1e-8
        
        kl_div = torch.sum(target_policies * torch.log(target_policies / pred_policies), dim=1)
        return torch.mean(kl_div).item()
    
    def _save_model(self, save_path: str, epoch: int, loss: float):
        """保存模型"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'config': self.config,
                'training_stats': self.training_stats
            }, save_path)
        except Exception as e:
            logger.error(f"保存模型失敗: {e}")
    
    def _final_evaluation(self):
        """最終評估"""
        logger.info("🔍 進行最終性能評估...")
        
        # 測試幾個關鍵局面
        test_positions = [
            ([0] * 42, "空局面"),
            ([0]*35 + [1, 0, 0, 0, 0, 0, 0], "中央開局"),
        ]
        
        for board, description in test_positions:
            try:
                valid_actions = [c for c in range(7) if board[c] == 0]
                if not valid_actions:
                    continue
                
                # 模型預測
                state = self.dataset._encode_state(board, 1)
                with torch.no_grad():
                    state_tensor = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
                    pred_policy, _ = self.model(state_tensor)
                    pred_policy = torch.softmax(pred_policy, dim=1).cpu().numpy()[0]
                
                # C4Solver預測
                expert_policy = self.dataset.expert_policy.get_expert_policy(board, valid_actions)
                
                # 比較
                model_action = np.argmax(pred_policy)
                expert_action = np.argmax(expert_policy)
                
                logger.info(f"📋 {description}:")
                logger.info(f"  專家動作: {expert_action}, 模型動作: {model_action}, 匹配: {'✅' if model_action == expert_action else '❌'}")
                logger.info(f"  專家策略: {expert_policy}")
                logger.info(f"  模型策略: {pred_policy}")
                
            except Exception as e:
                logger.warning(f"評估 {description} 失敗: {e}")

def load_enhanced_config(config_path: str = "enhanced_imitation_config.yaml") -> Dict:
    """載入增強配置"""
    default_config = {
        'model': {
            'input_size': 126,
            'hidden_size': 512,
            'num_layers': 32
        },
        'training': {
            'learning_rate': 0.0005,
            'weight_decay': 0.0001,
            'positions_per_depth': 200,  # 每個深度的局面數
            'num_epochs': 200,
            'batch_size': 64,
            'val_split': 0.15
        },
        'paths': {
            'output_model': 'perfect_imitation_model.pt'
        }
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
            
            # 深度合併配置
            for section, values in user_config.items():
                if section in default_config:
                    default_config[section].update(values)
                else:
                    default_config[section] = values
                    
        except Exception as e:
            logger.warning(f"載入配置文件失敗: {e}，使用默認配置")
    
    return default_config

def main():
    """主函數"""
    logger.info("🎯 Connect4 完美模仿學習系統")
    logger.info("=" * 60)
    
    try:
        # 載入配置
        config = load_enhanced_config()
        logger.info("✅ 配置載入完成")
        
        # 創建訓練器
        trainer = EnhancedImitationLearner(config)
        
        # 開始訓練
        training_config = config['training']
        output_path = config['paths']['output_model']
        
        trainer.train(
            positions_per_depth=training_config['positions_per_depth'],
            num_epochs=training_config['num_epochs'],
            batch_size=training_config['batch_size'],
            val_split=training_config['val_split'],
            save_path=output_path
        )
        
        logger.info("🎉 完美模仿學習訓練完成!")
        logger.info(f"📁 模型已保存: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ 訓練失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
