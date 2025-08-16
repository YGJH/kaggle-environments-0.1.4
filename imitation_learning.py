#!/usr/bin/env python3
"""
Connect4 模仿學習預訓練
使用C4Solver進行監督學習預訓練，然後可接入RL訓練
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
from collections import deque
from datetime import datetime
from typing import List, Dict, Tuple
import time

# 導入必要的組件
from train_connectx_rl_robust import ConnectXNet, PPOAgent
from c4solver_wrapper import get_c4solver, C4SolverWrapper
from kaggle_environments import make

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('imitation_learning.log')
    ]
)
logger = logging.getLogger(__name__)

class ImitationDataset:
    """模仿學習數據集"""
    
    def __init__(self, solver: C4SolverWrapper, max_moves: int = 20):
        """
        Args:
            solver: C4Solver包裝器
            max_moves: 每局最大移動數(避免過長的遊戲)
        """
        self.solver = solver
        self.max_moves = max_moves
        self.data = []
        
    def generate_position(self) -> Tuple[List[int], int]:
        """
        生成一個隨機局面
        
        Returns:
            (board, current_player): 棋盤狀態和當前玩家
        """
        env = make('connectx', debug=False)
        env.reset()
        
        board = [0] * 42
        current_player = 1
        move_count = 0
        
        # 隨機遊戲到指定步數
        target_moves = random.randint(0, min(self.max_moves, 30))
        
        while move_count < target_moves and not env.done:
            # 獲取有效動作
            valid_actions = [c for c in range(7) if board[c] == 0]
            if not valid_actions:
                break
                
            # 隨機選擇動作
            action = random.choice(valid_actions)
            
            # 執行動作
            env.step([action if current_player == 1 else None, 
                     action if current_player == 2 else None])
            
            # 更新棋盤
            board = env.state[0]['observation']['board']
            current_player = 3 - current_player
            move_count += 1
            
            # 檢查遊戲是否結束
            if env.done:
                break
        
        return board, current_player
    
    def get_expert_action_distribution(self, board: List[int], current_player: int) -> np.ndarray:
        """
        使用C4Solver獲取專家動作分佈
        
        Args:
            board: 棋盤狀態
            current_player: 當前玩家(1或2)
            
        Returns:
            7維的動作概率分佈
        """
        try:
            # 獲取有效動作
            valid_actions = [c for c in range(7) if board[c] == 0]
            if not valid_actions:
                return np.ones(7) / 7  # 均勻分佈作為fallback
            
            # 獲取C4Solver分析結果
            result = self.solver.evaluate_board(board, analyze=True)
            
            if not result['valid'] or len(result['scores']) != 7:
                # Fallback: 均勻分佈於有效動作
                dist = np.zeros(7)
                for action in valid_actions:
                    dist[action] = 1.0 / len(valid_actions)
                return dist
            
            scores = np.array(result['scores'])
            
            # 只考慮有效動作的分數
            valid_scores = np.full(7, -1000)  # 無效動作給極低分
            for action in valid_actions:
                valid_scores[action] = scores[action]
            
            # 轉換為概率分佈 (softmax with temperature)
            temperature = 1.0
            exp_scores = np.exp(valid_scores / temperature)
            
            # 確保無效動作概率為0
            for i in range(7):
                if i not in valid_actions:
                    exp_scores[i] = 0
            
            # 歸一化
            prob_dist = exp_scores / (np.sum(exp_scores) + 1e-8)
            
            return prob_dist
            
        except Exception as e:
            logger.warning(f"Expert action failed: {e}")
            # Fallback: 中央偏好分佈
            dist = np.zeros(7)
            center_weights = [0.05, 0.1, 0.15, 0.4, 0.15, 0.1, 0.05]
            for i, action in enumerate([0, 1, 2, 3, 4, 5, 6]):
                if action in valid_actions:
                    dist[action] = center_weights[i]
            
            if np.sum(dist) > 0:
                dist = dist / np.sum(dist)
            else:
                dist = np.ones(7) / 7
                
            return dist
    
    def generate_training_samples(self, num_samples: int) -> List[Dict]:
        """
        生成訓練樣本
        
        Args:
            num_samples: 樣本數量
            
        Returns:
            訓練樣本列表
        """
        samples = []
        
        logger.info(f"生成 {num_samples} 個訓練樣本...")
        
        for i in range(num_samples):
            if i % 1000 == 0:
                logger.info(f"進度: {i}/{num_samples}")
            
            try:
                # 生成隨機局面
                board, current_player = self.generate_position()
                
                # 檢查遊戲是否已結束
                env = make('connectx', debug=False)
                env.state = [{'observation': {'board': board, 'mark': current_player}}, 
                           {'observation': {'board': board, 'mark': 3-current_player}}]
                
                if env.done:
                    continue
                
                # 編碼狀態 (使用PPOAgent的編碼方式)
                encoded_state = self.encode_state_for_model(board, current_player)
                
                # 獲取專家動作分佈
                expert_dist = self.get_expert_action_distribution(board, current_player)
                
                samples.append({
                    'state': encoded_state,
                    'action_dist': expert_dist,
                    'board': board.copy(),
                    'player': current_player
                })
                
            except Exception as e:
                logger.warning(f"Sample {i} generation failed: {e}")
                continue
        
        logger.info(f"成功生成 {len(samples)} 個訓練樣本")
        return samples
    
    def encode_state_for_model(self, board: List[int], mark: int) -> np.ndarray:
        """
        編碼狀態為模型輸入格式 (126維)
        
        Args:
            board: 棋盤狀態
            mark: 當前玩家標記
            
        Returns:
            編碼後的狀態
        """
        try:
            # 轉換為6x7格式
            grid = np.array(board).reshape(6, 7)
            
            # 創建3通道編碼
            # 通道0: 當前玩家的棋子
            # 通道1: 對手的棋子  
            # 通道2: 空位
            encoded = np.zeros((3, 6, 7), dtype=np.float32)
            
            for r in range(6):
                for c in range(7):
                    if grid[r, c] == mark:
                        encoded[0, r, c] = 1.0
                    elif grid[r, c] == (3 - mark):
                        encoded[1, r, c] = 1.0
                    else:
                        encoded[2, r, c] = 1.0
            
            # 展平為126維
            return encoded.flatten()
            
        except Exception as e:
            logger.error(f"State encoding failed: {e}")
            return np.zeros(126, dtype=np.float32)


class ImitationLearner:
    """模仿學習訓練器"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化C4Solver
        self.solver = get_c4solver()
        if self.solver is None:
            raise RuntimeError("無法初始化C4Solver")
        
        # 初始化數據集
        self.dataset = ImitationDataset(self.solver)
        
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
            factor=0.8,
            patience=5,
            verbose=True
        )
        
        # 訓練統計
        self.training_stats = {
            'losses': [],
            'accuracies': [],
            'learning_rates': []
        }
        
    def load_model(self, model_path: str):
        """載入預訓練模型"""
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"✅ 成功載入模型: {model_path}")
                else:
                    self.model.load_state_dict(checkpoint)
                    logger.info(f"✅ 成功載入模型狀態: {model_path}")
            except Exception as e:
                logger.warning(f"載入模型失敗: {e}")
                logger.info("將使用隨機初始化的模型")
        else:
            logger.info(f"模型文件不存在: {model_path}，使用隨機初始化")
    
    def save_model(self, save_path: str, epoch: int = 0, loss: float = 0.0):
        """保存模型"""
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'config': self.config,
                'training_stats': self.training_stats
            }
            torch.save(checkpoint, save_path)
            logger.info(f"✅ 模型已保存: {save_path}")
        except Exception as e:
            logger.error(f"保存模型失敗: {e}")
    
    def train_epoch(self, samples: List[Dict], batch_size: int) -> Tuple[float, float]:
        """
        訓練一個epoch
        
        Args:
            samples: 訓練樣本
            batch_size: 批次大小
            
        Returns:
            (average_loss, accuracy)
        """
        self.model.train()
        
        # 打亂樣本
        random.shuffle(samples)
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i + batch_size]
            
            # 準備批次數據
            states = []
            target_dists = []
            
            for sample in batch_samples:
                states.append(sample['state'])
                target_dists.append(sample['action_dist'])
            
            # 轉換為張量
            states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
            target_dists_tensor = torch.FloatTensor(np.array(target_dists)).to(self.device)
            
            # 前向傳播
            policy_probs, _ = self.model(states_tensor)
            
            # 計算損失 (交叉熵損失)
            loss = nn.KLDivLoss(reduction='batchmean')(
                torch.log(policy_probs + 1e-8), 
                target_dists_tensor
            )
            
            # 反向傳播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 計算準確率 (預測動作是否與最佳動作一致)
            predicted_actions = torch.argmax(policy_probs, dim=1)
            target_actions = torch.argmax(target_dists_tensor, dim=1)
            accuracy = (predicted_actions == target_actions).float().mean().item()
            
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        avg_accuracy = total_accuracy / max(1, num_batches)
        
        return avg_loss, avg_accuracy
    
    def evaluate(self, samples: List[Dict], batch_size: int) -> Tuple[float, float]:
        """
        評估模型
        
        Args:
            samples: 評估樣本
            batch_size: 批次大小
            
        Returns:
            (average_loss, accuracy)
        """
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(samples), batch_size):
                batch_samples = samples[i:i + batch_size]
                
                # 準備批次數據
                states = []
                target_dists = []
                
                for sample in batch_samples:
                    states.append(sample['state'])
                    target_dists.append(sample['action_dist'])
                
                # 轉換為張量
                states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
                target_dists_tensor = torch.FloatTensor(np.array(target_dists)).to(self.device)
                
                # 前向傳播
                policy_probs, _ = self.model(states_tensor)
                
                # 計算損失
                loss = nn.KLDivLoss(reduction='batchmean')(
                    torch.log(policy_probs + 1e-8), 
                    target_dists_tensor
                )
                
                # 計算準確率
                predicted_actions = torch.argmax(policy_probs, dim=1)
                target_actions = torch.argmax(target_dists_tensor, dim=1)
                accuracy = (predicted_actions == target_actions).float().mean().item()
                
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        avg_accuracy = total_accuracy / max(1, num_batches)
        
        return avg_loss, avg_accuracy
    
    def train(self, num_samples: int = 10000, num_epochs: int = 50, batch_size: int = 64, 
              val_split: float = 0.1, save_path: str = "imitation_model.pt"):
        """
        主訓練循環
        
        Args:
            num_samples: 訓練樣本數量
            num_epochs: 訓練輪數
            batch_size: 批次大小
            val_split: 驗證集比例
            save_path: 模型保存路徑
        """
        logger.info("開始模仿學習訓練...")
        
        # 生成訓練數據
        logger.info("生成訓練數據...")
        all_samples = self.dataset.generate_training_samples(num_samples)
        
        if len(all_samples) == 0:
            logger.error("無法生成訓練樣本，訓練中止")
            return
        
        # 分割訓練和驗證集
        val_size = int(len(all_samples) * val_split)
        train_samples = all_samples[:-val_size] if val_size > 0 else all_samples
        val_samples = all_samples[-val_size:] if val_size > 0 else []
        
        logger.info(f"訓練樣本: {len(train_samples)}, 驗證樣本: {len(val_samples)}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 訓練
            train_loss, train_acc = self.train_epoch(train_samples, batch_size)
            
            # 驗證
            val_loss, val_acc = 0.0, 0.0
            if val_samples:
                val_loss, val_acc = self.evaluate(val_samples, batch_size)
            
            # 更新學習率
            self.scheduler.step(val_loss if val_samples else train_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 記錄統計
            self.training_stats['losses'].append(train_loss)
            self.training_stats['accuracies'].append(train_acc)
            self.training_stats['learning_rates'].append(current_lr)
            
            epoch_time = time.time() - start_time
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}, "
                f"lr={current_lr:.2e}, time={epoch_time:.1f}s"
            )
            
            # 保存最佳模型
            if val_samples and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_save_path = save_path.replace('.pt', '_best.pt')
                self.save_model(best_save_path, epoch, val_loss)
                logger.info(f"🏆 新的最佳模型! 驗證損失: {val_loss:.4f}")
            else:
                patience_counter += 1
            
            # 定期保存檢查點
            if (epoch + 1) % 10 == 0:
                checkpoint_path = save_path.replace('.pt', f'_epoch_{epoch+1}.pt')
                self.save_model(checkpoint_path, epoch, train_loss)
            
            # 早停
            if patience_counter >= max_patience:
                logger.info(f"Early stopping after {epoch+1} epochs (patience={max_patience})")
                break
        
        # 保存最終模型
        self.save_model(save_path, num_epochs, train_loss)
        logger.info("✅ 模仿學習訓練完成!")


def load_config(config_path: str = "imitation_config.yaml") -> Dict:
    """載入配置文件"""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"✅ 載入配置文件: {config_path}")
                return config
        except Exception as e:
            logger.warning(f"載入配置文件失敗: {e}")
    
    # 默認配置
    default_config = {
        'model': {
            'input_size': 126,
            'hidden_size': 192,
            'num_layers': 256
        },
        'training': {
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'num_samples': 20000,
            'num_epochs': 100,
            'batch_size': 128,
            'val_split': 0.1
        },
        'paths': {
            'input_model': None,  # 可選：載入現有模型
            'output_model': 'imitation_pretrained_model.pt'
        },
        'c4solver': {
            'path': './c4solver',
            'timeout': 5.0,
            'max_depth': 20
        }
    }
    
    logger.info("使用默認配置")
    return default_config


def main():
    """主函數"""
    logger.info("🎯 Connect4 模仿學習預訓練")
    logger.info("=" * 50)
    
    try:
        # 載入配置
        config = load_config()
        
        # 創建模仿學習器
        learner = ImitationLearner(config)
        
        # 載入現有模型 (如果指定)
        input_model_path = config.get('paths', {}).get('input_model')
        if input_model_path:
            learner.load_model(input_model_path)
        
        # 開始訓練
        training_config = config['training']
        output_path = config.get('paths', {}).get('output_model', 'imitation_pretrained_model.pt')
        
        learner.train(
            num_samples=training_config.get('num_samples', 20000),
            num_epochs=training_config.get('num_epochs', 100),
            batch_size=training_config.get('batch_size', 128),
            val_split=training_config.get('val_split', 0.1),
            save_path=output_path
        )
        
        logger.info("🎉 模仿學習預訓練完成!")
        logger.info(f"模型已保存: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ 訓練失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
