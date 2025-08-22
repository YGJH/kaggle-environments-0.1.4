#!/usr/bin/env python3

# uv run online_battle_trainer.py --main_ckpt checkpoints/best_model_wr_0.600.pt --winrate_threshold 0.8 --timesteps 10000 --max_rounds 1
"""
Online battle trainer: fine-tune a main agent against failed opponents until it beats them all.

Usage:
1. Run batch_dump_and_battle.py to identify weak opponents
2. Run this script to fine-tune the main agent against those opponents
3. Repeat until the main agent beats all checkpoints
"""
import os
import sys
import glob
import random
import logging
from typing import List, Tuple, Dict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

# Kaggle environments
from kaggle_environments import make, utils as kaggle_utils
from c4solver_wrapper import get_c4solver

# SB3 for PPO training
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces

# Import your existing components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_connectx_rl_robust import ConnectXNet, flat_to_2d, is_win_from, find_drop_row
from batch_dump_and_battle import load_agent_from_submission, evaluate_pair, dump_all_checkpoints, load_state_np, write_submission_from_np_state
# Import our ResNet extractor
from policies.resnet_features import ResNetExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration for online learning"""
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Training control
    timesteps_per_round: int = 50000
    eval_interval: int = 10000
    min_winrate_threshold: float = 0.8
    max_training_rounds: int = 10


class MultiOpponentEnv(gym.Env):
    """Gymnasium environment for training against multiple opponents"""
    
    def __init__(self, opponent_agents: List, opponent_weights: List[float] = None, seed: int = None):
        super().__init__()
        self.opponent_agents = opponent_agents
        self.opponent_weights = opponent_weights or [1.0] * len(opponent_agents)
        
        # Normalize weights
        total_weight = sum(self.opponent_weights)
        self.opponent_weights = [w / total_weight for w in self.opponent_weights]
        
        # ConnectX environment setup
        self.rows, self.cols = 6, 7
        self.observation_space = spaces.Box(low=0, high=2, shape=(42,), dtype=np.int32)
        self.action_space = spaces.Discrete(7)
        
        # Game state
        self.board = None
        self.current_player = None
        self.opponent_agent = None
        self.moves_count = 0
        
        # RNG
        self.np_random = np.random.RandomState(seed)
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
            
        # Reset board
        self.board = [0] * (self.rows * self.cols)
        self.moves_count = 0
        
        # Randomly choose starting player and opponent
        self.current_player = self.np_random.choice([1, 2])
        self.opponent_agent = self.np_random.choice(self.opponent_agents, p=self.opponent_weights)
        
        # If opponent starts, make opponent move
        if self.current_player == 2:
            self._make_opponent_move()
            
        obs = np.array(self.board, dtype=np.int32)
        return obs, {}
    
    def _get_valid_actions(self):
        """Get valid column indices where pieces can be dropped"""
        return [c for c in range(self.cols) if self.board[c] == 0]
    
    def _make_move(self, col: int, player: int) -> bool:
        """Make a move for the given player. Returns True if successful."""
        if col < 0 or col >= self.cols or self.board[col] != 0:
            return False
            
        # Find the lowest empty row in this column
        for row in range(self.rows - 1, -1, -1):
            idx = row * self.cols + col
            if self.board[idx] == 0:
                self.board[idx] = player
                self.moves_count += 1
                return True
        return False
    
    def _check_win(self, player: int) -> bool:
        """Check if the given player has won"""
        grid = flat_to_2d(self.board)
        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r][c] == player and is_win_from(grid, r, c, player):
                    return True
        return False
    
    def _is_draw(self) -> bool:
        """Check if the game is a draw (board full)"""
        return 0 not in self.board
    
    def _make_opponent_move(self):
        """Make a move for the opponent using their agent"""
        try:
            # Create observation for opponent (player 2 perspective)
            obs = {"board": self.board, "mark": 2}
            config = type('Config', (), {"rows": self.rows, "columns": self.cols, "inarow": 4})()
            
            action = self.opponent_agent(obs, config)
            
            # Validate and make move
            if action in self._get_valid_actions():
                self._make_move(action, 2)
            else:
                # Fallback to random valid move
                valid_actions = self._get_valid_actions()
                if valid_actions:
                    action = self.np_random.choice(valid_actions)
                    self._make_move(action, 2)
                    
        except Exception as e:
            logger.warning(f"Opponent move failed: {e}, using random fallback")
            valid_actions = self._get_valid_actions()
            if valid_actions:
                action = self.np_random.choice(valid_actions)
                self._make_move(action, 2)
    
    def step(self, action: int):
        """Execute one step in the environment"""
        # Validate action
        if action not in self._get_valid_actions():
            # Invalid move - penalize heavily
            obs = np.array(self.board, dtype=np.int32)
            return obs, -1.0, True, True, {"invalid_move": True}
        
        # Make player move
        self._make_move(action, 1)
        
        # Check if player won
        if self._check_win(1):
            obs = np.array(self.board, dtype=np.int32)
            return obs, 1.0, True, False, {"result": "win"}
        
        # Check for draw
        if self._is_draw():
            obs = np.array(self.board, dtype=np.int32)
            return obs, 0.0, True, False, {"result": "draw"}
        
        # Make opponent move
        self._make_opponent_move()
        
        # Check if opponent won
        if self._check_win(2):
            obs = np.array(self.board, dtype=np.int32)
            return obs, -1.0, True, False, {"result": "loss"}
        
        # Check for draw after opponent move
        if self._is_draw():
            obs = np.array(self.board, dtype=np.int32)
            return obs, 0.0, True, False, {"result": "draw"}
        
        # Continue game
        obs = np.array(self.board, dtype=np.int32)
        return obs, 0.0, False, False, {}


class BattleEvalCallback(BaseCallback):
    """Callback to evaluate against target opponents during training"""
    
    def __init__(self, main_agent_path: str, failed_opponents: List[Tuple[str, str]], 
                 eval_interval: int = 10000, target_winrate: float = 0.75, 
                 save_dir: str = "checkpoints", verbose: int = 1):
        super().__init__(verbose)
        self.main_agent_path = main_agent_path
        self.failed_opponents = failed_opponents  # List of (ckpt_path, submission_path)
        self.eval_interval = eval_interval
        self.target_winrate = target_winrate
        self.save_dir = save_dir
        
        self.best_avg_winrate = 0.0
        self.winrate_history = []
        self.eval_count = 0
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_interval != 0:
            return True
            
        logger.info(f"Evaluating at timestep {self.num_timesteps}...")

        # Build a live SB3-agent wrapper to avoid fragile temp submission conversion
        def _sb3_agent_from_model(model):
            def agent(obs, config):
                try:
                    board = obs['board'] if isinstance(obs, dict) and 'board' in obs else obs
                    # Build observation for policy
                    obs_arr = np.array(board, dtype=np.int32)
                    # Predict action
                    action, _ = model.predict(obs_arr, deterministic=True)
                    action = int(action)
                    # Ensure validity: fallback to random valid if invalid
                    valids = [c for c in range(7) if board[c] == 0]
                    if action not in valids:
                        action = int(valids[0]) if valids else 0
                    return action
                except Exception:
                    # Ultimate fallback
                    valids = [c for c in range(7) if obs['board'][c] == 0] if isinstance(obs, dict) else [c for c in range(7) if obs[c] == 0]
                    return int(valids[0]) if valids else 0
            return agent

        current_agent = _sb3_agent_from_model(self.model)
        
        # Evaluate against all failed opponents
        total_winrate = 0.0
        beaten_count = 0

        try:
            if not self.failed_opponents:
                logger.warning("No failed opponents to evaluate against.")
                avg_winrate = 0.0
            else:
                for ckpt_path, sub_path in self.failed_opponents:
                    try:
                        opp_agent = load_agent_from_submission(sub_path)
                    except Exception as e:
                        logger.warning(f"Failed to load opponent {sub_path}: {e}")
                        continue
                    try:
                        wins, draws, losses = evaluate_pair(current_agent, opp_agent, games=6)
                        wr = wins / max(1, wins + losses) if (wins + losses) > 0 else 0.0
                    except Exception as e:
                        logger.warning(f"Evaluation error vs {os.path.basename(ckpt_path)}: {e}")
                        wr = 0.0

                    total_winrate += wr
                    if wr >= self.target_winrate:
                        beaten_count += 1
                    logger.info(f"  vs {os.path.basename(ckpt_path)}: WR={wr:.3f}")

                # Avoid division by zero
                avg_winrate = total_winrate / len(self.failed_opponents) if len(self.failed_opponents) > 0 else 0.0

            self.winrate_history.append(avg_winrate)
            self.eval_count += 1
            logger.info(f"Average winrate: {avg_winrate:.3f}, Beaten: {beaten_count}/{len(self.failed_opponents)}")

            # Save if improved
            if avg_winrate > self.best_avg_winrate:
                self.best_avg_winrate = avg_winrate
                best_model_path = os.path.join(self.save_dir, f"battle_trained_wr_{avg_winrate:.3f}.pt")
                try:
                    torch.save(self.model.policy.state_dict(), best_model_path)
                    logger.info(f"New best model saved: {best_model_path}")
                except Exception as e:
                    logger.warning(f"Failed to save best model: {e}")

        except Exception as e:
            # Catch-all to prevent training crash due to evaluation issues
            logger.exception(f"Unhandled exception during evaluation: {e}")
            # Don't interrupt training; return True to continue
        finally:
            # No temp files to clean when using live agent
            pass

        return True


def load_failed_opponents_from_log(log_file: str = "battle_results.log") -> List[Tuple[str, float]]:
    """Parse failed opponents from battle log output"""
    failed = []
    if not os.path.exists(log_file):
        return failed
        
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    in_failed_section = False
    for line in lines:
        line = line.strip()
        if "Opponents not beaten yet" in line:
            in_failed_section = True
            continue
        if in_failed_section and line.startswith(" - "):
            # Parse " - best_model_wr_0.500.pt: WR=0.000"
            parts = line.split(": WR=")
            if len(parts) == 2:
                model_name = parts[0].replace(" - ", "")
                wr = float(parts[1])
                failed.append((f"checkpoints/{model_name}", wr))
    
    return failed


def train_against_opponents(main_ckpt: str, failed_opponents: List[Tuple[str, str]], 
                          config: TrainingConfig, save_dir: str = "checkpoints") -> str:
    """Train the main agent against failed opponents using PPO"""
    
    # Load opponent agents
    opponent_agents = []
    opponent_names = []
    for ckpt_path, sub_path in failed_opponents:
        try:
            agent = load_agent_from_submission(sub_path)
            opponent_agents.append(agent)
            opponent_names.append(os.path.basename(ckpt_path))
            logger.info(f"Loaded opponent: {os.path.basename(ckpt_path)}")
        except Exception as e:
            logger.warning(f"Failed to load opponent {ckpt_path}: {e}")

    # Built-in tactical/random opening agent
    def _tactical_random_opening_agent(obs, config):
        try:
            board = obs['board'] if isinstance(obs, dict) and 'board' in obs else obs
            mark = obs.get('mark', 2) if isinstance(obs, dict) else 2
            # valid actions: top cell empty
            valid = [c for c in range(7) if board[c] == 0]
            if not valid:
                return 0

            grid = flat_to_2d(board)
            # winning move
            for c in valid:
                r = find_drop_row(grid, c)
                if r is None:
                    continue
                grid2 = [row[:] for row in grid]
                grid2[r][c] = mark
                if is_win_from(grid2, r, c, mark):
                    return int(c)

            # block opponent
            opp = 3 - mark
            for c in valid:
                r = find_drop_row(grid, c)
                if r is None:
                    continue
                grid2 = [row[:] for row in grid]
                grid2[r][c] = opp
                if is_win_from(grid2, r, c, opp):
                    return int(c)

            # opening preference (center-first) with slight randomness
            center_order = [3, 4, 2, 5, 1, 6, 0]
            choices = [c for c in center_order if c in valid]
            if choices:
                if random.random() < 0.8:
                    return int(choices[0])
                return int(random.choice(choices))

            return int(random.choice(valid))
        except Exception as e:
            logger.warning(f"_tactical_random_opening_agent error: {e}")
            return int(random.choice(range(7)))

    # C4Solver-backed agent
    def _c4solver_agent(obs, config):
        try:
            board = obs['board'] if isinstance(obs, dict) and 'board' in obs else obs
            valid = [c for c in range(7) if board[c] == 0]
            solver = get_c4solver()
            if solver is None:
                return _tactical_random_opening_agent(obs, config)
            action, _ = solver.get_best_move(board, valid)
            return int(action)
        except Exception as e:
            logger.warning(f"c4solver_agent error: {e}")
            return _tactical_random_opening_agent(obs, config)

    # Add builtin agents to the pool (they act like additional opponents)
    try:
        opponent_agents.append(_tactical_random_opening_agent)
        opponent_names.append("_tactical_random_opening_agent")
        opponent_agents.append(_c4solver_agent)
        opponent_names.append("_c4solver_agent")
        logger.info("Added builtin agents: _tactical_random_opening_agent, _c4solver_agent")
    except Exception:
        logger.warning("Failed to add builtin agents; continuing with loaded opponents only")
    
    if not opponent_agents:
        logger.error("No opponents loaded for training")
        return main_ckpt
    
    # Create training environment
    def make_env():
        return MultiOpponentEnv(opponent_agents, seed=random.randint(0, 1000000))
    
    env = DummyVecEnv([make_env])
    
    # Load main agent policy for initialization
    device = torch.device("cpu")
    
    # Initialize PPO with loaded weights
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        # Use a feature extractor (ResNet) and small MLP heads; depth configurable via ResNetExtractor
        policy_kwargs=dict(
            features_extractor_class=ResNetExtractor,
            features_extractor_kwargs={"channels": 64, "num_blocks": 12, "feat_dim": 1024},
            net_arch=dict(pi=[512], vf=[512]),
            activation_fn=torch.nn.ReLU
        ),
        device=device,
        verbose=1
    )
    
    # Load pretrained weights
    try:
        checkpoint = torch.load(main_ckpt, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        # Try to load compatible weights
        model.policy.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded pretrained weights from {main_ckpt}")
    except Exception as e:
        logger.warning(f"Could not load pretrained weights: {e}, training from scratch")
    
    # Setup evaluation callback
    eval_callback = BattleEvalCallback(
        main_ckpt, failed_opponents, 
        eval_interval=config.eval_interval,
        target_winrate=config.min_winrate_threshold,
        save_dir=save_dir
    )
    
    # Train
    logger.info(f"Starting training against {len(opponent_agents)} opponents...")
    logger.info(f"Opponents: {', '.join(opponent_names)}")
    
    model.learn(
        total_timesteps=config.timesteps_per_round,
        callback=eval_callback,
        progress_bar=False  # Disable progress bar to avoid import issues
    )
    
    # Save final model
    final_path = os.path.join(save_dir, f"battle_final_{config.timesteps_per_round}.pt")
    torch.save(model.policy.state_dict(), final_path)
    
    # Return best model path
    best_path = os.path.join(save_dir, f"battle_trained_wr_{eval_callback.best_avg_winrate:.3f}.pt")
    return best_path if os.path.exists(best_path) else final_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Online battle trainer")
    parser.add_argument("--main_ckpt", type=str, default="checkpoints/best_model_wr_0.600.pt")
    parser.add_argument("--failed_log", type=str, help="Log file with failed opponents (optional)")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--sub_dir", type=str, default="sub")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--max_rounds", type=int, default=10)
    parser.add_argument("--winrate_threshold", type=float, default=0.8)
    args = parser.parse_args()
    
    config = TrainingConfig(
        timesteps_per_round=args.timesteps,
        max_training_rounds=args.max_rounds,
        min_winrate_threshold=args.winrate_threshold
    )
    
    # Find failed opponents
    if args.failed_log and os.path.exists(args.failed_log):
        failed_list = load_failed_opponents_from_log(args.failed_log)
        failed_opponents = [(ckpt, os.path.join(args.sub_dir, f"{os.path.splitext(os.path.basename(ckpt))[0]}.py")) 
                           for ckpt, _ in failed_list]
    else:
        # Run battle evaluation first to identify failed opponents
        logger.info("Running initial battle evaluation...")
        os.system(f"python batch_dump_and_battle.py --main_ckpt {os.path.basename(args.main_ckpt)} --winrate {args.winrate_threshold}")
        
        # Manual fallback - train against some known challenging opponents
        potential_opponents = [os.path.basename(f) for f in glob.glob(os.path.join(args.ckpt_dir, "*.pt"))]
        failed_opponents = []
        for opp in potential_opponents:
            ckpt_path = os.path.join(args.ckpt_dir, opp)
            sub_path = os.path.join(args.sub_dir, f"{os.path.splitext(opp)[0]}.py")
            if os.path.exists(ckpt_path) and os.path.exists(sub_path):
                failed_opponents.append((ckpt_path, sub_path))

        # Also include submission_vMega.py if present (visualizer expects it)
        vmega_paths = [
            os.path.join(os.getcwd(), 'submission_vMega.py'),
            os.path.join(os.getcwd(), 'sub', 'submission_vMega.py'),
        ]
        for vp in vmega_paths:
            if os.path.exists(vp):
                logger.info(f"Including special opponent: {vp}")
                # Use a synthetic checkpoint path name for logging consistency
                failed_opponents.append((os.path.basename(vp), vp))
    
    if not failed_opponents:
        logger.info("No failed opponents found - main agent may already be strong enough!")
        return
    
    logger.info(f"Training against {len(failed_opponents)} failed opponents...")
    
    current_main = args.main_ckpt
    for round_num in range(config.max_training_rounds):
        logger.info(f"\n=== Training Round {round_num + 1}/{config.max_training_rounds} ===")
        
        # Train one round
        new_main = train_against_opponents(current_main, failed_opponents, config, args.save_dir)
        
        # Re-evaluate
        logger.info("Re-evaluating after training...")
        # Here you could re-run the batch evaluation or implement inline evaluation
        
        current_main = new_main
        logger.info(f"Round {round_num + 1} complete. Current best: {current_main}")
    
    logger.info(f"\nTraining complete! Final model: {current_main}")
    logger.info("Run batch_dump_and_battle.py again to verify improvements.")


if __name__ == "__main__":
    main()
