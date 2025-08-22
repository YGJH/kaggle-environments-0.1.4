#!/usr/bin/env python3
"""
Connect Four RL training with Stable-Baselines3 and Optuna hyperparameter tuning

Features:
- Uses Kaggle ConnectX environment (6x7, in-a-row=4)
- Curriculum opponents: random -> weak tactical -> Kaggle negamax -> native C++ solver
- Optional integration with C++ minimax (connect4/c4solver) via c4solver_wrapper.C4SolverWrapper
- PPO baseline with tuned hyperparameters via Optuna
- Evaluation function with clear comments and logging
- Reward plot over time (moving average)
- Saves best model by evaluation win-rate as .pt (Torch state_dict)

Notes:
- Requires packages in requirements.txt including stable-baselines3 and optuna
- Will automatically seed for reproducibility
"""

import os
import sys
import time
import json
import math
import random
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import multiprocessing as mp

import numpy as np
import torch
"""Select a headless backend to avoid Qt/xcb issues on servers."""
import matplotlib
# Ensure no GUI backend is used even if DISPLAY is absent or Qt is present via cv2
matplotlib.use("Agg")
# Also hint Qt (if indirectly loaded) to use offscreen platform
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib.pyplot as plt

from kaggle_environments import make, evaluate

# Ensure project root is on sys.path (so we can import c4solver_wrapper when running from videos/)
try:
	_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
	_REPO_ROOT = os.path.dirname(_THIS_DIR)
	if _REPO_ROOT not in sys.path:
		sys.path.insert(0, _REPO_ROOT)
except Exception:
	pass

# SB3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Optuna
import optuna

# Local wrappers/agents
from kaggle_environments.envs.connectx.connectx import agents as kx_agents
from c4solver_wrapper import get_c4solver

logger = logging.getLogger("train_connectx")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ============================
# Environment helper functions
# ============================

EMPTY = 0


def valid_moves(board: List[int], columns: int = 7) -> List[int]:
	grid_top = board[:columns]
	return [c for c in range(columns) if grid_top[c] == EMPTY]


def play(board: List[int], column: int, mark: int, columns: int = 7, rows: int = 6) -> List[int]:
	newb = board[:]
	for r in range(rows - 1, -1, -1):
		idx = column + r * columns
		if newb[idx] == EMPTY:
			newb[idx] = mark
			return newb
	return board[:]  # full column fallback

# ----------------------------
# Pattern detectors for shaping
# ----------------------------

def flat_to_2d(board_flat: List[int], rows: int = 6, columns: int = 7) -> List[List[int]]:
	return [list(board_flat[r*columns:(r+1)*columns]) for r in range(rows)]


def count_open_ended_twos(board: List[int], mark: int, rows: int = 6, columns: int = 7) -> int:
	"""Count patterns of the form [0, mark, mark, 0] in any 4-length window.

	This focuses the agent on connected twos with both ends free (open-ended two),
	which are strong building blocks or immediate threats. We scan horizontal,
	vertical, and both diagonals.
	"""
	g = flat_to_2d(board, rows, columns)
	total = 0
	# Horizontal
	for r in range(rows):
		for c in range(columns - 3):
			window = [g[r][c+i] for i in range(4)]
			if window == [0, mark, mark, 0]:
				total += 1
	# Vertical
	for r in range(rows - 3):
		for c in range(columns):
			window = [g[r+i][c] for i in range(4)]
			if window == [0, mark, mark, 0]:
				total += 1
	# Diagonal down-right
	for r in range(rows - 3):
		for c in range(columns - 3):
			window = [g[r+i][c+i] for i in range(4)]
			if window == [0, mark, mark, 0]:
				total += 1
	# Diagonal up-right
	for r in range(3, rows):
		for c in range(columns - 3):
			window = [g[r-i][c+i] for i in range(4)]
			if window == [0, mark, mark, 0]:
				total += 1
	return total


# ============================
# Opponents
# ============================

def random_agent_fn(obs, config):
	return random.choice(valid_moves(obs.board, config.columns))


def negamax_agent_fn(obs, config):
	# Use Kaggle-provided simple negamax (limited depth)
	return kx_agents["negamax"](obs, config)


class SolverAgent:
	"""C++ solver-backed opponent. Falls back to center preference if solver unavailable."""

	def __init__(self, weak: bool = True):
		self.solver = get_c4solver()
		self.weak = weak

	def __call__(self, obs, config):
		columns = config.columns
		acts = valid_moves(obs.board, columns)
		if not acts:
			return 0
		if self.solver is None:
			# Center preference fallback
			center_pref = [3, 4, 2, 5, 1, 6, 0]
			for c in center_pref:
				if c in acts:
					return c
			return acts[0]
		# Ask solver for analysis scores and choose best valid move
		res = self.solver.evaluate_board(obs.board, analyze=True)
		if not res.get("valid", False):
			return random.choice(acts)
		scores = res.get("scores", [0] * columns)
		best = max(acts, key=lambda a: scores[a])
		return best


# ============================
# Gym-like wrapper around Kaggle env for SB3
# ============================

import gymnasium as gym
from gymnasium import spaces


class ConnectXSB3Env(gym.Env):
	metadata = {"render.modes": ["human"]}

	def __init__(self, opponent: str = "random", seed: int = 42):
		super().__init__()
		# Store an initial seed; Gymnasium will pass seed to reset()
		self._seed = seed
		self.rng = np.random.default_rng(seed)
		self.env = make("connectx", debug=False)
		self.config = self.env.configuration
		self.columns = self.config.columns
		self.rows = self.config.rows
		self.action_space = spaces.Discrete(self.columns)
		self.observation_space = spaces.Box(low=0, high=2, shape=(self.rows * self.columns,), dtype=np.int8)
		self._opponent_name = opponent
		self._opponent_callable = self._resolve_opponent(opponent)
		self.my_mark: Optional[int] = None
		self._reset_kaggle()

	def _resolve_opponent(self, name: str):
		if name == "random":
			return random_agent_fn
		if name == "negamax":
			return negamax_agent_fn
		if name == "solver":
			return SolverAgent(weak=True)
		return random_agent_fn

	def _reset_kaggle(self):
		# player order randomized: our agent index is returned, obs is environment state wrapper
		self.env.reset()
		self.trainer = self.env.train([None, self._opponent_callable])
		self.board = [0] * (self.rows * self.columns)
		self.done = False
		self.info = {}

	def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
		# Gymnasium-style reset: configure RNGs and return (obs, info)
		if seed is not None:
			self._seed = seed
		# Seed python/numpy/torch for determinism of opponent sampling, etc.
		try:
			random.seed(self._seed)
			np.random.seed(self._seed % (2**32 - 1))
			torch.manual_seed(self._seed)
		except Exception:
			pass

		self._reset_kaggle()
		obs = self.trainer.reset()
		# obs is dict with board, mark
		self.board = obs["board"][:]
		self.my_mark = int(obs.get("mark", 1))
		return np.array(self.board, dtype=np.int8), {}

	def step(self, action: int):
		# Gymnasium-style step: return (obs, reward, terminated, truncated, info)
		# The Kaggle env handles the opponent's turn internally.
		# Compute potential before move for shaping ([0, m, m, 0] patterns)
		prev_board = self.board[:]
		my_mark = int(self.my_mark) if self.my_mark in (1, 2) else 1
		opp_mark = 3 - my_mark
		prev_open2 = count_open_ended_twos(prev_board, my_mark, self.rows, self.columns) - \
					count_open_ended_twos(prev_board, opp_mark, self.rows, self.columns)

		obs, reward, done, info = self.trainer.step(int(action))
		# Kaggle env yields reward=None on non-terminal steps; SB3 expects a float each step.
		if reward is None:
			reward = 0.0
		self.board = obs["board"][:]
		# Compute potential after move and add shaped reward proportional to delta
		curr_open2 = count_open_ended_twos(self.board, my_mark, self.rows, self.columns) - \
					count_open_ended_twos(self.board, opp_mark, self.rows, self.columns)
		shaping = 0.05 * (curr_open2 - prev_open2)
		reward = float(reward) + shaping
		# Update my_mark from the new observation for next step (Kaggle keeps our mark constant between turns)
		try:
			self.my_mark = int(obs.get("mark", my_mark))
		except Exception:
			pass
		self.done = bool(done)
		self.info = info
		terminated = self.done
		truncated = False
		return np.array(self.board, dtype=np.int8), float(reward), terminated, truncated, info

	def render(self, mode="human"):
		# Optional: simple text render
		grid = np.array(self.board).reshape(self.rows, self.columns)
		print(grid)


# ============================
# Callback for tracking rewards and saving best by win-rate
# ============================


@dataclass
class EvalConfig:
	n_episodes: int = 30
	opponent: str = "random"  # use curriculum outside callback


class WinRateEvalCallback(BaseCallback):
	"""Evaluate periodically and save best model as .pt by win-rate.

	- Runs evaluation games against a mix of opponents for stability
	- Tracks moving average reward for plotting
	- Saves torch state_dict to disk when a new best win-rate is achieved
	"""

	def __init__(self, eval_interval: int, eval_cfg: EvalConfig, save_dir: str = "checkpoints"):
		super().__init__()
		self.eval_interval = eval_interval
		self.eval_cfg = eval_cfg
		self.save_dir = save_dir
		os.makedirs(save_dir, exist_ok=True)
		self.best_win_rate = -1.0
		self.reward_history: List[float] = []
		self.timesteps: List[int] = []

	def _on_step(self) -> bool:
		if self.n_calls % self.eval_interval != 0:
			return True

		# Evaluate against a curriculum set for robust estimate
		opponents = ["random", "negamax", "solver"]
		weights = [0.5, 0.3, 0.2]
		total_wins, total_games = 0, 0
		rewards = []
		for opp, w in zip(opponents, weights):
			env = DummyVecEnv([lambda: ConnectXSB3Env(opponent=opp, seed=123)])
			mean_reward, _ = evaluate_policy(self.model, env, n_eval_episodes=max(5, int(self.eval_cfg.n_episodes * w)), deterministic=True)
			rewards.append(mean_reward)
			# Approximate win rate: mean_reward since reward in {-1,0,1}
			# But mean_reward can be in [-1,1]; map to wins assuming no ties as rough proxy
			win_rate = (mean_reward + 1) / 2.0
			total_wins += win_rate * max(5, int(self.eval_cfg.n_episodes * w))
			total_games += max(5, int(self.eval_cfg.n_episodes * w))

		avg_reward = float(np.mean(rewards)) if rewards else 0.0
		overall_win_rate = total_wins / max(1, total_games)
		self.reward_history.append(avg_reward)
		self.timesteps.append(self.num_timesteps)

		logger.info(f"Eval@{self.num_timesteps}: avg_reward={avg_reward:.3f}, win_rate={overall_win_rate:.3f}")

		if overall_win_rate > self.best_win_rate:
			self.best_win_rate = overall_win_rate
			# Save underlying torch state_dict as .pt
			path = os.path.join(self.save_dir, f"best_agent_{overall_win_rate:.3f}.pt")
			torch.save(self.model.policy.state_dict(), path)
			logger.info(f"Saved new best model to {path}")

		return True

	def plot_rewards(self, out_path: str = "reward_plot.png", window: int = 10):
		if not self.reward_history:
			return
		y = np.array(self.reward_history)
		# Moving average smoothing
		if len(y) >= window:
			kernel = np.ones(window) / window
			y_smooth = np.convolve(y, kernel, mode="valid")
			x_smooth = self.timesteps[window - 1 :]
		else:
			y_smooth = y
			x_smooth = self.timesteps
		plt.figure(figsize=(7, 4))
		plt.plot(x_smooth, y_smooth, label="avg eval reward (MA)")
		plt.xlabel("timesteps")
		plt.ylabel("reward (-1..1)")
		plt.title("Evaluation Reward Over Time")
		plt.grid(True, alpha=0.3)
		plt.legend()
		plt.tight_layout()
		plt.savefig(out_path)
		logger.info(f"Saved reward plot to {out_path}")


# ============================
# Optuna study for PPO hyperparams
# ============================


def make_env_for_training(opponent: str) -> DummyVecEnv:
	return DummyVecEnv([lambda: ConnectXSB3Env(opponent=opponent, seed=random.randint(0, 1_000_000))])


def ppo_trial(trial: optuna.Trial) -> float:
	# Curriculum stage for this trial: start weak to stabilize learning
	opponent = trial.suggest_categorical("opponent", ["random", "negamax"])

	env = make_env_for_training(opponent)

	policy_kwargs = dict(
		net_arch=dict(
			pi=[trial.suggest_int("pi1", 64, 256, step=32), trial.suggest_int("pi2", 32, 128, step=32)],
			vf=[trial.suggest_int("vf1", 64, 256, step=32), trial.suggest_int("vf2", 32, 128, step=32)],
		)
	)
	# Choose batch size to evenly divide (n_steps * n_envs) without using a dynamic categorical space.
	# We sample a static number of minibatches and map it to the nearest divisor of total.
	n_envs = getattr(env, "num_envs", 1)
	n_steps = trial.suggest_int("n_steps", 256, 2048, step=256)
	total = int(n_steps) * int(n_envs)
	# Static candidate counts for number of minibatches
	nb_candidates = [1, 2, 4, 6, 8, 12, 16, 24, 32]
	num_minibatches = trial.suggest_categorical("num_minibatches", nb_candidates)
	# Find the largest divisor of total that is <= num_minibatches; fallback to 1
	def largest_divisor_leq(x: int, limit: int) -> int:
		best = 1
		i = 1
		while i * i <= x:
			if x % i == 0:
				d1, d2 = i, x // i
				if d1 <= limit and d1 > best:
					best = d1
				if d2 <= limit and d2 > best:
					best = d2
			i += 1
		return best
	k = largest_divisor_leq(total, int(num_minibatches))
	batch_size = max(1, total // k)
	model = PPO(
		"MlpPolicy",
		env,
		learning_rate=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
		n_steps=n_steps,
	batch_size=batch_size,
		gamma=trial.suggest_float("gamma", 0.95, 0.999),
		gae_lambda=trial.suggest_float("gae_lambda", 0.9, 0.98),
		clip_range=trial.suggest_float("clip", 0.1, 0.3),
		ent_coef=trial.suggest_float("ent_coef", 0.0, 0.02),
		vf_coef=trial.suggest_float("vf_coef", 0.3, 1.0),
		n_epochs=trial.suggest_int("n_epochs", 4, 15),
		policy_kwargs=policy_kwargs,
		verbose=0,
        device='cpu',
		seed=42,
	)

	# Compact training for trial
	total_timesteps = 50_000
	model.learn(total_timesteps=total_timesteps)

	# Evaluate vs random to get a robust quick signal
	eval_env = DummyVecEnv([lambda: ConnectXSB3Env(opponent="random", seed=999)])
	mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=30, deterministic=True)
	# Return higher is better; Optuna will maximize by default if we set study direction
	return float(mean_reward)


def tune_hyperparams(n_trials: int = 20, storage: Optional[str] = None, n_jobs: int = 1) -> Dict:
	# Reduce per-trial thread usage when parallelizing trials to avoid oversubscription
	if n_jobs and n_jobs > 1:
		try:
			os.environ.setdefault("OMP_NUM_THREADS", "1")
			torch.set_num_threads(1)
		except Exception:
			pass

	study = optuna.create_study(direction="maximize", storage=storage, study_name=None)
	# n_jobs > 1 will run trials in parallel threads
	study.optimize(ppo_trial, n_trials=n_trials, n_jobs=max(1, int(n_jobs)))
	logger.info(f"Best trial value={study.best_value:.3f}\nBest params={json.dumps(study.best_params, indent=2)}")
	return study.best_params


# ============================
# Main training loop with curriculum and evaluation
# ============================


def train_and_eval(total_timesteps: int = 300_000, n_optuna_trials: int = 15, save_dir: str = "checkpoints", optuna_workers: int = 1):
	os.makedirs(save_dir, exist_ok=True)

	# 1) Hyperparameter search
	logger.info("Starting Optuna tuning...")
	storage = f"sqlite:///{os.path.join(save_dir, 'optuna.db')}"
	best_params = tune_hyperparams(n_trials=n_optuna_trials, storage=storage, n_jobs=optuna_workers)

	# 2) Build model with best params
	opponent_curriculum = ["random", "negamax", "solver"]
	eval_cb = WinRateEvalCallback(eval_interval=10_000, eval_cfg=EvalConfig(n_episodes=30), save_dir=save_dir)

	# translate best_params into model init
	policy_kwargs = dict(
		net_arch=dict(
			pi=[best_params.get("pi1", 128), best_params.get("pi2", 64)],
			vf=[best_params.get("vf1", 128), best_params.get("vf2", 64)],
		)
	)

	# 3) Curriculum training
	steps_per_stage = total_timesteps // len(opponent_curriculum)
	model: Optional[PPO] = None
	for stage, opp in enumerate(opponent_curriculum, start=1):
		logger.info(f"Curriculum stage {stage}/{len(opponent_curriculum)} vs {opp}")
		env = make_env_for_training(opp)

		if model is None:
			model = PPO(
				"MlpPolicy",
				env,
				learning_rate=best_params.get("lr", 1e-3),
				n_steps=best_params.get("n_steps", 1024),
				batch_size=best_params.get("batch_size", 256),
				gamma=best_params.get("gamma", 0.99),
				gae_lambda=best_params.get("gae_lambda", 0.95),
				clip_range=best_params.get("clip", 0.2),
				ent_coef=best_params.get("ent_coef", 0.01),
				vf_coef=best_params.get("vf_coef", 0.5),
				n_epochs=best_params.get("n_epochs", 10),
				policy_kwargs=policy_kwargs,
				verbose=1,
                device='cpu',
				seed=42,
			)
		else:
			# Continue training with same model but new environment
			model.set_env(env)

		model.learn(total_timesteps=steps_per_stage, callback=eval_cb)

	# 4) Final evaluation and artifacts
	eval_cb.plot_rewards(os.path.join(save_dir, "reward_plot.png"), window=10)

	# Also save final model .pt for completeness
	final_path = os.path.join(save_dir, "final_agent.pt")
	torch.save(model.policy.state_dict(), final_path)
	logger.info(f"Saved final model to {final_path}")


# ============================
# Entrypoint
# ============================


if __name__ == "__main__":

	timesteps=300_000
	trials=10
	save_dir = "checkpoints"
	# Use up to half of the CPUs for Optuna parallel trials (at least 1)
	try:
		cpu_cnt = mp.cpu_count()
	except Exception:
		cpu_cnt = 2
	optuna_workers = max(1, cpu_cnt)


	print(f'Using {optuna_workers} Optuna workers.')

	# Quick check for solver binary presence, not required but recommended
	solver = get_c4solver()
	if solver is None:
		logger.warning("C++ solver not found; training will still run but strongest opponent won't be used.")
	else:
		logger.info("C++ solver detected and will be used during curriculum.")

	train_and_eval(total_timesteps=timesteps, n_optuna_trials=trials, save_dir=save_dir, optuna_workers=optuna_workers)
