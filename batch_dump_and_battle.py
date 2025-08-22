#!/usr/bin/env python3
"""
Batch dump all .pt checkpoints to submission.py files and run battles between them.

Usage:
1. Converts all checkpoints/*.pt files to submission_XXX.py using dump_weight_fixed_v2.py
2. Runs battles between main agent and all other agents
3. Reports win rates and identifies opponents that need more training
"""
import os
import sys
import glob
import subprocess
import time
from typing import List, Tuple, Dict

import numpy as np
import torch
from kaggle_environments import make, utils as kaggle_utils
from pathlib import Path

# Import existing utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_weight_fixed_v2 import generate_submission_code

# Prefer the battle-specific dumper when available (handles raw state_dict format saved by online trainer)
try:
    import dump_battle_checkpoints as battle_dumper
except Exception:
    battle_dumper = None

CHECKPOINT_DIR = "checkpoints"
SUB_DIR = "sub"
MAIN_AGENT_PT = "best_model_wr_0.600.pt"
DUMPER_SCRIPT = "dump_weight_fixed_v2.py"


def dump_all_checkpoints(ckpt_dir: str = CHECKPOINT_DIR, sub_dir: str = SUB_DIR) -> List[Tuple[str, str]]:
    """Convert all .pt files to submission.py files using dump_weight_fixed_v2.py"""
    os.makedirs(sub_dir, exist_ok=True)
    
    pt_files = glob.glob(os.path.join(ckpt_dir, "*.pt"))
    converted_pairs = []
    
    print(f"🔄 Converting {len(pt_files)} .pt files to submission scripts...")
    
    for pt_path in pt_files:
        base_name = os.path.splitext(os.path.basename(pt_path))[0]
        sub_path = os.path.join(sub_dir, f"{base_name}.py")
        
        try:
            # If this checkpoint name starts with 'battle' prefer the specialized dumper which
            # understands the raw state_dict format produced by `online_battle_trainer.py`.
            if base_name.startswith("battle") and battle_dumper is not None:
                try:
                    produced = battle_dumper.dump_checkpoint_to_submission(Path(pt_path), dry_run=False)
                    # produced is a Path to the generated submission
                    if produced is not None and os.path.exists(str(produced)):
                        converted_pairs.append((pt_path, str(produced)))
                        print(f"  ✅ {os.path.basename(pt_path)} -> {os.path.basename(produced)} (via dump_battle_checkpoints)")
                        continue
                except Exception as e:
                    print(f"  ⚠️  battle dumper failed for {base_name}, falling back: {e}")

            # Load checkpoint and generate submission
            checkpoint = torch.load(pt_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            # Convert tensors to numpy and normalize keys (strip common prefixes)
            def _normalize_key(k: str) -> str:
                # remove known prefixes frequently produced by SB3/torch wrappers
                prefixes = [
                    'policy.features_extractor.', 'policy.', 'model.', 'module.',
                    'net.', 'network.', 'actor.', 'critic.', 'features_extractor.'
                ]
                for p in prefixes:
                    if k.startswith(p):
                        return k[len(p):]
                return k

            np_weights = {}
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor):
                    nk = _normalize_key(k)
                    # if duplicate normalized key, prefer the first occurrence
                    if nk in np_weights:
                        # keep existing
                        continue
                    np_weights[nk] = v.detach().cpu().numpy()
                    
            # Encode weights to base64
            import base64
            import io
            buf = io.BytesIO()
            np.savez_compressed(buf, **np_weights)
            weights_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
            
            # Detect if this is a trunk architecture (look for SE blocks or attention)
            has_trunk = any('trunk' in k or 'se' in k.lower() or 'attention' in k.lower() 
                          for k in np_weights.keys())
                    
            # Generate submission code
            submission_code = generate_submission_code(weights_b64, has_trunk)
            
            # Write to file
            with open(sub_path, 'w') as f:
                f.write(submission_code)
            # Validate produced submission defines a callable
            try:
                code = kaggle_utils.read_file(sub_path)
                kaggle_utils.get_last_callable(code)
            except Exception as e:
                # Clean up bad submission for debugging
                if os.path.exists(sub_path):
                    os.remove(sub_path)
                raise RuntimeError(f"Generated submission {sub_path} is invalid: {e}")

            converted_pairs.append((pt_path, sub_path))
            print(f"  ✅ {os.path.basename(pt_path)} -> {os.path.basename(sub_path)}")
            
        except Exception as e:
            print(f"  ❌ Failed to convert {os.path.basename(pt_path)}: {e}")
            
    print(f"✅ Converted {len(converted_pairs)} checkpoints successfully")
    return converted_pairs


def load_agent_from_submission(sub_path: str):
    """Load agent function from submission.py file using Kaggle utils"""
    try:
        code = kaggle_utils.read_file(sub_path)
        return kaggle_utils.get_last_callable(code)
    except Exception as e:
        # Raise so caller can see the real failure and we can debug generation
        raise RuntimeError(f"Failed to load agent from {sub_path}: {e}")


def load_state_np(pt_path: str) -> Tuple[Dict, str]:
    """Load PyTorch state dict and convert to numpy"""
    checkpoint = torch.load(pt_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    np_weights = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            np_weights[k] = v.detach().cpu().numpy()
            
    return np_weights, "torch_state"


def write_submission_from_np_state(np_state: Dict, output_path: str):
    """Write submission.py from numpy state dict"""
    import base64
    import io
    
    # Encode weights to base64
    buf = io.BytesIO()
    np.savez_compressed(buf, **np_state)
    weights_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    
    # Normalize keys similar to dump_all_checkpoints so generated submission matches generate_submission_code expectations
    def _normalize_key(k: str) -> str:
        prefixes = [
            'policy.features_extractor.', 'policy.', 'model.', 'module.',
            'net.', 'network.', 'actor.', 'critic.', 'features_extractor.'
        ]
        for p in prefixes:
            if k.startswith(p):
                return k[len(p):]
        return k

    normalized = {}
    for k, v in np_state.items():
        nk = _normalize_key(k)
        if nk in normalized:
            continue
        normalized[nk] = v

    # Detect if this is a trunk architecture
    has_trunk = any('trunk' in k or 'se' in k.lower() or 'attention' in k.lower()
                  for k in normalized.keys())
    
    submission_code = generate_submission_code(weights_b64, has_trunk)
    with open(output_path, 'w') as f:
        f.write(submission_code)

    # Validate
    try:
        code = kaggle_utils.read_file(output_path)
        kaggle_utils.get_last_callable(code)
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"Generated submission {output_path} is invalid: {e}")


def evaluate_pair(agent1, agent2, games: int = 10) -> Tuple[int, int, int]:
    """Evaluate two agents against each other. Returns (wins, draws, losses) for agent1"""
    env = make("connectx", debug=False)
    
    wins, draws, losses = 0, 0, 0
    
    for game in range(games):
        # Alternate who goes first
        if game % 2 == 0:
            agents = [agent1, agent2]
        else:
            agents = [agent2, agent1]
            
        try:
            env.reset()
            env.run(agents)
            result = env.state  # Get the final state after running
            
            # Determine outcome from agent1's perspective
            if game % 2 == 0:  # agent1 was player 1
                if result[0].status == "DONE" and result[1].status == "DONE":
                    if result[0].reward == 1:
                        wins += 1
                    elif result[1].reward == 1:
                        losses += 1
                    else:
                        draws += 1
                else:
                    # Error occurred
                    draws += 1
            else:  # agent1 was player 2
                if result[0].status == "DONE" and result[1].status == "DONE":
                    if result[1].reward == 1:
                        wins += 1
                    elif result[0].reward == 1:
                        losses += 1
                    else:
                        draws += 1
                else:
                    draws += 1
                    
        except Exception as e:
            print(f"    Game {game+1} error: {e}")
            draws += 1
            
    return wins, draws, losses


def _safe_close_env(env):
    try:
        env.close()
    except Exception:
        pass
    try:
        del env
    except Exception:
        pass
    import gc
    gc.collect()


def run_battle_evaluation(main_ckpt: str, ckpt_dir: str = CHECKPOINT_DIR, 
                         sub_dir: str = SUB_DIR, winrate_threshold: float = 0.8,
                         games_per_opponent: int = 10) -> Dict:
    """Run comprehensive battle evaluation"""
    
    # Step 1: Convert all checkpoints
    print("="*60)
    print("🚀 BATCH DUMP AND BATTLE EVALUATION")
    print("="*60)
    
    converted_pairs = dump_all_checkpoints(ckpt_dir, sub_dir)
    
    if not converted_pairs:
        print("❌ No checkpoints converted successfully")
        return {}
        
    # Step 2: Load main agent
    main_sub_path = None
    for pt_path, sub_path in converted_pairs:
        if os.path.basename(pt_path) == main_ckpt:
            main_sub_path = sub_path
            break
            
    if not main_sub_path:
        print(f"❌ Main agent {main_ckpt} not found in converted submissions")
        return {}
        
    try:
        main_agent = load_agent_from_submission(main_sub_path)
        print(f"✅ Loaded main agent: {main_ckpt}")
    except Exception as e:
        print(f"❌ Failed to load main agent: {e}")
        return {}
        
    # Step 3: Battle against all opponents
    print(f"\n🥊 BATTLE EVALUATION (target WR: {winrate_threshold:.1%})")
    print("-"*60)
    
    results = {}
    beaten_opponents = []
    failed_opponents = []
    
    for pt_path, sub_path in converted_pairs:
        opponent_name = os.path.basename(pt_path)
        
        # Skip self-play
        if opponent_name == main_ckpt:
            continue
            
        try:
            opponent_agent = load_agent_from_submission(sub_path)
            
            print(f"⚔️  vs {opponent_name}...", end=" ", flush=True)
            
            wins, draws, losses = evaluate_pair(main_agent, opponent_agent, games_per_opponent)
            total_decisive = wins + losses
            winrate = wins / total_decisive if total_decisive > 0 else 0.0
            
            results[opponent_name] = {
                'wins': wins, 'draws': draws, 'losses': losses,
                'winrate': winrate, 'total_games': games_per_opponent
            }
            
            status = "✅ BEATEN" if winrate >= winrate_threshold else "❌ FAILED"
            print(f"{status} (WR: {winrate:.3f}, {wins}W-{draws}D-{losses}L)")
            
            if winrate >= winrate_threshold:
                beaten_opponents.append(opponent_name)
            else:
                failed_opponents.append((opponent_name, winrate))
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results[opponent_name] = {'error': str(e)}
            
    # Step 4: Summary
    total_opponents = len([p for p in converted_pairs if os.path.basename(p[0]) != main_ckpt])
    beaten_count = len(beaten_opponents)
    
    print("\n" + "="*60)
    print("📊 BATTLE SUMMARY")
    print("="*60)
    print(f"Main agent: {main_ckpt}")
    print(f"Total opponents: {total_opponents}")
    print(f"Beaten (WR ≥ {winrate_threshold:.1%}): {beaten_count}")
    print(f"Failed: {len(failed_opponents)}")
    print(f"Success rate: {beaten_count/total_opponents:.1%}" if total_opponents > 0 else "No opponents")
    
    if failed_opponents:
        print(f"\nOpponents not beaten yet (by WR threshold):")
        failed_opponents.sort(key=lambda x: x[1], reverse=True)  # Sort by winrate descending
        for name, wr in failed_opponents:
            print(f" - {name}: WR={wr:.3f}")
        print(f"\nNote: Online learning / fine-tuning is not wired in this script.")
        print(f"To truly 'learn' from battles, connect your PPO trainer to fine-tune the main checkpoint")
        print(f"against a pool of these opponents, then re-run this harness to validate.")
    else:
        print(f"\n🎉 CONGRATULATIONS! Main agent beats all opponents!")
        
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch dump checkpoints and run battle evaluation")
    parser.add_argument("--main_ckpt", type=str, default=MAIN_AGENT_PT,
                       help="Main agent checkpoint filename")
    parser.add_argument("--ckpt_dir", type=str, default=CHECKPOINT_DIR,
                       help="Directory containing .pt checkpoints")
    parser.add_argument("--sub_dir", type=str, default=SUB_DIR,
                       help="Directory to save converted submissions")
    parser.add_argument("--winrate", type=float, default=0.75,
                       help="Win rate threshold to consider opponent 'beaten'")
    parser.add_argument("--games", type=int, default=10,
                       help="Number of games per opponent")
    
    args = parser.parse_args()
    
    # Validate main checkpoint exists
    main_ckpt_path = os.path.join(args.ckpt_dir, args.main_ckpt)
    if not os.path.exists(main_ckpt_path):
        print(f"❌ Main checkpoint not found: {main_ckpt_path}")
        print(f"Available checkpoints:")
        for pt_file in glob.glob(os.path.join(args.ckpt_dir, "*.pt")):
            print(f"  - {os.path.basename(pt_file)}")
        return
        
    # Run evaluation
    results = run_battle_evaluation(
        args.main_ckpt, args.ckpt_dir, args.sub_dir, 
        args.winrate, args.games
    )
    
    # Save results to log file
    log_file = "battle_results.log"
    with open(log_file, 'w') as f:
        f.write(f"Battle evaluation results\n")
        f.write(f"Main agent: {args.main_ckpt}\n")
        f.write(f"Win rate threshold: {args.winrate}\n")
        f.write(f"Games per opponent: {args.games}\n\n")
        
        for opponent, result in results.items():
            if 'error' in result:
                f.write(f"{opponent}: ERROR - {result['error']}\n")
            else:
                wr = result['winrate']
                w, d, l = result['wins'], result['draws'], result['losses']
                f.write(f"{opponent}: WR={wr:.3f} ({w}W-{d}D-{l}L)\n")
    
    print(f"\n📝 Results saved to: {log_file}")


if __name__ == "__main__":
    main()