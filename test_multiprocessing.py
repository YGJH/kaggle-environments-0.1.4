#!/usr/bin/env python3
"""
测试多进程训练功能
"""

import os
import sys
import time
import multiprocessing as mp
from train_connectx_rl_robust import ConnectXTrainer

def test_multiprocessing():
    """测试多进程训练功能"""
    print("🚀 测试多进程 ConnectX 训练")
    print(f"📊 可用 CPU 核心数: {mp.cpu_count()}")
    
    # 创建测试配置
    test_config = {
        'agent': {
            'input_size': 126,
            'hidden_size': 128,  # 较小的网络用于测试
            'num_layers': 2,     # 更小的网络
            'learning_rate': 0.001,
            'gamma': 0.99,
            'k_epochs': 2,
            'eps_clip': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'min_batch_size': 16,
            'weight_decay': 0.01,  # 添加缺失的参数
            'lr_decay': 0.995,     # 添加缺失的参数
            'buffer_size': 1000,   # 添加缺失的参数
            'gae_lambda': 0.95     # 添加缺失的GAE参数
        },
        'training': {
            'max_episodes': 100,  # 少量episode用于测试
            'eval_frequency': 50,
            'eval_games': 20,
            'checkpoint_frequency': 100,
            'early_stopping_patience': 200,
            'opponent_diversity': True,
            'use_multiprocessing': True,
            'num_processes': min(4, mp.cpu_count() - 1),
            'parallel_episodes': 4
        },
        'evaluation': {
            'mode': 'random',
            'num_games': 20
        }
    }
    
    print("⚙️ 配置信息:")
    print(f"  - 进程数: {test_config['training']['num_processes']}")
    print(f"  - 并行episode数: {test_config['training']['parallel_episodes']}")
    print(f"  - 网络大小: {test_config['agent']['hidden_size']} 隐藏单元, {test_config['agent']['num_layers']} 层")
    
    # 创建训练器
    trainer = ConnectXTrainer(test_config)
    
    print("\n🧪 测试1: 并行自对弈")
    start_time = time.time()
    results = trainer.parallel_self_play_episodes(4)
    mp_time = time.time() - start_time
    print(f"  ✅ 完成 {len(results)} 个并行episode")
    print(f"  ⏱️ 多进程用时: {mp_time:.2f}s")
    
    print("\n🧪 测试2: 单进程对比")
    trainer.use_multiprocessing = False
    start_time = time.time()
    results_single = [trainer.self_play_episode() for _ in range(4)]
    single_time = time.time() - start_time
    print(f"  ✅ 完成 {len(results_single)} 个单进程episode")
    print(f"  ⏱️ 单进程用时: {single_time:.2f}s")
    
    print("\n📈 性能对比:")
    if single_time > 0:
        speedup = single_time / mp_time
        print(f"  🚀 加速比: {speedup:.2f}x")
        print(f"  💡 效率: {speedup / test_config['training']['num_processes']:.2f}")
    
    print("\n🧪 测试3: 并行评估")
    trainer.use_multiprocessing = True
    start_time = time.time()
    win_rate = trainer.parallel_evaluation(20, 'random')
    eval_time = time.time() - start_time
    print(f"  ✅ 评估胜率: {win_rate:.3f}")
    print(f"  ⏱️ 评估用时: {eval_time:.2f}s")
    
    print("\n🧪 测试4: 短期训练")
    print("  开始短期多进程训练...")
    start_time = time.time()
    
    # 运行少量训练步骤
    original_max_episodes = trainer.config['training']['max_episodes']
    trainer.config['training']['max_episodes'] = 20
    
    try:
        trainer.train()
        training_time = time.time() - start_time
        print(f"  ✅ 训练完成，用时: {training_time:.2f}s")
        print(f"  📊 总episode数: {len(trainer.episode_rewards)}")
        if trainer.win_rates:
            print(f"  🎯 最终评估分数: {trainer.win_rates[-1]:.3f}")
    except Exception as e:
        print(f"  ❌ 训练出错: {e}")
    finally:
        trainer.config['training']['max_episodes'] = original_max_episodes
    
    print("\n✅ 多进程功能测试完成！")

if __name__ == "__main__":
    # 设置多进程启动方法（重要！）
    if __name__ == "__main__":
        mp.set_start_method('spawn', force=True)
    
    test_multiprocessing()
