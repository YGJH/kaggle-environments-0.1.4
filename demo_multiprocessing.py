#!/usr/bin/env python3
"""
简化的多进程训练演示
展示如何在ConnectX训练中使用多进程提升性能
"""

import os
import sys
import time
import multiprocessing as mp
import numpy as np
from train_connectx_rl_robust import ConnectXTrainer

def main():
    """主函数"""
    print("🚀 ConnectX 多进程训练演示")
    print(f"📊 可用 CPU 核心数: {mp.cpu_count()}")
    
    # 使用现有的配置文件
    config_file = "config_multiprocessing.yaml"
    
    if not os.path.exists(config_file):
        print(f"❌ 配置文件 {config_file} 不存在")
        return
    
    print(f"📁 使用配置文件: {config_file}")
    
    # 创建训练器
    trainer = ConnectXTrainer(config_file)
    
    print(f"🔧 训练配置:")
    print(f"   - 多进程: {'启用' if trainer.use_multiprocessing else '禁用'}")
    print(f"   - 进程数: {trainer.num_processes}")
    print(f"   - 并行episode数: {trainer.parallel_episodes}")
    print(f"   - 网络大小: {trainer.config['agent']['hidden_size']} 隐藏单元")
    print(f"   - 最大训练回合: {trainer.config['training']['max_episodes']}")
    
    # 性能对比测试
    print("\n🧪 性能对比测试:")
    
    # 1. 多进程自对弈测试
    print("1️⃣ 多进程自对弈测试 (8 episodes)")
    start_time = time.time()
    mp_results = trainer.parallel_self_play_episodes(8)
    mp_time = time.time() - start_time
    print(f"   ✅ 完成: {len(mp_results)} episodes, 用时: {mp_time:.2f}s")
    
    # 2. 单进程对比测试
    print("2️⃣ 单进程对比测试 (8 episodes)")
    trainer.use_multiprocessing = False
    start_time = time.time()
    single_results = []
    for i in range(8):
        result = trainer.self_play_episode()
        single_results.append(result)
    single_time = time.time() - start_time
    print(f"   ✅ 完成: {len(single_results)} episodes, 用时: {single_time:.2f}s")
    
    # 性能分析
    print("\n📈 性能分析:")
    if single_time > 0:
        speedup = single_time / mp_time if mp_time > 0 else 0
        efficiency = speedup / trainer.num_processes if trainer.num_processes > 0 else 0
        print(f"   🚀 加速比: {speedup:.2f}x")
        print(f"   💡 并行效率: {efficiency:.2f} ({efficiency*100:.1f}%)")
        print(f"   ⚡ 时间节省: {single_time - mp_time:.2f}s ({(single_time - mp_time)/single_time*100:.1f}%)")
    
    # 启动完整训练（可选）
    print("\n🎯 是否启动完整多进程训练？")
    response = input("输入 'y' 开始训练，其他键跳过: ").lower().strip()
    
    if response == 'y':
        print("\n🚀 开始多进程训练...")
        trainer.use_multiprocessing = True
        
        # 限制训练轮数为演示用
        original_max_episodes = trainer.config['training']['max_episodes'] 
        trainer.config['training']['max_episodes'] = min(1000, original_max_episodes)
        
        start_time = time.time()
        try:
            trained_agent = trainer.train()
            training_time = time.time() - start_time
            
            print(f"\n✅ 训练完成!")
            print(f"   ⏱️ 总用时: {training_time:.1f}s ({training_time/60:.1f}m)")
            print(f"   📊 完成episode数: {len(trainer.episode_rewards)}")
            print(f"   🎯 最终胜率: {trainer.win_rates[-1]:.3f}" if trainer.win_rates else "N/A")
            
            # 保存最终模型
            trainer.save_checkpoint("multiprocessing_demo_final.pt")
            print(f"   💾 模型已保存: checkpoints/multiprocessing_demo_final.pt")
            
        except KeyboardInterrupt:
            print("\n⏹️ 训练被用户中断")
        except Exception as e:
            print(f"\n❌ 训练出错: {e}")
        finally:
            trainer.config['training']['max_episodes'] = original_max_episodes
    
    print("\n🎉 多进程训练演示完成!")

if __name__ == "__main__":
    # 设置多进程启动方法
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main()
