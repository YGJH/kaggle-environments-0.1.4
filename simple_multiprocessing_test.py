#!/usr/bin/env python3
"""
简化的多进程测试 - 直接使用基础函数
避免复杂的类序列化问题
"""

import os
import sys
import time
import multiprocessing as mp
import numpy as np
from kaggle_environments import make

def play_simple_game(args):
    """简化的游戏函数 - 用于多进程"""
    episode_id, seed = args
    
    # 设置随机种子
    np.random.seed(seed)
    
    # 创建环境
    env = make("connectx", debug=False)
    
    # 简单的随机对弈
    config = env.configuration
    
    def random_agent(obs, config):
        valid_actions = [c for c in range(config.columns) if obs.board[c] == 0]
        return int(np.random.choice(valid_actions)) if valid_actions else 0
    
    # 运行游戏
    steps = env.run([random_agent, random_agent])
    
    # 计算游戏结果
    reward_1 = 0
    reward_2 = 0
    
    if len(steps) > 0:
        last_step = steps[-1]
        if len(last_step) >= 2:
            # 提取奖励信息
            reward_1 = last_step[0].get('reward', 0) or 0
            reward_2 = last_step[1].get('reward', 0) or 0
            
            # 判断获胜者
            if reward_1 > reward_2:
                winner = 1
            elif reward_2 > reward_1:
                winner = 2
            else:
                winner = 0  # 平局
        else:
            winner = 0
    else:
        winner = 0
    
    return {
        'episode_id': episode_id,
        'winner': winner,
        'steps': len(steps),
        'reward_1': reward_1,
        'reward_2': reward_2
    }

def test_multiprocessing_performance():
    """测试多进程性能"""
    print("🚀 ConnectX 多进程性能测试")
    print(f"📊 可用 CPU 核心数: {mp.cpu_count()}")
    
    num_episodes = 20
    num_processes = min(4, mp.cpu_count())
    
    print(f"🎯 测试参数: {num_episodes} episodes, {num_processes} 进程")
    
    # 准备参数
    args = [(i, np.random.randint(0, 10000)) for i in range(num_episodes)]
    
    # 1. 多进程测试
    print("\n1️⃣ 多进程测试")
    start_time = time.time()
    
    with mp.Pool(processes=num_processes) as pool:
        mp_results = pool.map(play_simple_game, args)
    
    mp_time = time.time() - start_time
    print(f"   ✅ 完成: {len(mp_results)} episodes, 用时: {mp_time:.2f}s")
    
    # 分析多进程结果
    winners = [r['winner'] for r in mp_results]
    print(f"   📊 游戏结果: Player1胜={winners.count(1)}, Player2胜={winners.count(2)}, 平局={winners.count(0)}")
    
    # 2. 单进程测试
    print("\n2️⃣ 单进程测试")
    start_time = time.time()
    
    single_results = []
    for arg in args:
        result = play_simple_game(arg)
        single_results.append(result)
    
    single_time = time.time() - start_time
    print(f"   ✅ 完成: {len(single_results)} episodes, 用时: {single_time:.2f}s")
    
    # 分析单进程结果
    winners = [r['winner'] for r in single_results]
    print(f"   📊 游戏结果: Player1胜={winners.count(1)}, Player2胜={winners.count(2)}, 平局={winners.count(0)}")
    
    # 3. 性能分析
    print("\n📈 性能分析:")
    if single_time > 0 and mp_time > 0:
        speedup = single_time / mp_time
        efficiency = speedup / num_processes
        print(f"   🚀 加速比: {speedup:.2f}x")
        print(f"   💡 并行效率: {efficiency:.2f} ({efficiency*100:.1f}%)")
        print(f"   ⚡ 时间节省: {single_time - mp_time:.2f}s ({(single_time - mp_time)/single_time*100:.1f}%)")
        
        if speedup > 1.0:
            print("   ✅ 多进程带来了性能提升!")
        else:
            print("   ⚠️ 多进程未带来性能提升，可能是:")
            print("      - 任务过于简单，多进程开销大于收益")
            print("      - 环境创建开销较大")
            print("      - 需要更多episode数量才能体现优势")
    
    return mp_results, single_results, speedup if 'speedup' in locals() else 0

def main():
    """主函数"""
    try:
        # 设置多进程启动方法
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    mp_results, single_results, speedup = test_multiprocessing_performance()
    
    print(f"\n🎉 测试完成! 最终加速比: {speedup:.2f}x")
    
    # 建议
    if speedup > 1.5:
        print("💡 建议: 多进程效果良好，可以在训练中使用")
    elif speedup > 1.0:
        print("💡 建议: 多进程有轻微提升，可根据需要使用")
    else:
        print("💡 建议: 当前任务不适合多进程，或需要调整参数")
        print("   - 尝试增加episode数量")
        print("   - 减少进程数量")
        print("   - 使用更复杂的agent逻辑")

if __name__ == "__main__":
    main()
