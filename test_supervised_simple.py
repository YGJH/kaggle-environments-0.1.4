#!/usr/bin/env python3
"""
测试简化版本的 ConnectX 监督学习训练程序
"""

import os
import torch
import numpy as np

def test_basic_setup():
    """测试基本设置"""
    print("🧪 测试基本设置...")
    
    # 检查数据集文件
    if not os.path.exists("connectx-state-action-value.txt"):
        print("   ❌ 找不到数据集文件")
        return False
    
    print("   ✅ 找到数据集文件")
    
    # 检查CUDA
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
    
    return True

def test_imports():
    """测试导入"""
    print("\n🧪 测试导入...")
    
    try:
        from train_connectx_supervised import ConnectXNet, ConnectXTrainer, create_config
        print("   ✅ 成功导入所有类")
        return True
    except ImportError as e:
        print(f"   ❌ 导入失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n🧪 测试模型创建...")
    
    try:
        from train_connectx_supervised import ConnectXNet, ConnectXTrainer, create_config
        
        config = create_config()
        trainer = ConnectXTrainer(config)
        
        print("   ✅ 训练器创建成功")
        
        # 测试编码功能
        test_board = [0] * 42  # 空棋盘
        encoded = trainer.encode_state(test_board, 1)
        print(f"   编码状态形状: {encoded.shape}")
        
        # 测试网络
        test_input = torch.randn(1, 126).to(trainer.device)  # 确保在正确设备上
        policy, value = trainer.policy_net(test_input)
        print(f"   策略输出: {policy.shape}, 值: {value.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """测试数据载入"""
    print("\n🧪 测试数据载入...")
    
    try:
        from train_connectx_supervised import ConnectXTrainer, create_config
        
        config = create_config()
        trainer = ConnectXTrainer(config)
        
        # 只载入前10行测试
        states, action_values = trainer.load_dataset(max_lines=10)
        
        if states is not None and action_values is not None:
            print(f"   ✅ 数据载入成功: {len(states)} 个样本")
            print(f"   状态形状: {states.shape}")
            print(f"   动作值形状: {action_values.shape}")
            return True
        else:
            print("   ❌ 数据载入失败")
            return False
            
    except Exception as e:
        print(f"   ❌ 数据载入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mini_training():
    """测试迷你训练"""
    print("\n🧪 测试迷你训练...")
    
    try:
        from train_connectx_supervised import ConnectXTrainer, create_config
        
        config = create_config()
        config['training']['epochs'] = 2  # 只训练2个epoch
        config['training']['max_lines'] = 50  # 只用50行数据
        
        trainer = ConnectXTrainer(config)
        
        # 开始迷你训练
        model = trainer.train(epochs=2, max_lines=50)
        
        if model is not None:
            print("   ✅ 迷你训练成功完成")
            return True
        else:
            print("   ❌ 迷你训练失败")
            return False
            
    except Exception as e:
        print(f"   ❌ 迷你训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🎮 ConnectX 监督学习训练测试 - 简化版本")
    print("=" * 60)
    
    tests = [
        ("基本设置", test_basic_setup),
        ("导入测试", test_imports),
        ("模型创建", test_model_creation),
        ("数据载入", test_data_loading),
        ("迷你训练", test_mini_training)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"✅ {test_name} 通过")
                passed += 1
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 出错: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！可以开始正式训练")
        print("\n💡 开始训练:")
        print("   python train_connectx_supervised.py")
    elif passed >= 3:
        print("💡 大部分功能正常，可以尝试训练")
        print("   python train_connectx_supervised.py")
    else:
        print("⚠️ 部分重要功能失败，请检查环境")

if __name__ == "__main__":
    main()
