#!/usr/bin/env python3
"""
ConnectX 監督學習訓練測試腳本
驗證訓練環境是否正常工作
"""

import os
import sys
import torch
import numpy as np

def test_environment():
    """測試基本環境"""
    print("🧪 測試基本環境...")
    
    # 檢查Python版本
    print(f"   Python版本: {sys.version}")
    
    # 檢查PyTorch
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   GPU數量: {torch.cuda.device_count()}")
    
    # 檢查必要文件
    required_files = [
        "connectx-state-action-value.txt",
        "train_supervised.py"
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ 找到文件: {file}")
        else:
            print(f"   ❌ 缺少文件: {file}")
            return False
    
    return True

def test_kaggle_environments():
    """測試kaggle_environments"""
    print("\n🧪 測試Kaggle環境...")
    
    try:
        from kaggle_environments import make
        env = make("connectx", debug=False)
        print("   ✅ ConnectX環境創建成功")
        
        # 測試環境配置
        config = env.configuration
        print(f"   遊戲配置: {config.rows}x{config.columns}, 連{config.inarow}子獲勝")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ 導入kaggle_environments失敗: {e}")
        return False
    except Exception as e:
        print(f"   ❌ 創建ConnectX環境失敗: {e}")
        return False

def test_data_loading():
    """測試數據載入"""
    print("\n🧪 測試數據載入...")
    
    try:
        # 簡單讀取前幾行測試
        dataset_file = "connectx-state-action-value.txt"
        if not os.path.exists(dataset_file):
            print(f"   ❌ 找不到數據集文件: {dataset_file}")
            return False
        
        with open(dataset_file, 'r') as f:
            lines = []
            for i, line in enumerate(f):
                lines.append(line.strip())
                if i >= 4:  # 只讀前5行測試
                    break
        
        print(f"   ✅ 成功讀取 {len(lines)} 行數據")
        
        # 測試第一行解析
        if lines:
            first_line = lines[0]
            print(f"   第一行數據: {first_line[:50]}...")
            
            # 簡單解析測試
            parts = first_line.split(',')
            if len(parts) >= 8:  # 棋盤狀態 + 7個動作值
                board_part = parts[0]
                action_parts = parts[1:8]
                
                if len(board_part) == 42:
                    print(f"   ✅ 棋盤狀態長度正確: {len(board_part)}")
                else:
                    print(f"   ⚠️ 棋盤狀態長度異常: {len(board_part)}")
                
                print(f"   ✅ 動作值數量: {len(action_parts)}")
                print(f"   動作值示例: {action_parts}")
            else:
                print(f"   ⚠️ 數據格式可能有問題，分割後只有 {len(parts)} 部分")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 數據載入測試失敗: {e}")
        return False

def test_model_creation():
    """測試模型創建"""
    print("\n🧪 測試模型創建...")
    
    try:
        from train_supervised import ConnectXNet, PPOAgent, create_training_config
        
        # 創建配置
        config = create_training_config()
        print("   ✅ 配置創建成功")
        
        # 創建網絡
        net = ConnectXNet(
            input_size=config['agent']['input_size'],
            hidden_size=config['agent']['hidden_size'],
            num_layers=config['agent']['num_layers']
        )
        print("   ✅ ConnectXNet創建成功")
        
        # 測試前向傳播
        test_input = torch.randn(1, config['agent']['input_size'])
        policy, value = net(test_input)
        
        print(f"   策略輸出形狀: {policy.shape}")
        print(f"   價值輸出形狀: {value.shape}")
        print(f"   策略概率和: {policy.sum().item():.4f}")
        print(f"   價值範圍: [{value.min().item():.4f}, {value.max().item():.4f}]")
        
        # 創建PPO智能體
        agent = PPOAgent(config['agent'])
        print("   ✅ PPO智能體創建成功")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 模型創建測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_setup():
    """測試訓練設置"""
    print("\n🧪 測試訓練設置...")
    
    try:
        from train_supervised import ConnectXTrainer, create_training_config
        
        config = create_training_config()
        trainer = ConnectXTrainer(config)
        print("   ✅ 訓練器創建成功")
        
        # 測試數據集載入（只載入少量數據）
        print("   正在測試數據集載入（限制10行）...")
        states, action_values = trainer.load_state_action_dataset(max_lines=10)
        
        if states is not None and action_values is not None:
            print(f"   ✅ 數據載入成功: {len(states)} 個樣本")
            print(f"   狀態形狀: {states.shape}")
            print(f"   動作值形狀: {action_values.shape}")
        else:
            print("   ⚠️ 數據載入失敗或無有效數據")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ 訓練設置測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主測試函數"""
    print("🎮 ConnectX 監督學習環境測試")
    print("=" * 50)
    
    tests = [
        ("基本環境", test_environment),
        ("Kaggle環境", test_kaggle_environments),
        ("數據載入", test_data_loading),
        ("模型創建", test_model_creation),
        ("訓練設置", test_training_setup)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"✅ {test_name} 測試通過")
                passed += 1
            else:
                print(f"❌ {test_name} 測試失敗")
        except Exception as e:
            print(f"❌ {test_name} 測試出錯: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 測試結果: {passed}/{total} 通過")
    
    if passed == total:
        print("🎉 所有測試通過！可以開始訓練")
        print("\n💡 使用方法:")
        print("   python train_supervised.py")
    else:
        print("⚠️ 部分測試失敗，請檢查環境配置")
        
        if passed >= 3:
            print("💡 大部分功能正常，可以嘗試開始訓練")
            print("   python train_supervised.py")

if __name__ == "__main__":
    main()
