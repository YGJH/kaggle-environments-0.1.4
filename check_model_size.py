#!/usr/bin/env python3
"""
檢查模型大小和預估 submission.py 文件大小
"""
import torch
import numpy as np
import os
import base64

def check_model_size():
    """檢查模型大小和參數數量"""
    print("🔍 檢查模型大小和參數")
    print("=" * 50)
    
    # 檢查是否有模型文件
    checkpoint_files = []
    if os.path.exists("checkpoints"):
        for file in os.listdir("checkpoints"):
            if file.endswith(".pt"):
                checkpoint_files.append(file)
    
    if not checkpoint_files:
        print("❌ 沒有找到任何 .pt 模型文件")
        return
    
    # 使用最新的最佳模型
    best_models = [f for f in checkpoint_files if f.startswith('best_model')]
    if best_models:
        model_file = sorted(best_models)[-1]
    else:
        model_file = sorted(checkpoint_files)[-1]
    
    model_path = f"checkpoints/{model_file}"
    print(f"📁 檢查模型: {model_path}")
    
    # 檢查文件大小
    file_size = os.path.getsize(model_path)
    print(f"📊 模型文件大小: {file_size / 1024 / 1024:.2f} MB")
    
    # 載入模型
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint['model_state_dict']
        
        # 計算參數數量
        total_params = 0
        for name, param in state_dict.items():
            param_count = param.numel()
            total_params += param_count
            print(f"  {name}: {param.shape} -> {param_count:,} 參數")
        
        print(f"\n📈 總參數數量: {total_params:,}")
        print(f"📈 估計模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
        
        # 測試 numpy 壓縮效果
        print(f"\n🗜️ 測試壓縮效果:")
        
        # 轉換為 numpy 並保存為 .npz
        temp_file = "temp_weights.npz"
        np.savez_compressed(temp_file, **{k: v.numpy() for k, v in state_dict.items()})
        
        npz_size = os.path.getsize(temp_file)
        print(f"  .npz 壓縮後: {npz_size / 1024 / 1024:.2f} MB")
        
        # 測試 Base64 編碼大小
        with open(temp_file, "rb") as f:
            weights_bytes = f.read()
            weights_b64 = base64.b64encode(weights_bytes).decode('utf-8')
        
        b64_size = len(weights_b64)
        print(f"  Base64 編碼: {b64_size / 1024 / 1024:.2f} MB")
        
        # 估計最終 submission.py 大小
        # Base64 字符串 + Python 代碼
        code_overhead = 5000  # 估計 Python 代碼大小
        estimated_submission_size = b64_size + code_overhead
        
        print(f"\n📋 預估 submission.py 大小:")
        print(f"  Base64 權重: {b64_size / 1024 / 1024:.2f} MB")
        print(f"  Python 代碼: {code_overhead / 1024:.2f} KB")
        print(f"  總大小: {estimated_submission_size / 1024 / 1024:.2f} MB")
        
        # 檢查是否超過限制
        size_limit_mb = 100
        if estimated_submission_size / 1024 / 1024 > size_limit_mb:
            print(f"\n⚠️ 警告: 預估大小 ({estimated_submission_size / 1024 / 1024:.2f} MB) 超過 {size_limit_mb} MB 限制！")
            
            # 建議優化方案
            print(f"\n💡 優化建議:")
            print(f"  1. 減少隱藏層大小 (當前: 512)")
            print(f"  2. 減少殘差塊數量 (當前: 6)")
            print(f"  3. 使用 float16 精度")
            print(f"  4. 進行模型剪枝")
            
            # 計算建議的參數大小
            target_size_mb = 80  # 留一些緩衝
            target_params = int(target_size_mb * 1024 * 1024 / 4)  # 假設 float32
            print(f"\n🎯 建議參數數量: < {target_params:,} (當前: {total_params:,})")
            
        else:
            print(f"\n✅ 模型大小在限制範圍內 (< {size_limit_mb} MB)")
        
        # 清理臨時文件
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    except Exception as e:
        print(f"❌ 載入模型失敗: {e}")

def check_config_impact():
    """檢查配置對模型大小的影響"""
    print(f"\n🔧 當前配置分析:")
    print("=" * 30)
    
    try:
        import yaml
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        agent_config = config.get('agent', {})
        
        input_size = agent_config.get('input_size', 126)
        hidden_size = agent_config.get('hidden_size', 512)
        num_layers = agent_config.get('num_layers', 4)
        
        print(f"  輸入大小: {input_size}")
        print(f"  隱藏層大小: {hidden_size}")
        print(f"  隱藏層數量: {num_layers}")
        
        # 計算預期參數數量
        # 輸入層: input_size * hidden_size + hidden_size
        input_params = input_size * hidden_size + hidden_size
        
        # 每個殘差塊: 2 * (hidden_size * hidden_size + hidden_size)
        residual_params = num_layers * 2 * (hidden_size * hidden_size + hidden_size)
        
        # 策略頭: hidden_size * (hidden_size // 2) + (hidden_size // 2) + (hidden_size // 2) * 7 + 7
        policy_params = hidden_size * (hidden_size // 2) + (hidden_size // 2) + (hidden_size // 2) * 7 + 7
        
        # 價值頭: hidden_size * (hidden_size // 2) + (hidden_size // 2) + (hidden_size // 2) * 1 + 1
        value_params = hidden_size * (hidden_size // 2) + (hidden_size // 2) + (hidden_size // 2) * 1 + 1
        
        total_estimated = input_params + residual_params + policy_params + value_params
        
        print(f"\n📊 預估參數分布:")
        print(f"  輸入層: {input_params:,}")
        print(f"  殘差塊: {residual_params:,}")
        print(f"  策略頭: {policy_params:,}")
        print(f"  價值頭: {value_params:,}")
        print(f"  總計: {total_estimated:,}")
        print(f"  估計大小: {total_estimated * 4 / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"❌ 無法讀取配置: {e}")

if __name__ == "__main__":
    check_model_size()
    check_config_impact()
