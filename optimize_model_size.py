#!/usr/bin/env python3
"""
自動化測試模型配置以找到最大可用模型大小（接近 100MB）
"""
import torch
import torch.nn as nn
import numpy as np
import yaml
import os
import base64
import math
from itertools import product

class ConnectXNet(nn.Module):
    """ConnectX 神經網路"""
    
    def __init__(self, input_size=126, hidden_size=512, num_layers=4):
        super(ConnectXNet, self).__init__()
        
        # 輸入層
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 隱藏層（殘差連接）
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size)
            ) for _ in range(num_layers)
        ])
        
        # 策略頭（動作概率）
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 7),
            nn.Softmax(dim=-1)
        )
        
        # 價值頭（狀態價值）
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # 輸入處理
        x = self.input_layer(x)
        
        # 殘差連接
        for hidden_layer in self.hidden_layers:
            residual = x
            x = hidden_layer(x)
            x = torch.relu(x + residual)
        
        # 輸出頭
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value

def calculate_model_params(input_size, hidden_size, num_layers):
    """計算模型參數數量"""
    # 輸入層: input_size * hidden_size + hidden_size
    input_params = input_size * hidden_size + hidden_size
    
    # 每個殘差塊: 2 * (hidden_size * hidden_size + hidden_size)
    residual_params = num_layers * 2 * (hidden_size * hidden_size + hidden_size)
    
    # 策略頭: hidden_size * (hidden_size // 2) + (hidden_size // 2) + (hidden_size // 2) * 7 + 7
    policy_params = hidden_size * (hidden_size // 2) + (hidden_size // 2) + (hidden_size // 2) * 7 + 7
    
    # 價值頭: hidden_size * (hidden_size // 2) + (hidden_size // 2) + (hidden_size // 2) * 1 + 1
    value_params = hidden_size * (hidden_size // 2) + (hidden_size // 2) + (hidden_size // 2) * 1 + 1
    
    total_params = input_params + residual_params + policy_params + value_params
    return total_params

def estimate_submission_size(total_params):
    """估計 submission.py 大小（MB）"""
    # 經驗公式：基於實際測試數據
    # 參數數量 -> numpy 大小 -> Base64 大小 -> 最終文件大小
    numpy_size_mb = total_params * 4 / 1024 / 1024  # float32
    npz_compressed_ratio = 0.965  # 基於實際測試的壓縮比
    base64_ratio = 1.333  # Base64 編碼增加約 33%
    code_overhead_mb = 0.005  # Python 代碼大小
    
    estimated_size = numpy_size_mb * npz_compressed_ratio * base64_ratio + code_overhead_mb
    return estimated_size

def test_model_configuration(input_size, hidden_size, num_layers, target_size_mb=95):
    """測試特定配置是否符合大小要求"""
    total_params = calculate_model_params(input_size, hidden_size, num_layers)
    estimated_size = estimate_submission_size(total_params)
    
    return {
        'config': (hidden_size, num_layers),
        'params': total_params,
        'estimated_size_mb': estimated_size,
        'within_limit': estimated_size <= target_size_mb,
        'efficiency': total_params / (estimated_size * 1024 * 1024)  # 參數/字節比
    }

def create_and_test_model(hidden_size, num_layers):
    """創建並實際測試模型大小"""
    print(f"  🧪 測試實際模型: hidden_size={hidden_size}, num_layers={num_layers}")
    
    try:
        # 創建模型
        model = ConnectXNet(input_size=126, hidden_size=hidden_size, num_layers=num_layers)
        
        # 計算實際參數數量
        actual_params = sum(p.numel() for p in model.parameters())
        
        # 模擬保存和壓縮過程
        state_dict = model.state_dict()
        
        # 轉換為 numpy
        temp_file = f"temp_test_{hidden_size}_{num_layers}.npz"
        np.savez_compressed(temp_file, **{k: v.numpy() for k, v in state_dict.items()})
        
        # 檢查壓縮大小
        npz_size = os.path.getsize(temp_file)
        
        # Base64 編碼大小
        with open(temp_file, "rb") as f:
            weights_bytes = f.read()
            weights_b64 = base64.b64encode(weights_bytes).decode('utf-8')
        
        b64_size = len(weights_b64)
        code_overhead = 5000  # 估計 Python 代碼大小
        actual_submission_size = (b64_size + code_overhead) / 1024 / 1024
        
        # 清理臨時文件
        os.remove(temp_file)
        
        return {
            'actual_params': actual_params,
            'npz_size_mb': npz_size / 1024 / 1024,
            'submission_size_mb': actual_submission_size,
            'success': True
        }
        
    except Exception as e:
        print(f"    ❌ 測試失敗: {e}")
        return {'success': False, 'error': str(e)}

def find_optimal_config():
    """找到最優配置"""
    print("🔍 尋找最大可用模型配置")
    print("=" * 60)
    
    input_size = 126
    target_size_mb = 95  # 95MB 安全邊界
    
    # 定義搜索範圍
    hidden_sizes = [384, 512, 640, 768, 896, 1024, 1152, 1280]
    num_layers_range = [3, 4, 5, 6, 7, 8, 9, 10]
    
    results = []
    
    print(f"📊 理論估算階段:")
    print(f"{'Hidden Size':<12} {'Layers':<8} {'Parameters':<12} {'Est. Size':<12} {'Within Limit':<12}")
    print("-" * 66)
    
    # 第一階段：理論估算
    for hidden_size, num_layers in product(hidden_sizes, num_layers_range):
        result = test_model_configuration(input_size, hidden_size, num_layers, target_size_mb)
        results.append(result)
        
        status = "✅" if result['within_limit'] else "❌"
        print(f"{hidden_size:<12} {num_layers:<8} {result['params']:<12,} {result['estimated_size_mb']:<12.1f} {status:<12}")
    
    # 篩選出在限制內的配置
    valid_configs = [r for r in results if r['within_limit']]
    
    if not valid_configs:
        print("\n❌ 沒有找到符合大小限制的配置！")
        return None
    
    # 按參數數量排序，找出最大的幾個
    valid_configs.sort(key=lambda x: x['params'], reverse=True)
    top_configs = valid_configs[:5]
    
    print(f"\n🎯 前 5 個最大的有效配置:")
    print(f"{'Rank':<6} {'Config':<15} {'Parameters':<12} {'Est. Size':<12} {'Efficiency':<12}")
    print("-" * 66)
    
    for i, config in enumerate(top_configs, 1):
        hidden_size, num_layers = config['config']
        print(f"{i:<6} {hidden_size}x{num_layers:<10} {config['params']:<12,} {config['estimated_size_mb']:<12.1f} {config['efficiency']:<12.1f}")
    
    # 第二階段：實際測試前三個配置
    print(f"\n🧪 實際測試階段:")
    print("-" * 40)
    
    best_config = None
    best_actual_size = 0
    
    for i, config in enumerate(top_configs[:3]):
        hidden_size, num_layers = config['config']
        print(f"\n📋 測試配置 {i+1}: {hidden_size}x{num_layers}")
        
        actual_result = create_and_test_model(hidden_size, num_layers)
        
        if actual_result['success']:
            actual_size = actual_result['submission_size_mb']
            print(f"    實際參數: {actual_result['actual_params']:,}")
            print(f"    .npz 大小: {actual_result['npz_size_mb']:.2f} MB")
            print(f"    提交文件大小: {actual_size:.2f} MB")
            
            if actual_size <= target_size_mb and actual_size > best_actual_size:
                best_config = {
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'params': actual_result['actual_params'],
                    'size_mb': actual_size
                }
                best_actual_size = actual_size
                print(f"    ✅ 新的最佳配置！")
            elif actual_size > target_size_mb:
                print(f"    ❌ 超過大小限制 ({target_size_mb} MB)")
            else:
                print(f"    ⚠️ 可用但不是最大")
    
    return best_config

def update_config_file(best_config):
    """更新配置文件"""
    if not best_config:
        print("\n❌ 沒有找到最佳配置，不更新文件")
        return
    
    print(f"\n📝 更新配置文件")
    print("-" * 30)
    
    # 讀取當前配置
    config_file = 'config.yaml'
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 備份原始配置
    backup_file = 'config_backup.yaml'
    with open(backup_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"✅ 原始配置已備份到: {backup_file}")
    
    # 更新配置
    old_hidden = config['agent']['hidden_size']
    old_layers = config['agent']['num_layers']
    
    config['agent']['hidden_size'] = best_config['hidden_size']
    config['agent']['num_layers'] = best_config['num_layers']
    
    # 保存新配置
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 配置已更新:")
    print(f"  hidden_size: {old_hidden} → {best_config['hidden_size']}")
    print(f"  num_layers: {old_layers} → {best_config['num_layers']}")
    print(f"  預估參數: {best_config['params']:,}")
    print(f"  預估大小: {best_config['size_mb']:.2f} MB")

def main():
    """主函數"""
    print("🎯 自動化模型大小優化工具")
    print("目標：找到接近 100MB 限制的最大模型配置")
    print("=" * 60)
    
    # 檢查當前配置
    print("📋 當前配置:")
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            current_config = yaml.safe_load(f)
        
        current_hidden = current_config['agent']['hidden_size']
        current_layers = current_config['agent']['num_layers']
        current_params = calculate_model_params(126, current_hidden, current_layers)
        current_size = estimate_submission_size(current_params)
        
        print(f"  hidden_size: {current_hidden}")
        print(f"  num_layers: {current_layers}")
        print(f"  參數數量: {current_params:,}")
        print(f"  預估大小: {current_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ 讀取當前配置失敗: {e}")
        return
    
    # 尋找最優配置
    best_config = find_optimal_config()
    
    if best_config:
        print(f"\n🏆 找到最佳配置:")
        print(f"  hidden_size: {best_config['hidden_size']}")
        print(f"  num_layers: {best_config['num_layers']}")
        print(f"  參數數量: {best_config['params']:,}")
        print(f"  實際大小: {best_config['size_mb']:.2f} MB")
        
        # 詢問是否更新配置
        response = input(f"\n是否要更新配置文件？ (y/n): ").lower().strip()
        if response == 'y':
            update_config_file(best_config)
            print(f"\n🎉 配置優化完成！")
            print(f"💡 提示：現在可以使用新配置訓練更大的模型了")
        else:
            print(f"\n配置未更改")
    else:
        print(f"\n❌ 未找到合適的配置")

if __name__ == "__main__":
    main()
