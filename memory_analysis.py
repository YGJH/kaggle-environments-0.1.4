#!/usr/bin/env python3
"""
記憶體使用分析腳本
分析不同載入方式的理論記憶體使用情況
"""

import os
import sys
import numpy as np
import time
from datetime import datetime

def analyze_file_size(file_path):
    """分析文件大小"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None
    
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / 1024 / 1024
    
    print(f"📁 文件分析: {file_path}")
    print(f"   文件大小: {file_size_mb:.1f} MB ({file_size:,} bytes)")
    
    # 估算行數
    with open(file_path, 'r') as f:
        first_line = f.readline()
        line_length = len(first_line)
    
    estimated_lines = file_size // line_length if line_length > 0 else 0
    print(f"   估算行數: {estimated_lines:,}")
    print(f"   平均行長: {line_length} 字符")
    
    return {
        'file_size_mb': file_size_mb,
        'estimated_lines': estimated_lines,
        'line_length': line_length
    }

def calculate_memory_usage(num_samples):
    """計算理論記憶體使用情況"""
    # 每個樣本的數據結構大小
    state_size = 126 * 4  # 126個float32，每個4字節
    action_values_size = 7 * 4  # 7個float32，每個4字節
    
    sample_size = state_size + action_values_size  # 每個樣本的總大小
    
    # 原始方式的記憶體使用（重複儲存）
    original_memory = (
        num_samples * 50 +  # 原始文本行（平均50字符per行）
        num_samples * sample_size * 2 +  # Python列表儲存（有overhead）
        num_samples * sample_size  # numpy數組
    ) / 1024 / 1024  # 轉換為MB
    
    # 優化方式的記憶體使用（預分配）
    optimized_memory = (
        num_samples * sample_size  # 直接numpy數組
    ) / 1024 / 1024  # 轉換為MB
    
    return {
        'original': original_memory,
        'optimized': optimized_memory,
        'saved': original_memory - optimized_memory,
        'ratio': original_memory / optimized_memory if optimized_memory > 0 else 1
    }

def analyze_memory_expansion():
    """分析記憶體膨脹原因"""
    print(f"\n🔍 記憶體膨脹原因分析")
    print("=" * 50)
    
    print("原始數據格式:")
    print("  - 文本文件: 每行約50個字符 (ASCII)")
    print("  - 儲存大小: 50 bytes per sample")
    
    print("\n載入後數據格式:")
    print("  - 狀態編碼: 126個float32 = 126 × 4 = 504 bytes")
    print("  - 動作價值: 7個float32 = 7 × 4 = 28 bytes")
    print("  - 總計: 532 bytes per sample")
    
    print(f"\n膨脹比例: 532 / 50 = {532/50:.1f}x")
    
    print("\n原始載入方式的問題:")
    print("  1. 🔄 重複儲存:")
    print("     - 原始文本行")
    print("     - Python列表 (有額外overhead)")
    print("     - 最終numpy數組")
    print("  2. 📈 動態擴展:")
    print("     - list.append()會預分配額外空間")
    print("     - 造成記憶體碎片")
    print("  3. 🔀 數據類型轉換:")
    print("     - 字符串 → 數字 → numpy數組")
    print("     - 多次類型轉換和複製")

def main():
    """主分析函數"""
    print("🔍 ConnectX 記憶體使用分析")
    print("=" * 60)
    
    dataset_file = "connectx-state-action-value.txt"
    
    # 分析文件
    file_info = analyze_file_size(dataset_file)
    if file_info is None:
        print("使用模擬數據進行分析...")
        file_info = {
            'file_size_mb': 7800,  # 7.8GB
            'estimated_lines': 150000000,  # 1.5億行
            'line_length': 50
        }
    
    # 分析記憶體膨脹
    analyze_memory_expansion()
    
    # 測試不同樣本數量
    test_sizes = [1000, 10000, 50000, 100000, 500000]
    
    print(f"\n📊 記憶體使用比較")
    print(f"{'樣本數':<10} {'原始方式':<12} {'優化方式':<12} {'節省':<10} {'膨脹比':<8}")
    print("-" * 60)
    
    for test_size in test_sizes:
        memory_info = calculate_memory_usage(test_size)
        
        print(f"{test_size:<10} {memory_info['original']:.1f} MB     {memory_info['optimized']:.1f} MB     {memory_info['saved']:.1f} MB   {memory_info['ratio']:.1f}x")
    
    # 特別分析大數據集
    full_dataset_samples = min(file_info['estimated_lines'], 1000000)  # 限制到100萬樣本
    full_memory_info = calculate_memory_usage(full_dataset_samples)
    
    print(f"\n🎯 完整數據集分析 ({full_dataset_samples:,} 樣本):")
    print(f"   原始方式: {full_memory_info['original']:.1f} MB ({full_memory_info['original']/1024:.1f} GB)")
    print(f"   優化方式: {full_memory_info['optimized']:.1f} MB ({full_memory_info['optimized']/1024:.1f} GB)")
    print(f"   節省記憶體: {full_memory_info['saved']:.1f} MB ({full_memory_info['saved']/1024:.1f} GB)")
    print(f"   效率提升: {full_memory_info['ratio']:.1f}x")
    
    print(f"\n💡 優化效果:")
    print(f"   ✅ 消除重複儲存，記憶體使用減少 {(full_memory_info['ratio']-1)/full_memory_info['ratio']*100:.0f}%")
    print(f"   ✅ 預分配機制避免動態擴展開銷")
    print(f"   ✅ 減少記憶體碎片化")
    print(f"   ✅ 支援分批載入，適應大數據集")
    
    print(f"\n🎯 建議:")
    if file_info['file_size_mb'] > 1000:  # 大於1GB
        print(f"   📁 文件較大 ({file_info['file_size_mb']:.1f} MB)，強烈建議使用記憶體優化模式")
        print(f"   💾 可將 max_lines 設為 20000-100000 進行分批訓練")
        print(f"   🔧 在訓練配置中設置 memory_efficient=True")
    else:
        print(f"   📁 文件適中 ({file_info['file_size_mb']:.1f} MB)，可使用標準模式一次載入")
        print(f"   ⚡ 但優化模式仍然更有效率")

if __name__ == "__main__":
    main()
