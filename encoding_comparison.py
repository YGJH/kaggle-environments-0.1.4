#!/usr/bin/env python3
"""
編碼方式記憶體使用對比分析
"""

import numpy as np

def analyze_encoding_memory():
    """分析不同編碼方式的記憶體使用"""
    
    print("🔍 ConnectX 編碼方式記憶體分析")
    print("=" * 60)
    
    # 測試不同樣本數量
    sample_counts = [10000, 50000, 100000, 500000, 1000000]
    
    print("\n📊 記憶體使用對比 (只計算狀態數據):")
    print(f"{'樣本數':<10} {'42維編碼':<12} {'126維編碼':<12} {'節省記憶體':<12} {'節省比例':<10}")
    print("-" * 70)
    
    for samples in sample_counts:
        # 42維編碼記憶體使用 (緊湊編碼)
        compact_memory = (samples * 42 * 4) / 1024 / 1024  # MB
        
        # 126維編碼記憶體使用 (多通道編碼) 
        multichannel_memory = (samples * 126 * 4) / 1024 / 1024  # MB
        
        # 節省的記憶體
        saved_memory = multichannel_memory - compact_memory
        saved_ratio = (saved_memory / multichannel_memory) * 100
        
        print(f"{samples:<10} {compact_memory:.1f} MB    {multichannel_memory:.1f} MB    {saved_memory:.1f} MB    {saved_ratio:.1f}%")
    
    print("\n🎯 編碼方式特性對比:")
    print("\n42維緊湊編碼:")
    print("  ✅ 記憶體使用: 僅原來的 1/3")
    print("  ✅ 載入速度: 更快")
    print("  ✅ 訓練速度: 輸入層小，計算快")
    print("  ⚠️  特徵表達: 需要模型自己學習特徵分離")
    print("  📏 編碼邏輯: -1=對手, 0=空位, 1=自己")
    
    print("\n126維多通道編碼:")
    print("  ✅ 特徵分離: 明確區分自己/對手/空位")  
    print("  ✅ 學習效率: 特徵已預處理，更容易學習")
    print("  ✅ CNN友好: 類似圖像多通道結構")
    print("  ⚠️  記憶體使用: 是42維的3倍")
    print("  ⚠️  計算開銷: 輸入層更大")
    print("  📏 編碼邏輯: 3個42維二元通道")
    
    print("\n💡 建議:")
    print("  📁 小數據集 (<100K樣本): 可使用126維多通道編碼")
    print("  📁 大數據集 (>100K樣本): 建議使用42維緊湊編碼")
    print("  🎯 性能對比: 兩種編碼方式的最終性能通常相近")
    print("  ⚡ 實用選擇: 緊湊編碼在大數據集上更實用")

def demo_encoding():
    """演示兩種編碼方式"""
    print("\n🔍 編碼方式演示:")
    
    # 示例棋盤 (6x7=42)
    # 1=玩家1, 2=玩家2, 0=空位
    board = [
        0, 0, 0, 0, 0, 0, 0,  # 第6行 (頂部)
        0, 0, 0, 0, 0, 0, 0,  # 第5行
        0, 0, 0, 0, 0, 0, 0,  # 第4行
        0, 0, 0, 1, 0, 0, 0,  # 第3行
        0, 0, 2, 1, 0, 0, 0,  # 第2行
        2, 1, 2, 1, 0, 0, 0   # 第1行 (底部)
    ]
    
    print("原始棋盤 (6x7):")
    for row in range(6):
        row_data = board[row*7:(row+1)*7]
        print(f"  {row_data}")
    
    # 42維緊湊編碼 (從玩家1視角)
    mark = 1
    compact_encoded = np.array(board, dtype=np.float32)
    opponent_mark = 3 - mark  # 2
    compact_encoded[compact_encoded == mark] = 1.0      # 自己=1
    compact_encoded[compact_encoded == opponent_mark] = -1.0  # 對手=-1
    # 空位保持0
    
    print(f"\n42維緊湊編碼 (玩家{mark}視角):")
    print(f"  維度: {len(compact_encoded)}")
    print(f"  記憶體: {compact_encoded.nbytes} bytes")
    print("  編碼值: -1=對手, 0=空位, 1=自己")
    for row in range(6):
        row_data = compact_encoded[row*7:(row+1)*7]
        print(f"  {[f'{x:2.0f}' for x in row_data]}")
    
    # 126維多通道編碼
    state = np.array(board).reshape(6, 7)
    
    # 通道1: 當前玩家的棋子
    player_pieces = (state == mark).astype(np.float32)
    # 通道2: 對手的棋子  
    opponent_pieces = (state == (3 - mark)).astype(np.float32)
    # 通道3: 空位
    empty_spaces = (state == 0).astype(np.float32)
    
    # 拉平並連接
    multichannel_encoded = np.concatenate([
        player_pieces.flatten(),
        opponent_pieces.flatten(), 
        empty_spaces.flatten()
    ])
    
    print(f"\n126維多通道編碼:")
    print(f"  維度: {len(multichannel_encoded)}")
    print(f"  記憶體: {multichannel_encoded.nbytes} bytes") 
    print("  通道1 (自己棋子):")
    for row in range(6):
        row_data = player_pieces[row]
        print(f"    {[f'{x:.0f}' for x in row_data]}")
    print("  通道2 (對手棋子):")
    for row in range(6):
        row_data = opponent_pieces[row]
        print(f"    {[f'{x:.0f}' for x in row_data]}")
    print("  通道3 (空位):")
    for row in range(6):
        row_data = empty_spaces[row]
        print(f"    {[f'{x:.0f}' for x in row_data]}")
    
    print(f"\n記憶體對比:")
    print(f"  42維編碼: {compact_encoded.nbytes} bytes")
    print(f"  126維編碼: {multichannel_encoded.nbytes} bytes")
    print(f"  膨脹倍數: {multichannel_encoded.nbytes / compact_encoded.nbytes:.1f}x")

if __name__ == "__main__":
    analyze_encoding_memory()
    demo_encoding()
