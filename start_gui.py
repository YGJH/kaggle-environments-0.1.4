#!/usr/bin/env python3
"""
啟動 ConnectX GUI 界面的腳本
讓你可以在瀏覽器中與 AI 對戰
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_submission():
    """檢查 submission.py 是否存在"""
    if not os.path.exists("submission.py"):
        print("❌ 找不到 submission.py")
        print("請先運行: uv run python dump_weight_fixed.py")
        return False
    print("✅ AI 模型文件存在")
    return True

def start_server():
    """啟動 HTTP 服務器"""
    print("🚀 啟動 Kaggle Environments GUI 服務器...")
    
    # 啟動服務器
    try:
        proc = subprocess.Popen([
            sys.executable, "main.py", "http-server"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 等待服務器啟動
        time.sleep(3)
        
        # 檢查服務器是否正在運行
        if proc.poll() is None:
            print("✅ 服務器已啟動！")
            print("🌐 服務器地址: http://127.0.0.1:8000")
            return proc
        else:
            print("❌ 服務器啟動失敗")
            stdout, stderr = proc.communicate()
            print(f"錯誤: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ 啟動失敗: {e}")
        return None

def open_browser():
    """打開瀏覽器"""
    url = "http://127.0.0.1:8000/kaggle_environments/static/player.html"
    print(f"🌐 打開瀏覽器: {url}")
    try:
        webbrowser.open(url)
        print("✅ 瀏覽器已打開")
    except Exception as e:
        print(f"❌ 無法打開瀏覽器: {e}")
        print(f"請手動在瀏覽器中打開: {url}")

def main():
    print("🎯 ConnectX GUI 啟動器")
    print("=" * 50)
    
    # 檢查必要文件
    if not check_submission():
        return
    
    # 啟動服務器
    server_proc = start_server()
    if not server_proc:
        return
    
    # 打開瀏覽器
    open_browser()
    
    print()
    print("📖 使用說明:")
    print("1. 在瀏覽器中選擇環境: ConnectX")
    print("2. 添加智能體:")
    print("   - 玩家1: submission.py (你的 AI)")
    print("   - 玩家2: random (隨機對手) 或 submission.py (AI vs AI)")
    print("3. 點擊 'Run' 開始遊戲")
    print("4. 觀看 AI 對戰的動畫效果！")
    print()
    print("⌨️  按 Ctrl+C 停止服務器")
    
    try:
        # 保持服務器運行
        server_proc.wait()
    except KeyboardInterrupt:
        print("\n\n🛑 正在停止服務器...")
        server_proc.terminate()
        server_proc.wait()
        print("👋 服務器已停止")

if __name__ == "__main__":
    main()
