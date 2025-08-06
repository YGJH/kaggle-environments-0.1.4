import os
import shutil
import time

while True:
    best_models = [f for f in os.listdir('game_videos') if f.startswith('episode_')]
    
    if not best_models:
        print("沒有找到任何 episode 文件")
        time.sleep(60*60*3)
        continue
        
    best_model_path = sorted(best_models)[-1]
    print(f"保留最新的文件: {best_model_path}")

    files_removed = 0
    for filename in os.listdir('game_videos'):
        if filename.startswith('episode_') and filename != best_model_path:
            file_path = os.path.join('game_videos', filename)
            try:
                os.remove(file_path)
                print(f"刪除文件: {filename}")
                files_removed += 1
            except Exception as e:
                print(f"刪除文件 {filename} 時出錯: {e}")
    
    print(f"總共刪除了 {files_removed} 個文件")
    time.sleep(60*60*3)