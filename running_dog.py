import subprocess
import os

while True:
    try:
        cmd = [
            'uv',
            'run',
            'train_connectx_rl_robust.py',
            ]
        subprocess.run(cmd , check=True)
        print("\033[92mTraining completed successfully.\033[0m")
    except Exception as e:
        print(f"\033[91mAn error occurred: {e}\033[0m")
        print("Restarting training...")
