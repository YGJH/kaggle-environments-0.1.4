import os
import re
import time
from datetime import datetime
from typing import Optional, Tuple
import glob
VIDEOS_DIR = "videos"
FILENAME_RE = re.compile(r"^episode_(\d+)_(\d{8}_\d{6})\.mp4$")
FINAL_RE = re.compile(r"^final_(\d{8})_(\d{6})\.pt$")
TS_TAIL_RE = re.compile(r"_(\d{8})_(\d{6})\.pt$")



def parse_filename(fname: str) -> Optional[Tuple[int, datetime]]:
    """Parse episode number and timestamp from filename.
    Expected: episode_{episode}_{YYYYMMDD_HHMMSS}.mp4
    """
    m = FILENAME_RE.match(fname)
    if not m:
        return None
    ep = int(m.group(1))
    ts_str = m.group(2)
    try:
        ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    return ep, ts


def find_latest_episode_file(files):
    candidates = []
    for f in files:
        parsed = parse_filename(f)
        if parsed is not None:
            ep, ts = parsed
            candidates.append((ep, ts, f))
    if not candidates:
        return None
    # Pick the one with the largest (episode, timestamp)
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1]  # (ep, ts, filename)

# def cleanup_checkpoints(keep_path: str):
#     ckpt_dir = os.path.dirname(keep_path) or "."
#     if not os.path.isdir(ckpt_dir):
#         print(f"⚠️  checkpoints 目錄不存在: {ckpt_dir}")
#         return
#     files = glob.glob(os.path.join(ckpt_dir, "*.pt"))
#     removed = 0
#     for p in files:
#         if os.path.abspath(p) == os.path.abspath(keep_path):
#             continue
#         try:
#             os.remove(p)
#             removed += 1
#         except Exception as e:
#             print(f"⚠️  刪除 {p} 失敗: {e}")
#     print(f"🧹 已清理 checkpoints：保留 {os.path.basename(keep_path)}，刪除 {removed} 個檔案")
def choose_latest_checkpoint_by_name(files):
    latest = None
    latest_time = None
    for p in files:
        fname = os.path.basename(p)
        ts = parse_ts_from_name(fname)
        try:
            t = ts.timestamp() if ts is not None else os.path.getmtime(p)
        except OSError:
            t = 0
        if latest is None or t > latest_time:
            latest = p
            latest_time = t
    return latest

def parse_ts_from_name(fname: str):
    m = FINAL_RE.match(fname)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except Exception:
            return None
    m = TS_TAIL_RE.search(fname)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except Exception:
            return None
    return None
def main():
    while True:

        import glob
        try:
            keep = choose_latest_checkpoint_by_name(glob.glob(os.path.join('checkpoints', '*.pt')))
            print(f'keep: {keep}')
            if keep:
                pass
                # cleanup_checkpoints(keep)
            else:
                print("⚠️  無可清理的 checkpoints。")
        except Exception as e:
            print(f"⚠️  清理 checkpoints 失敗: {e}")

        files = [f for f in os.listdir(VIDEOS_DIR) if f.startswith("episode_")]
        if not files:
            print("沒有找到任何 episode 檔案")
            time.sleep(60 * 60 * 3)
            continue

        latest = find_latest_episode_file(files)
        if latest is None:
            print("沒有找到符合命名格式的 MP4 檔案 (episode_{episode}_{ts}.mp4)")
            time.sleep(60 * 60 * 3)
            continue

        latest_ep, latest_ts, latest_file = latest
        print(f"保留最新的文件: {latest_file} (episode={latest_ep}, ts={latest_ts})")

        files_removed = 0
        for filename in files:
            # 僅刪除非最新的 mp4；保留最新那支
            if filename == latest_file:
                continue
            # 刪除所有其他 episode_*.mp4（不等於最新）
            if FILENAME_RE.match(filename):
                file_path = os.path.join(VIDEOS_DIR, filename)
                try:
                    os.remove(file_path)
                    print(f"刪除文件: {filename}")
                    files_removed += 1
                except Exception as e:
                    print(f"刪除文件 {filename} 時出錯: {e}")

        print(f"總共刪除了 {files_removed} 個文件")
        # 每3小時清理一次
        time.sleep(60 * 30)


if __name__ == "__main__":
    main()
