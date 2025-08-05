# import sys
# from kaggle_environments import evaluate, make, utils

# out = sys.stdout
# submission = utils.read_file("submission_optimized.py")
# agent = utils.get_last_callable(submission)
# sys.stdout = out

# env = make("connectx", debug=True)
# env.run([None, agent])
# print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")


import requests
import os
token = os.getenv("TELEGRAM_BOT_TOKEN")
if not token:
    print("請設置環境變量 TELEGRAM_BOT_TOKEN")
    
BOT_TOKEN = token
CHAT_ID   = "6166024220"
def send_telegram(msg: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg}
    r = requests.post(url, data=payload)
    if not r.ok:
        print("❌ 發送失敗：", r.text)
send_telegram("🎉 訓練完畢！！")