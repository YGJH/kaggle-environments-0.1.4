# import sys
# from kaggle_environments import evaluate, make, utils

# out = sys.stdout
# submission = utils.read_file("submission.py")
# agent = utils.get_last_callable(submission)
# sys.stdout = out

# env = make("connectx", debug=True)
# env.run([agent, agent])
# env.render()
# print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")


from matplotlib.animation import FFMpegWriter
print(FFMpegWriter.isAvailable())

import shutil
print(shutil.which('ffmpeg'))
