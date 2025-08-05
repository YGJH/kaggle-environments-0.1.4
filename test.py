import sys
from kaggle_environments import evaluate, make, utils

out = sys.stdout
submission = utils.read_file("submission_optimized.py")
agent = utils.get_last_callable(submission)
sys.stdout = out

env = make("connectx", debug=True)
env.run([None, agent])
print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")


