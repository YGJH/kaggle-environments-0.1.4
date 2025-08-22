import sys
from kaggle_environments import evaluate, make, utils

out = sys.stdout
#submission  = utils.read_file("battle_trained_wr_0.472.py")
submission = utils.read_file("submission.py")
agent = utils.get_last_callable(submission)
# agent2 = utils.get_last_callable(submission2)
sys.stdout = out

env = make("connectx", debug=True)
env.run([agent, agent])
env.render()
print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")
