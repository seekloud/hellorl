# Author: Taoz
# Date  : 8/29/2018
# Time  : 5:28 PM
# FileName: runner.py


# import src.dqn.experiment as runner
from src.dqn2.coach import start_coach
if __name__ == '__main__':
    # runner.train()
    # runner.test(render=True)
    start_coach()
    # print('os.name:', os.name)
