# Author: Taoz
# Date  : 8/29/2018
# Time  : 5:28 PM
# FileName: runner.py


import multiprocessing as mp
# import src.dqn.experiment as runner
from src.dqn2.coach import start_coach
if __name__ == '__main__':
    mp.freeze_support()
    # runner.train()
    # runner.test(render=True)
    start_coach()
    # test_process()
    # print('os.name:', os.name)
