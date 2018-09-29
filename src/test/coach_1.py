# Author: Taoz
# Date  : 9/16/2018
# Time  : 10:57 PM
# FileName: coach.py


import time
import multiprocessing as mp


def start_player2(input):
    # create player and start it.
    print(' I am a player  !!!!   !!!!!!!!!!!!!!!!!! ')


def start_coach():
    # create coach, and start it.
    print('START ...')
    # pid = os.getpid()
    # ppid = os.getppid()
    # print('++++++++++++++++++ Coach starting.... pid=[%s] ppid=[%s]' % (str(pid), str(ppid)))
    mp_ctx = mp.get_context('forkserver')
    aa = mp_ctx.Value('i', -1)
    p = mp_ctx.Process(target=start_player2, args=(aa,))
    p.start()
    # p.join()

    time.sleep(20)

    # coach = Coach(mp_ctx)
    # coach.start()
    print('Coach finish.')


if __name__ == '__main__':
    start_coach()
