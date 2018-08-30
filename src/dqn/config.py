# Author: Taoz
# Date  : 8/29/2018
# Time  : 3:08 PM
# FileName: config.py


import time

GAME_NAME = 'riverraid'
# PRE_TRAIN_MODEL_FILE = None
PRE_TRAIN_MODEL_FILE = '/home/zhangtao/model_file/hello_rl/net_params_test1_114009_20180830_115830.model'
# PRE_TRAIN_MODEL_FILE = 'D:\data\\rl\\net_params_test1_175536_20180829_182429.model'
OBSERVATION_TYPE = 'image'  # image or ram
FRAME_SKIP = 4
EPOCH_NUM = 60
EPOCH_LENGTH = 20000

PHI_LENGTH = 4
CHANNEL = 3
WIDTH = 160
HEIGHT = 210

BEGIN_RANDOM_STEP = 10000

BUFFER_MAX = 20000
#BUFFER_MAX = 200000
DISCOUNT = 0.99
RANDOM_SEED = int(time.time() * 1000) % 100000000
EPSILON_MIN = 0.10
EPSILON_START = 1.0
EPSILON_DECAY = 100000

if PRE_TRAIN_MODEL_FILE is not None:
    BEGIN_RANDOM_STEP = 3000
    EPSILON_MIN = 0.10
    EPSILON_START = 0.4
    EPSILON_DECAY = 20000

TRAIN_PER_STEP = 4
UPDATE_TARGET_PER_STEP = 30000

MODEL_PATH = '/home/zhangtao/model_file/hello_rl'

BEGIN_TIME = time.strftime("%Y%m%d_%H%M%S")


print('\n\n\n\n++++++++++++++++ edited time: 1204 ++++++++++++++++++')



