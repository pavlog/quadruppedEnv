from gym.envs.registration import register
#from gym.scoreboard.registration import add_task, add_group

import os
import os.path as osp
import subprocess

register(
    id='QuadruppedWalk-v1',
    entry_point='quadruppedEnv:QuadruppedWalker',
    max_episode_steps= 10000, #10000,
    reward_threshold=2500.0,
    tags={ "pg_complexity": 8*1000000 },
    )

register(
    id='QAnt-v0',
    entry_point='quadruppedEnv:QAntOrg',
    max_episode_steps=1000,
    reward_threshold=2500.0,
    tags={ "pg_complexity": 8*1000000 },
    )

register(
    id='QAnt-v1',
    entry_point='quadruppedEnv:QAnt',
    max_episode_steps=1000,
    reward_threshold=2500.0,
    tags={ "pg_complexity": 8*1000000 },
    )

from quadruppedEnv.robot import QuadruppedWalker
from quadruppedEnv.qant import QAnt
from quadruppedEnv.qant import QAntOrg

