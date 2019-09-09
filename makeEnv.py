import gym
import numpy as np
import glm

from .wrappers import MaxAndSkipEnv
from .wrappers import FrameStack

frames_stack = 2
max_and_skip = 8



def make_env_with_best_settings(envName, advancedLevel = True):
  env = gym.make(envName)
  env.env.advancedLevel = advancedLevel
  env.env.advancedLevelRandom = advancedLevel
  env.env.randomInitDir = True
  env.env.addObstacles = False
  env.env.ActionIsAngles = True
  env.env.ActionIsAnglesType = 2
  env.env.ActionsIsAdditive = False
  env.env.inputsSpace = 0
  env.env.actionsSpace = 0
  env.env.simRewardOnly = False

  env.env.maxAnglesPerSec = 90.0


  env.env.targetDesired_episode_from = 0
  env.env.targetDesired_episode_to = 10000
  env.env.targetDesired_angleFrom = np.pi/2.0
  env.env.targetDesired_angleTo = np.pi/2.0

  env.env.spawnYawMultiplier = 0.0
  env.env.targetDesiredYawMultiplier = 0.0
  
  env.env.analyticReward = False
  env.env.analyticRewardType = 1

  env.env.anglesRewardMultiplier = 0.0

  env.env.check90Angles = False

  env.env.goalRandomTargetDirClamp = glm.radians(40.0)
  env.env.goalRandomChassisDirClamp = glm.radians(40.0)

  env.env.progressDirChassisMultiplier = 0.5
  env.env.progressDirTargetMultiplier = 0.5

  env.env.aliveMultiplier = 1.0
  env.env.aliveMultiplierClampMax = 0.0

  env.env.yawRewardMultiplier = 0.0

  env.env.holdingTorqueMultiplier=0.0

  env.env.energyCostMultiplier = 0.0

  env.env.useLegsHeightReward = True


  env = MaxAndSkipEnv(env,max_and_skip,False)
  env = FrameStack(env,frames_stack,True)
  return env
