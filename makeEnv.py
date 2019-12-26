import gym
import numpy as np
import glm

from .wrappers import MaxAndSkipEnv
from .wrappers import FrameStack

frames_stack = 2
max_and_skip = 8



def make_env_with_best_settings(envName):
  env = gym.make(envName)
  env.env.advancedLevel = True
  env.env.advancedLevelRandom = False
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

  env.env.progressDirChassisMultiplier = 0.6
  env.env.progressDirTargetMultiplier = 0.6
  env.env.progressMultiplier = 1.0

  env.env.aliveMultiplier = 1.0
  env.env.aliveMultiplierClampMax = 0.0

  env.env.yawRewardMultiplier = 0.0

  env.env.holdingTorqueMultiplier=0.0

  env.env.energyCostMultiplier = 0.0

  env.env.useLegsHeightReward = True

  env.env.angleLimitszFromMp2ToP2 = False

  env.env.ground_pos_random_x = 0.5
  env.env.ground_pos_random_y = 0.5

  env = MaxAndSkipEnv(env,max_and_skip,False)
  env = FrameStack(env,frames_stack,True)
  return env


def make_env_for_balance(envName):
  env = gym.make(envName)
  #env.env.spec.max_episode_steps = 2000
  env.env.advancedLevel = True
  env.env.advancedLevelRandom = False
  env.env.randomInitDir = False
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
  env.env.targetDesired_angleFrom = glm.radians(0.0)
  env.env.targetDesired_angleTo = glm.radians(0.0)

  env.env.spawnYawMultiplier = 1.0
  env.env.targetDesiredYawMultiplier = 1.0
  
  env.env.analyticReward = False
  env.env.analyticRewardType = 1

  env.env.anglesRewardMultiplier = 0.0

  env.env.check90Angles = False

  env.env.goalRandomTargetDirClamp = glm.radians(0.0)
  env.env.goalRandomChassisDirClamp = glm.radians(0.0)

  env.env.randomizeDefaultAngles = glm.radians(0.0)

  env.env.goalInitRandomAngle = glm.radians(0.0)
  env.env.goalInitRandomChassisAngle = glm.radians(90.0)

  env.env.progressDirChassisMultiplier = 3.0
  env.env.progressDirTargetMultiplier = 0.0
  env.env.progressMultiplier = 0.0

  env.env.aliveMultiplier = 0.3
  env.env.aliveMultiplierClampMax = 1.0

  env.env.yawRewardMultiplier = 0.0

  env.env.holdingTorqueMultiplier=0.0

  env.env.energyCostMultiplier = 0.0

  env.env.useLegsHeightReward = False
  env.env.useLegsHeightRewardAdditive = True
  env.env.legsHeightMultiplier = 0.3

  env.env.stopIfNoMovement = False
  
  env.env.spawn_rot_x = glm.radians(5.0)
  env.env.spawn_rot_y = glm.radians(5.0)

  env.env.angleLimitszFromMp2ToP2 = False

  env.env.changeGoalEvery = 1000
  env.env.useZeroGoalLogic = True

  env.env.ground_pos_random_x = 0.7
  env.env.ground_pos_random_y = 0.7
  
  env = MaxAndSkipEnv(env,max_and_skip,False)
  env = FrameStack(env,frames_stack,True)

  
  return env
