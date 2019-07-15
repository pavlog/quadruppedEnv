Best with


env = gym.make(envName)
env.env.advancedLevel = True
env.env.addObstacles = False
env.env.ActionIsAngles = True
env.env.ActionIsAnglesType = 2
env.env.ActionsIsAdditive = False
env.env.inputsSpace = 0
env.env.actionsSpace = 0
env.env.simRewardOnly = False


env.env.targetDesired_episode_from = 0
env.env.targetDesired_episode_to = 10000
env.env.targetDesired_angleFrom = np.pi/8.0
env.env.targetDesired_angleTo = np.pi/4.0

env.env.spawnYawMultiplier = 0.0
env.env.targetDesiredYawMultiplier = 0.0

env.env.analyticReward = True
env.env.analyticRewardType = 1
