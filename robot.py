
import numpy as np
import gym
from gym import wrappers
import argparse
import pprint as pp
import roboschool
import math
from gym.envs.registration import register
from roboschool.scene_abstract import cpp_household
from roboschool.gym_forward_walker import RoboschoolForwardWalker
from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv
from roboschool.gym_urdf_robot_env import RoboschoolUrdfEnv
from roboschool.scene_stadium import SinglePlayerStadiumScene
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import os, sys
from collections import deque
from numpy import arccos, array, dot, pi, cross
from numpy.linalg import det, norm
from importlib import import_module
import time
import random
import glm
import xml.etree.ElementTree as ET
from roboschool.scene_abstract import Scene, cpp_household

episode_index = 0

try:
    from quadruppedEnv import settings
    settings.robotLibAvailable = 1
    robotlib = os.path.realpath("/Users/pavlogryb/Dropbox/Projects/Robot/Robot/pyRobotLib")
    sys.path.insert(0, robotlib)
    import pyRobotLib
    #print (pyRobotLib.greet())
except ImportError:
    settings.robotLibAvailable = 0
    print("robot lib not fouund")
    pass

class DummyJoint(object):
    pass

class RoboschoolForwardWalkerMujocoXML(RoboschoolForwardWalker, RoboschoolUrdfEnv):
    def __init__(self, fn, robot_name):

        self.advancedLevel = True
        self.addObstacles = False
        self.analyticReward = False
        self.analyticRewardType = 0
        self.simRewardOnly = False
        self.spawnYawMultiplier = 0.0
        self.targetDesiredYawMultiplier = 0.0
        self.targetDesired_episode_from = 0
        self.targetDesired_episode_to = 1000
        self.targetDesired_angleFrom = np.pi/8.0
        self.targetDesired_angleTo = np.pi/4.0


        self.footLinks =  [["fl1","fl2","fl3"],["fr1","fr2","fr3"],["bl1","bl2","bl3"],["br1","br2","br3"]]

        urdfFilename = os.path.join(os.path.dirname(__file__), "models_robot", fn)

        self.state_index_legs = 9
        self.state_index_legs_count = 12

        if hasattr(settings,"jointVelocities")==False:
            settings.jointVelocities = 0.0
        
        self.velObservations = 0
        sensorsObservations = 4 #4 legs from 0 or 1
        rotationObservation = 3 #roll pith yaw
        sinCosToTarget = 2 # sin cos from angle to target
        servoObservation = 12 # rel angles
        jointVelocities = 0
        if settings.jointVelocities>0.0:
            jointVelocities = self.state_index_legs_count
        self.NeedServoObservation = True #True
        numObservation = sensorsObservations+rotationObservation+sinCosToTarget*2+self.velObservations+jointVelocities
        if self.NeedServoObservation:
            numObservation +=servoObservation

        self.cycleGenerators = False
        if self.cycleGenerators:
            numObservation+=3

        if hasattr(settings,"history1Len")==False:
            settings.history1Len = 0.0
        if hasattr(settings,"history2Len")==False:
            settings.history2Len = 0.0
        if hasattr(settings,"history3Len")==False:
            settings.history3Len = 0.0


        baseObservations = numObservation
        if settings.history1Len>0.0:
            numObservation += baseObservations
        if settings.history2Len>0.0:
            numObservation += baseObservations
        if settings.history3Len>0.0:
            numObservation += baseObservations
        self.servo_kp = 0.3
        self.servo_kd = 0.9
        self.servo_maxTorque = 2.0
        self.time = 0.0
        self.physics_step_per_second = 240.0 #120.0 #240.0 #240.0
        self.frame_skip = 1
        self.physics_time_step = (1.0/self.physics_step_per_second)
        self.max_servo_speed = math.radians(180.0)
        self.AdditiveAnglesClip = self.max_servo_speed*self.physics_time_step
        self.walk_target_x = 1000.0 # 1km  meters
        self.walk_target_z = 0.15

        self.debugStats = 0
        self.movement_dir = [1,0,0]
        self.allowRecover = False

        self.jointsLocalPos = {}
        tree = ET.parse(urdfFilename)
        root = tree.getroot()
        for robot in root.iter('robot'):
            for joint in robot.iter('joint'):
                origin = joint.find("origin")
                posStr = origin.attrib["xyz"].split(" ")
                pos = [float(posStr[0]),float(posStr[1]),float(posStr[2])]
                self.jointsLocalPos[joint.attrib["name"]] = pos
                
        self.link1 = self.jointsLocalPos["fl3"]
        self.link2 = self.jointsLocalPos["fl4"]

        self.extLenX = 0.15 # 15 cm
        self.extLenY = 0.15 # 15 cm
        self.minZLeg = -0.16
        self.maxZLeg = -0.06

        self.minArea = [-self.extLenX,-self.extLenY,self.minZLeg] # 20 cm away from chassis
        self.maxArea = [+self.extLenX,+self.extLenY,self.maxZLeg] # 2 cm from chassis 

        self.flMinArea = glm.vec3(self.minArea)
        self.flMaxArea = glm.vec3(self.maxArea)
        self.frMinArea = glm.vec3(self.minArea)
        self.frMaxArea = glm.vec3(self.maxArea)

        self.blMinArea = glm.vec3(self.minArea)
        self.blMaxArea = glm.vec3(self.maxArea)
        self.brMinArea = glm.vec3(self.minArea)
        self.brMaxArea = glm.vec3(self.maxArea)

        self.areaSize = [self.maxArea[0]-self.minArea[0],self.maxArea[1]-self.minArea[1],self.maxArea[2]-self.minArea[2]]

        self.chassisSpaceX = 0.30
        self.chassisSpaceY = 0.20
        self.chassisSpaceMin = [-self.chassisSpaceX,-self.chassisSpaceY,-0.2]
        self.chassisSpaceMax = [+self.chassisSpaceX,+self.chassisSpaceY,-0.02]
        rotSize = 0.12+0.1
        self.chassisRotSpaceMin = [-rotSize,-rotSize,-rotSize]
        self.chassisRotSpaceMax = [+rotSize,+rotSize,+rotSize]

        self.len1 = abs(self.link1[2])
        self.len2 = abs(self.link2[2])
        
        self.ActionIsAngles = True
        self.ActionIsAnglesType = 1 # 0 - pi/2 1 pi 2 limits
        self.ActionsIsAdditive = False
        self.AdditiveAnglesType = 2 # 2 best
        self.actionsMult = 1.0
        self.inputsSpace = 0 # 0 is local 1 is chassis 2 is world 3 world with yaw
        self.actionsSpace = 5 # 0 local, 1 chassis, 2 cycle, 3 is world now yaw 4 world with yaw 5 local cycles, 6 - local with chassis world dirs
        self.inputsIsIKTargets = False
        self.moveChassisActions = False

        '''
        # ik style
        self.ActionIsAngles = False
        self.ActionsIsAdditive = False
        self.AdditiveAnglesType = 2 # 2 best
        self.inputsSpace = 0 # 0 is local 1 is chassis 2 is world 3 world with yaw
        self.actionsSpace = 5 # 0 local, 1 chassis, 2 cycle, 3 is world now yaw 4 world with yaw 5 local cycles, 6 - local with chassis world dirs
        self.inputsIsIKTargets = True
        self.moveChassisActions = True
        '''

        self.perLegSpace = True
        if self.perLegSpace:
            self.extLenYOut = 0.1 # 10  cm
            self.extLenYIn = 0.0 # 2 cm
            self.flMinArea = glm.vec3(-self.extLenX,-self.extLenYIn,self.minZLeg)
            self.flMaxArea = glm.vec3(self.extLenX,self.extLenYOut,self.maxZLeg)
            self.blMinArea = glm.vec3(-self.extLenX,-self.extLenYIn,self.minZLeg)
            self.blMaxArea = glm.vec3(self.extLenX,self.extLenYOut,self.maxZLeg)

            self.frMinArea = glm.vec3(-self.extLenX,-self.extLenYOut,self.minZLeg)
            self.frMaxArea = glm.vec3(self.extLenX,self.extLenYIn,self.maxZLeg)
            self.brMinArea = glm.vec3(-self.extLenX,-self.extLenYOut,self.minZLeg)
            self.brMaxArea = glm.vec3(self.extLenX,self.extLenYIn,self.maxZLeg)


        if settings.robotLibAvailable:
            self.robotLib = pyRobotLib.Robot()
            self.robotLib.load("/Users/pavlogryb/Dropbox/Projects/Robot/Robot/Robot/")
            #print(os.getpid())
        
        action_dim = 12
        if self.moveChassisActions and not self.ActionIsAngles:
            action_dim+=3

        RoboschoolUrdfEnv.__init__(self, urdfFilename,robot_name, action_dim, numObservation,fixed_base=False,self_collision=False)
        RoboschoolForwardWalker.__init__(self, 0.1)


    def create_single_player_scene(self):
        return QSinglePlayerStadiumScene(gravity=9.8, timestep=self.physics_time_step, frame_skip=self.frame_skip)

    def calc_state(self):
        #cp.enable()
        robot_state = self.calc_state_single()
        assert( np.isfinite(robot_state).all() )
        self.state_history.append(robot_state)
        if settings.jointVelocities:
            timeDelay = 0.5
            indexForVel = timeDelay*self.physics_step_per_second
            index  = int(len(self.state_history)-1-indexForVel)
            if index<0:
                index = int(0)
            for x in range(self.state_index_legs, self.state_index_legs+self.state_index_legs_count):
                legVel = (robot_state[x]-self.state_history[index][x])/timeDelay
                robot_state = np.append(robot_state,[legVel])
        if settings.history1Len>0.0:
            self.history1.append(robot_state)
        if settings.history2Len>0.0:
            self.history2.append(robot_state)
        if settings.history3Len>0.0:
            self.history3.append(robot_state);
        self.last_state = robot_state
        if settings.history1Len>0.0:
            robot_state= np.append(robot_state,self.history1[0])
        if settings.history2Len>0.0:
            robot_state= np.append(robot_state,self.history2[0])
        if settings.history3Len>0.0:
            robot_state= np.append(robot_state,self.history3[0])
        #cp.disable()
        #cp.print_stats()
        return robot_state

    def getLastSingleState(self):
        return self.last_single_state


    def __getstate__(self):
        return { "dummy": "dummy"}

    def rotation(self,orig,dest):
        identityQuat = glm.quat(1.0,0.0,0.0,0.0)
        epsilon = 0.00001

        cosTheta = glm.dot(orig, dest);

        if cosTheta >= 1.0 - epsilon:
            #// orig and dest point in the same direction
            return identityQuat

        if cosTheta < -1.0 + epsilon:
            '''
            // special case when vectors in opposite directions :
            // there is no "ideal" rotation axis
            // So guess one; any will do as long as it's perpendicular to start
            // This implementation favors a rotation around the Up axis (Y),
            // since it's often what you want to do.
            '''
            rotationAxis = glm.cross(glm.vec3(0.0, 0.0, 1.0), orig);
            if glm.length(rotationAxis) < epsilon: # // bad luck, they were parallel, try again!
                rotationAxis = glm.cross(glm.vec3(1.0, 0.0, 0.0), orig)

            rotationAxis = glm.normalize(rotationAxis)
            return glm.angleAxis(glm.pi(), rotationAxis)

        #// Implementation from Stan Melax's Game Programming Gems 1 article
        rotationAxis = glm.cross(orig, dest);

        s = math.sqrt((1.0 + cosTheta) * 2.0);
        invs = 1.0 / s;

        return glm.quat(s * 0.5,rotationAxis.x * invs,rotationAxis.y * invs,rotationAxis.z * invs)



    def solve_square_equation(self,a,b,c):
        d = b * b - 4 * a * c; #// disc
        if d < 0:
            return 0,0.0,0.0

        q = 0.0
        if b >= 0:
            q = (-b - math.sqrt(d)) / 2.0
        else:
            q = (-b + math.sqrt(d)) / 2.0

        x1 = q / a
        x2 = c / q

        if d == 0.0:
            return 1,x1,x2
        return 2,x1,x2

    def angleBetween(self,a3,b3):
        a = glm.vec2(a3)
        b = glm.vec2(b3)
        p = glm.vec2(-b[1], b[0])
        b_coord = glm.dot(a, b)
        p_coord = glm.dot(a, p)
        return math.atan2(p_coord, b_coord)

    def normAngle(self,angle):
        #// 180-270
        if angle >= glm.pi() and angle <= glm.three_over_two_pi():
            return angle - glm.two_pi()
        #// 270-360
        if angle >= glm.three_over_two_pi() and angle <= glm.two_pi():
            return angle - glm.two_pi()
        return angle

    def solveIK(self,coord, Linkage_1, Linkage_2):
        l2 = Linkage_1 * Linkage_1;
        L2 = Linkage_2 * Linkage_2;

        l = Linkage_1
        L = Linkage_2

        x = coord[0];
        y = coord[1];

        SCARA_C2 = ((x * x) + (y * y) - l2 - L2) / (2.0 * l * L)

        bb = 1.0 - (SCARA_C2 * SCARA_C2)
        if bb < 0.0:
            norm = glm.normalize(glm.vec3(coord[0], coord[1], 0.0))
            q1 = math.atan2(norm[1], norm[0])
            #q1 = math.atan2(y, x)
            return q1, 0.0, 0.0
  
        SCARA_S2 = math.sqrt(bb)

        SCARA_K1 = l + L * SCARA_C2
        SCARA_K2 = L * SCARA_S2

        SCARA_theta = (math.atan2(x, y) - math.atan2(SCARA_K1, SCARA_K2)) * -1.0
        SCARA_psi = math.atan2(SCARA_S2, SCARA_C2)

        q11 = SCARA_theta
        q22a = SCARA_psi

        q11 = self.normAngle(q11)
        q22a = self.normAngle(q22a)
        return q11, q22a, 1.0

    def angleDiff(self,a,b):
        dif = math.fmod(b - a + glm.pi(), 2.0 * glm.pi())
        if dif < 0:
            dif += 2.0 *glm.pi()
        return dif - glm.pi()

    def solveLeg(self,upperLegOffset,l1,l2,target):
        upperLeg = glm.vec3(upperLegOffset[1], upperLegOffset[2], 0.0);
        upperLegNorm = glm.normalize(upperLeg);
        '''
        //
        //                 0
        //           a    /
        //              /
        //         ulo/  cosa
        //            |
        //            |  L
        //         b  |
        //            |
        //            t
        //
        // c = |0-t|
        // b = |ulo-t| , is unknown
        '''
        cosa = upperLegNorm[1]
        upperLegLen = glm.length(upperLeg)

        targetYZ = glm.vec3(target[1], target[2], 0.0);
        targetYZLen = glm.length(targetYZ);
        '''
        //
        // c^2 = a^2 + b^2 - 2*a*b*cosa
        //
        //  b^2 - b*(2*a*cosa) + (a^2-c^2)
        //
        '''
        ka = 1.0;
        kb = -2.0 * upperLegLen * cosa;
        kc = upperLegLen * upperLegLen - targetYZLen * targetYZLen;
        numRoots, x1, x2 = self.solve_square_equation(ka, kb, kc)

        root = max(x1, x2)

        targetYZOrg = glm.vec3(upperLegOffset[1], upperLegOffset[2] - root, 0.0)
        targetYZOrgNorm = glm.normalize(targetYZOrg)
        targetYZCurNorm = glm.normalize(targetYZ)

        alpha = self.angleBetween(targetYZCurNorm, targetYZOrgNorm)
        targetYZCurNorm3D = glm.vec3(0.0, targetYZCurNorm[0], targetYZCurNorm[1])
        targetYZOrgNorm3D = glm.vec3(0.0, targetYZOrgNorm[0], targetYZOrgNorm[1])
        rot = self.rotation(targetYZOrgNorm3D, targetYZCurNorm3D)

        newTarget = glm.mat3_cast(rot)*glm.vec3(target[0], target[2], target[1])

        target2d = newTarget - glm.vec3(upperLegOffset[0], upperLegOffset[2], upperLegOffset[1])
        rotLeg = self.solveIK(target2d, l1, l2)

        angleOffset = glm.vec3(0.0, glm.radians(-90.0), 0.0)

        angle1 = self.angleDiff(angleOffset[0], alpha)
        angle2 =  -1.0*self.angleDiff(angleOffset[1], rotLeg[0])
        angle3 = -1.0*self.angleDiff(angleOffset[2], rotLeg[1])
        jangles = glm.vec4(angle1, angle2,angle3, rotLeg[2])

        return jangles

    def getJ1TM(self,jointName):
        resultTm = glm.mat4x4()
        #resultTm = glm.rotate(resultTm,angles[0], glm.vec3([1.0,0.0,0.0]))
        resultTm = glm.translate(resultTm,self.jointsLocalPos[jointName])
        return resultTm

    def getLocalXYZFromAngles(self,angles,len1,len2,joint2Offs):
        #state_start = time.time()
        resultTm = glm.mat4x4()
        resultTm = glm.rotate(resultTm,angles[0], glm.vec3([1.0,0.0,0.0]))
        resultTm = glm.translate(resultTm,joint2Offs)
        resultTm = glm.rotate(resultTm,angles[1], glm.vec3([0.0,1.0,0.0]))
        resultTm = glm.translate(resultTm,len1)
        resultTm = glm.rotate(resultTm,angles[2], glm.vec3([0.0,1.0,0.0]))
        resultTm = glm.translate(resultTm,len2)
        point = glm.vec3(glm.column(resultTm,3))
        #print("tm",(time.time()-state_start)*1000.0)
        return point

    def getLeg4XYZFromAngles(self,angles,len1,len2,joint1Offs,joint2Offs,chassisTM):
        #state_start = time.time()
        resultTm = glm.mat4x4()
        resultTm = resultTm * chassisTM
        resultTm = glm.translate(resultTm,joint1Offs)
        resultTm = glm.rotate(resultTm,angles[0], glm.vec3([1.0,0.0,0.0]))
        resultTm = glm.translate(resultTm,joint2Offs)
        resultTm = glm.rotate(resultTm,angles[1], glm.vec3([0.0,1.0,0.0]))
        resultTm = glm.translate(resultTm,len1)
        resultTm = glm.rotate(resultTm,angles[2], glm.vec3([0.0,1.0,0.0]))
        resultTm = glm.translate(resultTm,len2)
        point = glm.vec3(glm.column(resultTm,3))
        #print("tm",(time.time()-state_start)*1000.0)
        return point

    def getLeg3XYZFromAngles(self,angles,len1,len2,joint1Offs,joint2Offs,chassisTM):
        #state_start = time.time()
        resultTm = glm.mat4x4()
        resultTm = resultTm * chassisTM
        resultTm = glm.translate(resultTm,joint1Offs)
        resultTm = glm.rotate(resultTm,angles[0], glm.vec3([1.0,0.0,0.0]))
        resultTm = glm.translate(resultTm,joint2Offs)
        resultTm = glm.rotate(resultTm,angles[1], glm.vec3([0.0,1.0,0.0]))
        resultTm = glm.translate(resultTm,len1)
        point = glm.vec3(glm.column(resultTm,3))
        #print("tm",(time.time()-state_start)*1000.0)
        return point

    def getLeg2XYZFromAngles(self,angles,len1,len2,joint1Offs,joint2Offs,chassisTM):
        #state_start = time.time()
        resultTm = glm.mat4x4()
        resultTm = resultTm * chassisTM
        resultTm = glm.translate(resultTm,joint1Offs)
        resultTm = glm.rotate(resultTm,angles[0], glm.vec3([1.0,0.0,0.0]))
        resultTm = glm.translate(resultTm,joint2Offs)
        point = glm.vec3(glm.column(resultTm,3))
        #print("tm",(time.time()-state_start)*1000.0)
        return point

    def getLeg1XYZFromAngles(self,angles,len1,len2,joint1Offs,chassisTM):
        #state_start = time.time()
        resultTm = glm.mat4x4()
        resultTm = resultTm * chassisTM
        resultTm = glm.translate(resultTm,joint1Offs)
        point = glm.vec3(glm.column(resultTm,3))
        #print("tm",(time.time()-state_start)*1000.0)
        return point

    def rangeNormalize(self,value,minArea,maxArea):
        areaSize = maxArea-minArea
        clipped = np.clip(value,minArea,maxArea)
        x = (clipped[0]-minArea[0])/(areaSize[0])*2.0-1.0
        y = (clipped[1]-minArea[1])/(areaSize[1])*2.0-1.0
        z = (clipped[2]-minArea[2])/(areaSize[2])*2.0-1.0
        return [x,y,z]

    def chassisNormalize(self,value):
        clipped = np.clip(value,self.chassisSpaceMin,self.chassisSpaceMax)
        x = (clipped[0]-self.chassisSpaceMin[0])/(self.chassisSpaceMax[0]-self.chassisSpaceMin[0])*2.0-1.0
        y = (clipped[1]-self.chassisSpaceMin[1])/(self.chassisSpaceMax[1]-self.chassisSpaceMin[1])*2.0-1.0
        z = (clipped[2]-self.chassisSpaceMin[2])/(self.chassisSpaceMax[2]-self.chassisSpaceMin[2])*2.0-1.0
        return [x,y,z]

    def chassisRotNormalize(self,value):
        clipped = np.clip(value,self.chassisRotSpaceMin,self.chassisRotSpaceMax)
        x = (clipped[0]-self.chassisRotSpaceMin[0])/(self.chassisRotSpaceMax[0]-self.chassisRotSpaceMin[0])*2.0-1.0
        y = (clipped[1]-self.chassisRotSpaceMin[1])/(self.chassisRotSpaceMax[1]-self.chassisRotSpaceMin[1])*2.0-1.0
        z = (clipped[2]-self.chassisRotSpaceMin[2])/(self.chassisRotSpaceMax[2]-self.chassisRotSpaceMin[2])*2.0-1.0
        return [x,y,z]

    def syncLocalFromXYZ(self,fl,fr,bl,br,inputsSpace):
        self.flPosLocal = self.getLocalXYZFromAngles(fl,self.link1,self.link2,self.jointsLocalPos["fl2"])
        self.jdict["fl1"].localPos = self.flPosLocal
        if inputsSpace==1:
            self.flPos = self.flPosLocal+self.jointsLocalPos["fl1"]
            self.flPosN = np.float32(self.chassisNormalize(self.flPos))
        elif inputsSpace==2:
            self.flPos = self.flPosLocal+self.jointsLocalPos["fl1"]
            self.flPos = glm.vec3(self.bodyRotNoYaw * glm.vec4(self.flPos,0.0))
            self.flPosN = np.float32(self.chassisRotNormalize(self.flPos))
        elif inputsSpace==3:
            self.flPos = self.flPosLocal+self.jointsLocalPos["fl1"]
            self.flPos = glm.vec3(self.bodyRot * glm.vec4(self.flPos,0.0))
            self.flPosN = np.float32(self.chassisRotNormalize(self.flPos))
        else:
            self.flPosN = np.float32(self.rangeNormalize(self.flPosLocal,self.flMinArea,self.flMaxArea))

        self.frPosLocal = self.getLocalXYZFromAngles(fr,self.link1,self.link2,self.jointsLocalPos["fr2"])
        self.jdict["fr1"].localPos = self.frPosLocal
        if inputsSpace==1:
            self.frPos = self.frPosLocal+self.jointsLocalPos["fr1"]
            self.frPosN = np.float32(self.chassisNormalize(self.frPos))
        elif inputsSpace==2:
            self.frPos = self.frPosLocal+self.jointsLocalPos["fr1"]
            self.frPos = glm.vec3(self.bodyRotNoYaw * glm.vec4(self.frPos,0.0))
            self.frPosN = np.float32(self.chassisRotNormalize(self.frPos))
        elif inputsSpace==3:
            self.frPos = self.frPosLocal+self.jointsLocalPos["fr1"]
            self.frPos = glm.vec3(self.bodyRot * glm.vec4(self.frPos,0.0))
            self.frPosN = np.float32(self.chassisRotNormalize(self.frPos))
        else:
            self.frPosN = np.float32(self.rangeNormalize(self.frPosLocal,self.frMinArea,self.frMaxArea))

        self.blPosLocal = self.getLocalXYZFromAngles(bl,self.link1,self.link2,self.jointsLocalPos["bl2"])
        self.jdict["bl1"].localPos = self.blPosLocal
        if inputsSpace==1:
            self.blPos = self.blPosLocal+self.jointsLocalPos["bl1"]
            self.blPosN = np.float32(self.chassisNormalize(self.blPos))
        elif inputsSpace==2:
            self.blPos = self.blPosLocal+self.jointsLocalPos["bl1"]
            self.blPos = glm.vec3(self.bodyRotNoYaw * glm.vec4(self.blPos,0.0))
            self.blPosN = np.float32(self.chassisRotNormalize(self.blPos))
        elif inputsSpace==3:
            self.blPos = self.blPosLocal+self.jointsLocalPos["bl1"]
            self.blPos = glm.vec3(self.bodyRot * glm.vec4(self.blPos,0.0))
            self.blPosN = np.float32(self.chassisRotNormalize(self.blPos))
        else:
            self.blPosN = np.float32(self.rangeNormalize(self.blPosLocal,self.blMinArea,self.blMaxArea))

        self.brPosLocal = self.getLocalXYZFromAngles(br,self.link1,self.link2,self.jointsLocalPos["br2"])
        self.jdict["br1"].localPos = self.brPosLocal
        if inputsSpace==1:
            self.brPos = self.brPosLocal+self.jointsLocalPos["br1"]
            self.brPosN = np.float32(self.chassisNormalize(self.brPos))
        elif inputsSpace==2:
            self.brPos = self.brPosLocal+self.jointsLocalPos["br1"]
            self.brPos = glm.vec3(self.bodyRotNoYaw * glm.vec4(self.brPos,0.0))
            self.brPosN = np.float32(self.chassisRotNormalize(self.brPos))
        elif inputsSpace==3:
            self.brPos = self.brPosLocal+self.jointsLocalPos["br1"]
            self.brPos = glm.vec3(self.bodyRot * glm.vec4(self.brPos,0.0))
            self.brPosN = np.float32(self.chassisRotNormalize(self.brPos))
        else:
            self.brPosN = np.float32(self.rangeNormalize(self.brPosLocal,self.brMinArea,self.brMaxArea))


    def calc_state_single(self):
        #state_start = time.time()
        body_pose = self.robot_body.pose()
        
        self.body_rpy = body_pose.rpy()
        r, p, yaw = self.body_rpy

        j  = []

        self.bodyRot = glm.mat4()
        self.bodyRot = glm.rotate(self.bodyRot,yaw, glm.vec3(0.0, 0.0, 1.0))
        self.bodyRot = glm.rotate(self.bodyRot,p, glm.vec3(0.0, 1.0, 0.0))
        self.bodyRot = glm.rotate(self.bodyRot,r, glm.vec3(1.0, 0.0, 0.0))

        self.bodyRotNoYaw = glm.mat4()
        self.bodyRotNoYaw = glm.rotate(self.bodyRotNoYaw,p, glm.vec3(0.0, 1.0, 0.0))
        self.bodyRotNoYaw = glm.rotate(self.bodyRotNoYaw,r, glm.vec3(1.0, 0.0, 0.0))

        if self.inputsIsIKTargets:
            fl = [self.jdict["fl1"].servo_target,self.jdict["fl2"].servo_target,self.jdict["fl3"].servo_target]
            fr = [self.jdict["fr1"].servo_target,self.jdict["fr2"].servo_target,self.jdict["fr3"].servo_target]
            bl = [self.jdict["bl1"].servo_target,self.jdict["bl2"].servo_target,self.jdict["bl3"].servo_target]
            br = [self.jdict["br1"].servo_target,self.jdict["br2"].servo_target,self.jdict["br3"].servo_target]
            self.syncLocalFromXYZ(fl,fr,bl,br,self.inputsSpace)
            j = np.concatenate([self.flPosN]+[self.frPosN]+[self.blPosN]+[self.brPosN])
            self.lastJointsStates = j
        else:
            servo_angles =  []
            for j in self.ordered_joints:
                # scaled beetween -1..1 between limits
                limits = j.limits()
                limitsMiddle = (limits[0]+limits[1])*0.5
                jointObservation = 2 * (j.servo_target - limitsMiddle) / (limits[1]-limits[0])
                servo_angles.append(jointObservation)
            j  = servo_angles

        #print("Joints3",(time.time()-state_start)*1000.0)
        #print("Joints",time.time()-state_start)

        #print(j)
        #parts_xyz = np.array( [p.pose().xyz() for p in self.parts.values()] ).flatten()
        curBodyXYZ = (body_pose.xyz()[0], body_pose.xyz()[1], body_pose.xyz()[2])
        self.prev_body_xyz = self.body_xyz
        self.body_xyz = curBodyXYZ
        self.movement_per_frame = [self.body_xyz[0] - self.prev_body_xyz[0],self.body_xyz[1] - self.prev_body_xyz[1], self.body_xyz[2] - self.prev_body_xyz[2]]
        self.walk_target_theta = np.arctan2( self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0] )
        self.vecToTarget = [self.walk_target_x - self.body_xyz[0],self.walk_target_y - self.body_xyz[1], self.walk_target_z - self.body_xyz[2]]
        self.walk_target_dist_prev = self.walk_target_dist
        self.walk_target_dist  = self.getTargetDist()
        self.angle_to_target = self.walk_target_theta - yaw
        frontRot = glm.rotate(glm.mat4x4(),yaw,glm.vec3(0.0,0.0,1.0))
        self.body_frontVec = glm.mat3(frontRot)* glm.vec3(1.0,0.0,0.0);
        '''
        self.rot_minus_yaw = np.array(
            [[np.cos(-yaw), -np.sin(-yaw), 0],
             [np.sin(-yaw),  np.cos(-yaw), 0],
             [           0,             0, 1]]
            )
        '''
        self.angle_to_desired_chassis_yaw = self.walk_target_desired_yaw-yaw
        #yaw = 0
        #normalize yaw
        observationsBase = np.array([
            r, 
            p, 
            yaw/glm.pi(),
            np.sin(self.angle_to_target),
            np.cos(self.angle_to_target),
            np.sin(self.angle_to_desired_chassis_yaw),
            np.cos(self.angle_to_desired_chassis_yaw)
            ], dtype=np.float32)

        # contacts
        feetMemory = 1.0 # sec
        foot_contact_increment = 1.0/(feetMemory*self.physics_step_per_second) # 2 sec
        #print(self.physics_step_per_second) 

        for i,f in enumerate(self.feet):
            self.feet_contact_pos_prev[i] = self.feet_contact_pos[i]
            #pos= self.parts[f]
            self.feet_contact_pos[i] = f.pose().xyz()

            contact_names = set(x.name for x in f.contact_list())
            #print("CONTACT OF '%s' WITH %s" % (f.name, ",".join(contact_names)) )
            #self.feet_contact[i] = 1.0 if (self.foot_ground_object_names & contact_names) else 0.0
            self.feet_contact_prev[i] = self.feet_contact[i]
            self.feet_contact[i] = 1.0 if (len(contact_names)>0) else 0.0
            for f in self.footLinks[i]:
                if self.feet_contact[i]>0.0:
                    self.jdict[f].inAirCost = 0.1
                else:
                    self.jdict[f].inAirCost = 1.0

            if self.feet_contact[i]==1.0:
                self.feet_contact_time[i] = 1.0
            else:
                self.feet_contact_time[i]-=foot_contact_increment
                self.feet_contact_time[i] = np.clip(self.feet_contact_time[i],-1.0,1.0)

            if self.feet_contact_time[i]>0.9:
                self.feet_contact[i]=1.0

        if self.NeedServoObservation:
            observations = np.concatenate([observationsBase]+[self.feet_contact]+[np.float32(j)])
        else:
            observations = np.concatenate([observationsBase]+[self.feet_contact])

        # 0.3 is just scaling typical speed into -1..+1, no physical sense here
        if self.velObservations>0:
            vx, vy, vz = self.robot_body.speed()
            body_vel = np.array([0.3*vx, 0.3*vy, 0.3*vz], dtype=np.float32)
            observations = np.append(observations,body_vel)

        if self.cycleGenerators:
            observations = np.append(observations,[math.sin(self.time*np.pi*2.0)])
            observations = np.append(observations,[math.cos(self.time*np.pi*2.0)])
            observations = np.append(observations,[math.sin(self.time*np.pi*4.0)])

        self.last_single_state = observations

        #print("all",time.time()-state_start)
        return observations.astype(np.float32)

    def getTargetDist(self):
        body_pose = self.robot_body.pose()
        vecToTarget = glm.vec3([self.walk_target_x - self.body_xyz[0],self.walk_target_y - self.body_xyz[1], self.walk_target_z - self.body_xyz[2]])
        walk_target_dist  = glm.length( vecToTarget )
        return walk_target_dist
 
    def calc_potential(self):
        return - self.walk_target_dist / self.physics_time_step

    def calculateAnalyticalActions(self,state,max_skip_frames, ZeroDeltaTime = False):
        # pyrobot lib -1..1  -pi...pi
        deltaTime = self.physics_time_step*max_skip_frames
        if ZeroDeltaTime:
            deltaTime = 0.0
        actions = self.robotLib.getActions(state.tolist(),deltaTime)
        a = np.asarray(actions)
        # convert actions to angles
        angles = np.multiply(a,glm.pi())
        a = np.copy(angles)
        if self.ActionIsAngles==False:
            if self.ActionsIsAdditive:
                '''
                fl = [self.jdict["fl1"].servo_target,self.jdict["fl2"].servo_target,self.jdict["fl3"].servo_target]
                fr = [self.jdict["fr1"].servo_target,self.jdict["fr2"].servo_target,self.jdict["fr3"].servo_target]
                bl = [self.jdict["bl1"].servo_target,self.jdict["bl2"].servo_target,self.jdict["bl3"].servo_target]
                br = [self.jdict["br1"].servo_target,self.jdict["br2"].servo_target,self.jdict["br3"].servo_target]
                self.syncLocalFromXYZ(fl,fr,bl,br,0)
                self.lastJointsStates = np.concatenate([self.flPosN]+[self.frPosN]+[self.blPosN]+[self.brPosN])
                '''
                fl = [a[0],a[1],a[2]]
                fr = [a[3],a[4],a[5]]
                bl = [a[6],a[7],a[8]]
                br = [a[9],a[10],a[11]]
                self.syncLocalFromXYZ(fl,fr,bl,br,0)
                a = np.concatenate([self.flPosN]+[self.frPosN]+[self.blPosN]+[self.brPosN])
                a = np.subtract(a,self.lastJointsStates)
            else:
                fl = [a[0],a[1],a[2]]
                fr = [a[3],a[4],a[5]]
                bl = [a[6],a[7],a[8]]
                br = [a[9],a[10],a[11]]
                self.syncLocalFromXYZ(fl,fr,bl,br,0)
                a = np.concatenate([self.flPosN]+[self.frPosN]+[self.blPosN]+[self.brPosN])
        else:
            if self.ActionsIsAdditive:
                curAngles = [
                    self.jdict["fl1"].servo_target,self.jdict["fl2"].servo_target,self.jdict["fl3"].servo_target,
                    self.jdict["fr1"].servo_target,self.jdict["fr2"].servo_target,self.jdict["fr3"].servo_target,
                    self.jdict["bl1"].servo_target,self.jdict["bl2"].servo_target,self.jdict["bl3"].servo_target,
                    self.jdict["br1"].servo_target,self.jdict["br2"].servo_target,self.jdict["br3"].servo_target]
                a = np.subtract(a,curAngles)
            # convert back to action space
            if self.ActionIsAnglesType==0:
                a = np.divide(a,glm.pi()*0.5)
            elif self.ActionIsAnglesType==1:
                a = np.divide(a,glm.pi())
            elif self.ActionIsAnglesType==2:
                for i,j in enumerate(self.ordered_joints):
                    limits = j.limits()
                    limitsMiddle = (limits[0]+limits[1])*0.5
                    a[i] = 2 * (a[i] - limitsMiddle) / (limits[1]-limits[0])
        return a,angles
        
    def stepMotionCapture(self):
        if self.analyticReward:
            if self.analyticRewardType==0:
                if self.frame==5:
                    self.robotLib.executeCommand("cmd#testAction")
                self.simActions,self.simAngles = self.calculateAnalyticalActions(self.last_state,1,False)
        
    def stepPG(self, a):
        #cp.enable()
        self.time+=self.physics_time_step

        self.apply_action(a)
        self.scene.global_step()

        state = self.calc_state()  # also calculates self.joints_at_limit

        alive_multiplier = 1.0
        alive = float(self.alive_bonus(state))
        if alive<0 and self.frame==0:
            print("bad transition")
        alive*=alive_multiplier

        aliveRecover = 0
        if self.allowRecover:
            if alive<0:
                self.recoverTime+=self.physics_time_step
                aliveRecover = -1
            elif self.recoverTime>0.0:
                self.recoverTime+=self.physics_time_step
                aliveRecover = -1
            if alive>0.0 and self.recoverTime>0.5:
                self.recoverTime = 0.0
            if self.recoverTime>0.0 and self.recoverTime<1.5:
                alive = 0.1
                aliveRecover = -1
            elif self.recoverTime>1.5:
                alive = -1

        #if alive<=-1.0:
        #    alive = -100.0
        if alive<=-1.0 and self.frame==0:
             self.alive_bonus(state)

        done = alive <= -1.0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        alive +=aliveRecover
        # closer is positive , away is negative
        maxDesiredTargetSpeed = 0.10
        deltaDistanceVelocity = (self.walk_target_dist_prev-self.walk_target_dist)/self.physics_time_step
        # clamp only positive
        if deltaDistanceVelocity>maxDesiredTargetSpeed:
            deltaDistanceVelocity = np.clip(deltaDistanceVelocity,-maxDesiredTargetSpeed*2.0,maxDesiredTargetSpeed*2.0)-maxDesiredTargetSpeed
            progress = 1.0-deltaDistanceVelocity/maxDesiredTargetSpeed
        else:
            deltaDistanceVelocity = np.clip(deltaDistanceVelocity,-maxDesiredTargetSpeed,maxDesiredTargetSpeed)
            progress = deltaDistanceVelocity/maxDesiredTargetSpeed
        '''
        potential_old = self.potential
        self.potential = self.calc_potential()
        # progress in m/sec dist to target
        progress = float(self.potential - potential_old)
        self.progressHistory.append(progress)
        '''
        #speed = glm.length(self.movement_per_frame)/self.physics_time_step
        # current robot desired speed is a 10cm per second so 1 is a 0.1
        progressPositiveMultiplier = 5.0
        progressPositiveBias = 0.0
        progressNegativeMultiplier = 5.0
        progressNegativeBias = -0.0
        if progress>0.0:
            progress=progress*progressPositiveMultiplier+progressPositiveBias
        else:
            progress=progress*progressNegativeMultiplier+progressNegativeBias

        chassisDir = glm.vec3(self.bodyRot * glm.vec4([1.0,0.0,0.0,0.0]))
        chassisDir[2] = 0.0
        chassisDir = glm.normalize(chassisDir)

        progressDir = 0.0
        progressDirMultiplier = 1.0
        progressDir = glm.dot(self.desiredChassisDir,chassisDir)
        progressDir = np.clip(progressDir,-1.0,1.0)
        progressDir = np.clip(math.acos(progressDir),0.0,np.pi)
        #try:
        #    progressDir = np.clip(math.acos(progressDir),-1.0,1.0)
        #except ValueError:
        #    print(self.desiredChassisDir,progressDir,chassisDir,flush=True)  
        progressDir = (1.0-progressDir/np.pi)*2.0-1.0
        #-x^0.5+1

        progressDir*=progressDirMultiplier
        #progressDir += progressDir

        '''
        progressDirMultiplier = 0.2                  #10.0
        self.moveDirNormalizer = glm.normalize(self.movement_per_frame)
        progressDir = glm.dot(self.moveDirNormalizer,self.movement_dir)
        progressDir*=progressDirMultiplier
        progress += progressDir

        progressDirMultiplier = 0.1 #10.0
        progressDir = glm.dot(self.body_frontVec,self.movement_dir)
        progressDir*=progressDirMultiplier
        progress += progressDir
        '''
        
        '''
        if progress<0:
            progress = -1.0
        else:
            progress = 1.0
        '''
        '''
        if self.walk_target_dist>0.1 and self.prev_body_xyz is not None:
            vecStep = [self.prev_body_xyz[0] - self.body_xyz[0],self.prev_body_xyz[1] - self.body_xyz[1]] #, self.prev_body_xyz[2] - self.body_xyz[2]]
            vecMovement = np.linalg.norm( vecStep )
            minSpeedToPunishPerSec = 0.01 # 1 cm
            minMovePerFrame = minSpeedToPunishPerSec/self.numStepsPerSecond
            if vecMovement<minMovePerFrame:
                progress = -1
        '''
        atlimits_cost = 0.0
        maxServoAnglePerFrame = self.max_servo_speed*self.physics_time_step
        angleDiffSpeed_cost_max = 0.0
        angleDiffSpeed_cost_min = 1.0
        angleDiffSpeed_cost = 0.0
        angleDiffSpeed_cost_multiplier = 1.0
        for n,j in enumerate(self.ordered_joints):
            #if j.name=="fl1" or j.name=="fr1" or j.name=="bl1" or j.name=="br1":
            #    continue

            limits = j.limits()
            minLimitDiff = limits[0]-j.servo_target
            maxLimitDiff = limits[1]-j.servo_target
            limitAngleOk = math.radians(15)
            minLimRatio = abs(np.clip(minLimitDiff,-limitAngleOk,limitAngleOk))
            maxLimRatio = abs(np.clip(maxLimitDiff,-limitAngleOk,limitAngleOk))
            curJointLimitCost = 0.0
            curJointLimitCost += minLimRatio/limitAngleOk
            curJointLimitCost += maxLimRatio/limitAngleOk
            atlimits_cost += curJointLimitCost * j.angle_cost

            #limits

            angleDiff = j.servo_target-j.idealTargetAngle

            angleOk = math.radians(20) # math.radian maxServoAnglePerFrame #np.pi*0.25
            ratio = abs(np.clip(angleDiff,-angleOk,angleOk))/angleOk
            #ratio = min(abs(j.targetAngleDeltaFrame),maxServoAnglePerFrame)/maxServoAnglePerFrame
            ratio*=j.energy_cost
            ratio*=j.inAirCost
            angleDiffSpeed_cost+=ratio
            angleDiffSpeed_cost_min = min(ratio,angleDiffSpeed_cost_min)
            angleDiffSpeed_cost_max = max(ratio,angleDiffSpeed_cost_max)
            '''
            ratio*=j.energy_cost
            ratio+=(1.0-j.newCurTargetReachable)*0.5
            angleDiffSpeed_cost+=ratio
            angleDiffSpeed_cost_min = min(ratio,angleDiffSpeed_cost_min)
            angleDiffSpeed_cost_max = max(ratio,angleDiffSpeed_cost_max)
            '''
        '''
        lerpFactor = 0.5
        angleDiffSpeed_cost = angleDiffSpeed_cost_min+lerpFactor*(angleDiffSpeed_cost_max-angleDiffSpeed_cost_min)
        '''
        angleDiffSpeed_cost/=12.0
        angleDiffSpeed_cost = np.clip(angleDiffSpeed_cost,-1.0,1.0)
        angleDiffSpeed_cost *= angleDiffSpeed_cost_multiplier
        '''
        index = 0
        avgZ = 0.0
        if self.feet_contact[0]>0.0:
            avgZ = self.parts["fl4_link"].pose().xyz()[2]
        avgZ = self.parts["fl4_link"].pose().xyz()[2]+self.parts["fr4_link"].pose().xyz()[2]+self.parts["bl4_link"].pose().xyz()[2]+self.parts["br4_link"].pose().xyz()[2]
        avgZ =avgZ/4.0
        '''

        atlimits_cost = atlimits_cost/12.0
        atlimits_cost_multiplier = 0.5 #TODO check with 0.5
        atlimits_cost*=atlimits_cost_multiplier

        angleDiffSpeed_cost = 0.0
        #'''
        #energy cost for non contacted chain is better
        # use joint_to_foot
        

        legSlideCost = 0.0
        for i,pos in enumerate(self.feet_contact_pos_prev):
            slideCost = 1.0
            if self.feet_contact[i] and self.feet_contact_prev[i]:
                dist = glm.distance(self.feet_contact_pos[i],self.feet_contact_pos_prev[i])
                vel = dist/self.physics_time_step
                if vel>0.05:
                    slideCost=0.0

            legSlideCost +=slideCost

        legSlideCostMultiplier = (1.0/4.0)*0.25
        legSlideCost=legSlideCost*legSlideCostMultiplier
        

        
        self.rewards_progress.append(progress)
        self.rewards_progressDir.append(progressDir)
        self.reward_alive.append(alive)
        self.reward_angleDiff.append(angleDiffSpeed_cost+atlimits_cost+legSlideCost)

        simulatedReward = 0.0
        if self.analyticReward:
            if self.analyticRewardType==0:
                if self.robotLib.getCurrentCommand()=="cmd#safePose":
                    done = True

                sumDiff = 0.0
                angleMaxDif = glm.radians(10.0)
                for i in range(len(a)):
                    simAction = self.simAngles[i]
                    diff = self.ordered_joints[i].servo_target-simAction
                    if abs(diff)<angleMaxDif:
                        sumDiff+=1.0-abs(diff)/angleMaxDif
                    #sumDiff += diff*diff
                simulatedReward = np.clip((sumDiff/12.0)*3,0.0,1.0)
                #simulatedReward = simulatedReward*simulatedReward
                #simulatedReward = math.exp(-0.1*simulatedReward)
                #simulatedReward = simulatedReward
                self.stepMotionCapture()
            elif self.analyticRewardType==1:
                # sync prev local pos
                fl = [self.jdict["fl1"].servo_target_prev,self.jdict["fl2"].servo_target_prev,self.jdict["fl3"].servo_target_prev]
                fr = [self.jdict["fr1"].servo_target_prev,self.jdict["fr2"].servo_target_prev,self.jdict["fr3"].servo_target_prev]
                bl = [self.jdict["bl1"].servo_target_prev,self.jdict["bl2"].servo_target_prev,self.jdict["bl3"].servo_target_prev]
                br = [self.jdict["br1"].servo_target_prev,self.jdict["br2"].servo_target_prev,self.jdict["br3"].servo_target_prev]
                self.syncLocalFromXYZ(fl,fr,bl,br,0)
                prevLocalsZ = [self.flPosLocal[2],self.frPosLocal[2],self.blPosLocal[2],self.brPosLocal[2]]
                # cur
                fl = [self.jdict["fl1"].servo_target,self.jdict["fl2"].servo_target,self.jdict["fl3"].servo_target]
                fr = [self.jdict["fr1"].servo_target,self.jdict["fr2"].servo_target,self.jdict["fr3"].servo_target]
                bl = [self.jdict["bl1"].servo_target,self.jdict["bl2"].servo_target,self.jdict["bl3"].servo_target]
                br = [self.jdict["br1"].servo_target,self.jdict["br2"].servo_target,self.jdict["br3"].servo_target]
                self.syncLocalFromXYZ(fl,fr,bl,br,0)
                curLocalsZ = [self.flPosLocal[2],self.frPosLocal[2],self.blPosLocal[2],self.brPosLocal[2]]

                curLegsZ = [1,1,1,1]
                for i in range(len(curLegsZ)):
                    if curLocalsZ[i]<prevLocalsZ[i]:
                        curLegsZ[i] = -1

                legFreqUpDown = 0.6 # up down
                starts = [0.0,legFreqUpDown*0.5,legFreqUpDown,legFreqUpDown*1.5]
                cycleLen = starts[-1]+legFreqUpDown

                self.cycleTime = self.time % cycleLen

                legsZ = []
                for cycleStart in starts:
                    ratio = 0.0
                    if self.cycleTime>=cycleStart and self.cycleTime<=(cycleStart+legFreqUpDown):
                        ratio = (self.cycleTime-cycleStart)/legFreqUpDown
                    legZ = 1.0
                    if ratio>0.5:
                        legZ = -1.0
                    legsZ.append(legZ)

                sumDiff = 1.0
                for i in range(4):
                    simAction = legsZ[i]
                    diff = curLegsZ[i]*simAction # just sign
                    if diff>0:
                        sumDiff=sumDiff*2.0
                simulatedReward = np.clip(sumDiff/8.0,0.0,1.0)

        self.reward_sim.append(simulatedReward)

        if self.simRewardOnly:
            self.rewards = [simulatedReward]
        else:
            self.rewards = [
                alive,
                progress,
                progressDir,
                angleDiffSpeed_cost+atlimits_cost+legSlideCost,
                simulatedReward,
                ]

        self.frame  += 1
        rewardSummary = {}

        if (done and not self.done) or (self.spec is not None and self.frame==self.spec.max_episode_steps):
            self.episode_over(self.frame)
            done = True

            reward_progress = np.sum(self.rewards_progress)
            reward_progressDir = np.sum(self.rewards_progressDir)
            reward_angleDiff = np.sum(self.reward_angleDiff)
            reward_alive = np.sum(self.reward_alive)
            reward_sim = np.sum(self.reward_sim)
            rewardTotal = reward_progress+reward_progressDir+reward_angleDiff+reward_alive+reward_sim
            rewardSummary = { "alive":reward_alive, "progress":reward_progress, "servo":reward_angleDiff, "distToTarget":self.walk_target_dist,"simReward":reward_sim, "episode_steps":self.frame, "episode_reward":rewardTotal, 'robot_endpos':self.body_xyz  }

            if self.debugStats:
                print("Episode stats::")
                print("Reward_progress: ", reward_progress)
                print("Reward_angleDiff: ", reward_angleDiff)
                print("Reward_alive: ", reward_alive)
                print("Reward_total: ", reward_alive+reward_progress+reward_angleDiff+simulatedReward)
            self.reward_alive.clear()
            self.rewards_progress.clear()
            self.rewards_progressDir.clear()
            self.reward_angleDiff.clear()
            #find projection to target vector to figureout dist walked
            #distWalked = np.linalg.norm( [self.body_xyz[1],self.body_xyz[0]])
            #print("Dist Walked: ",distWalked)
        self.done   += done   # 2 == 1+True
        if bool(done) and self.frame==1:
            print("First frame done - something bad happended")
        stepReward = sum(self.rewards)
        self.reward += stepReward
        if self.ActionIsAngles:
            self.HUD(state, a, done, "{:0.2f}".format(self.walk_target_dist))
        else:
            self.HUD(state, a, done, "{:0.2f} {:0.2f},{:0.2f},{:0.2f},{:0.2f}".format(self.walk_target_dist,self.jdict['fl1'].linkPos[2],self.jdict['fr1'].linkPos[2],self.jdict['bl1'].linkPos[2],self.jdict['br1'].linkPos[2]))
        #cp.disable()
        #cp.print_stats()
        if done:
            debugBrek = 0
        return state, stepReward, bool(done), rewardSummary

    def HUD(self, s, a, done, msg=''):
        active = self.scene.actor_is_active(self)
        if active and self.done<=2:
            self.scene.cpp_world.test_window_history_advance()
            self.scene.cpp_world.test_window_observations(s.tolist())
            self.scene.cpp_world.test_window_actions(a.tolist())
            self.scene.cpp_world.test_window_rewards(self.rewards)
        if self.done<=1: # Only post score on first time done flag is seen, keep this score if user continues to use env
            s = "%04i %07.1f %s %s" % (self.frame, self.reward, "DONE" if self.done else "", msg) 
            if active:
                self.scene.cpp_world.test_window_score(s)
            #self.camera.test_window_score(s)  # will appear on video ("rgb_array"), but not on cameras istalled on the robot (because that would be different camera)

    def step(self, a):
        #a = self.simActions
        return self.stepPG(a)

    def setupEnergyCost(self,link):
        self.jdict[link+"1"].energy_cost = 1.7 # full link mass
        self.jdict[link+"2"].energy_cost = 2.0 # link mass
        self.jdict[link+"3"].energy_cost = 1.0
        self.jdict[link+"1"].angle_cost = 0.5 # full link mass
        self.jdict[link+"2"].angle_cost = 0.8 # link mass
        self.jdict[link+"3"].angle_cost = 1.0
        self.jdict[link+"1"].idealTargetAngle = 0.0
        self.jdict[link+"2"].idealTargetAngle = math.radians(33)
        self.jdict[link+"3"].idealTargetAngle = -math.radians(70)


    def end_reset(self):

        if self.analyticReward:
            if self.analyticRewardType==0:
                #self.robotLib.executeCommand("cmd#zero")
                self.simActions,self.simAngles = self.calculateAnalyticalActions(self.last_state,1,True)
                #self.robotLib.executeCommand("cmd#initIdle")


    def robot_specific_reset(self):

        global episode_index
        episode_index+=1

        ratio = np.clip((episode_index-self.targetDesired_episode_from)/(self.targetDesired_episode_to-self.targetDesired_episode_from), 0.0,1.0)
        angleVar = self.targetDesired_angleFrom+ratio*(self.targetDesired_angleTo-self.targetDesired_angleFrom)

        angleVar = angleVar*self.targetDesiredYawMultiplier

        self.spawn_yaw = random.uniform(-angleVar, angleVar)*self.spawnYawMultiplier # 0.0 #np.pi

        self.walk_target_desired_yaw = random.uniform(-angleVar, angleVar) # 0.0 #np.pi #np.pi/2.0 #0.0
        desiredBodyDirTM = glm.mat4()
        desiredBodyDirTM = glm.rotate(desiredBodyDirTM,self.walk_target_desired_yaw, glm.vec3([0.0,0.0,1.0]))
        self.desiredChassisDir = glm.normalize(glm.vec3(desiredBodyDirTM * glm.vec4([1.0,0.0,0.0,0.0])))



        self.curTime = 0.0
        self.rewards_progress = []
        self.rewards_progressDir = []
        self.reward_angleDiff = []
        self.reward_alive = []
        self.reward_sim = []
        self.ordered_joints = []
        self.jdict = {}
        self.feet_contact_time = [0.0,0.0,0.0,0.0]
        self.feet_contact_prev = [0.0,0.0,0.0,0.0]
        self.feet_contact_pos = [glm.vec3(0.0),glm.vec3(0.0),glm.vec3(0.0),glm.vec3(0.0)]
        self.feet_contact_pos_prev = [glm.vec3(0.0),glm.vec3(0.0),glm.vec3(0.0),glm.vec3(0.0)]
        self.lastJointsStatesNonClipped = []

        self.defaultAngles = [ 0.207928  
                        ,0.631224  
                        ,-1.173827  
                        ,-0.207928  
                        ,0.631224  
                        ,-1.173827  
                        ,0.207928  
                        ,0.631224  
                        ,-1.173827  
                        ,-0.207928  
                        ,0.631224  
                        ,-1.173827  ]
        '''
fl1   11.91 0.207928  
fl2   36.17 0.631224  
fl3  -67.26 -1.173827  
fr1  -11.91 -0.207928  
fr2   36.17 0.631224  
fr3  -67.26 -1.173827  
bl1   11.91 0.207928  
bl2   36.17 0.631224  
bl3  -67.26 -1.173827  
br1  -11.91 -0.207928  
br2   36.17 0.631224  
br3  -67.26 -1.173827  
        '''
        self.jointToUse = ["fl1","fl2","fl3","fr1","fr2","fr3","bl1","bl2","bl3","br1","br2","br3"]
        for i,j in enumerate(self.urdf.joints):
            if j.name in self.jointToUse:
                #print("\tJoint '%s' limits = %+0.2f..%+0.2f effort=%0.3f speed=%0.3f" % ((j.name,) + j.limits()) )
                j.zCycleTimeLeft = 0.0
                j.zCycle = [0.0,0.0,0.0]
                j.zCycleDir = [0.0,0.0,0.0]
                j.localTargetCycle = self.flMinArea+(self.flMaxArea-self.flMinArea)*0.5
                j.servo_target = self.defaultAngles[i]
                j.servo_target_prev = j.servo_target
                j.type_1_servo_target = j.servo_target
                j.set_servo_target(j.servo_target,self.servo_kp,self.servo_kd,self.servo_maxTorque)
                j.power_coef, j.max_velocity = j.limits()[2:4]
                j.jointsLocalPos = self.jointsLocalPos[j.name]

                self.ordered_joints.append(j)
                self.lastJointsStatesNonClipped.append(0.0)
                self.jdict[j.name] = j
                continue
        
        self.jdict["fl4"]=DummyJoint()
        self.jdict["fr4"]=DummyJoint()
        self.jdict["bl4"]=DummyJoint()
        self.jdict["br4"]=DummyJoint()
        self.jdict["fl4"].jointsLocalPos = self.jointsLocalPos["fl4"]
        self.jdict["fr4"].jointsLocalPos = self.jointsLocalPos["fr4"]
        self.jdict["bl4"].jointsLocalPos = self.jointsLocalPos["bl4"]
        self.jdict["br4"].jointsLocalPos = self.jointsLocalPos["br4"]
 
        self.iklinks_fl = [self.jdict["fl1"],self.jdict["fl2"],self.jdict["fl3"]]
        self.iklinks_fr = [self.jdict["fr1"],self.jdict["fr2"],self.jdict["fr3"]]
        self.iklinks_bl = [self.jdict["bl1"],self.jdict["bl2"],self.jdict["bl3"]]
        self.iklinks_br = [self.jdict["br1"],self.jdict["br2"],self.jdict["br3"]]
        self.legs1Bodies = [self.jdict["fl1"],self.jdict["fr1"],self.jdict["bl1"],self.jdict["br1"]]
        self.legs4Bodies = [self.jdict["fl4"],self.jdict["fr4"],self.jdict["bl4"],self.jdict["br4"]]


        self.jdict["fl1"].child = self.jdict["fl2"]
        self.jdict["fl2"].child = self.jdict["fl3"]
        self.jdict["fl3"].child = self.jdict["fl4"]
        self.jdict["fl4"].child = None

        self.jdict["fr1"].child = self.jdict["fr2"]
        self.jdict["fr2"].child = self.jdict["fr3"]
        self.jdict["fr3"].child = self.jdict["fr4"]
        self.jdict["fr4"].child = None

        self.jdict["bl1"].child = self.jdict["bl2"]
        self.jdict["bl2"].child = self.jdict["bl3"]
        self.jdict["bl3"].child = self.jdict["bl4"]
        self.jdict["bl4"].child = None

        self.jdict["br1"].child = self.jdict["br2"]
        self.jdict["br2"].child = self.jdict["br3"]
        self.jdict["br3"].child = self.jdict["br4"]
        self.jdict["br4"].child = None

        self.ldict = {}
        self.ldict[0] = self.jdict["fl1"]
        self.ldict[1] = self.jdict["fr1"]
        self.ldict[2] = self.jdict["bl1"]
        self.ldict[3] = self.jdict["br1"]

        self.jdict["fl1"].debug_color = 0xFF8080
        self.jdict["fl1"].legIndex = 0
        self.jdict["fr1"].debug_color = 0xFFff80
        self.jdict["fr1"].legIndex = 1

        self.jdict["bl1"].debug_color = 0xFF80ff
        self.jdict["bl1"].legIndex = 2
        self.jdict["br1"].debug_color = 0xFFffff
        self.jdict["br1"].legIndex = 3

        self.setupEnergyCost("fl")
        self.setupEnergyCost("fr")
        self.setupEnergyCost("bl")
        self.setupEnergyCost("br")
 
        RoboschoolForwardWalker.robot_specific_reset(self)
        self.body_xyz = [0.0,0,0.18]
        self.body_rpy = [0.0,0.0,self.spawn_yaw] #np.pi/2.0]
        self.move_robot(self.body_xyz[0],self.body_xyz[1],self.body_xyz[2],self.body_rpy[0],self.body_rpy[1],self.body_rpy[2])
        self.numStepsPerSecond = 1.0/self.physics_time_step
        self.numSecondsToTrack = 1.0
        self.numSecondsToTrackDistance = 1.0
        self.walked_distance = deque(maxlen=int(self.numSecondsToTrackDistance*self.numStepsPerSecond))
        self.progressHistory = deque(maxlen=int(self.numSecondsToTrack*self.numStepsPerSecond))
        if settings.history1Len>0.0:
            self.history1 = deque(maxlen=int(settings.history1Len*self.numStepsPerSecond))
        if settings.history2Len>0.0:
            self.history2 = deque(maxlen=int(settings.history2Len*self.numStepsPerSecond))
        if settings.history3Len>0.0:
            self.history3 = deque(maxlen=int(settings.history3Len*self.numStepsPerSecond))
        self.state_historyLen = 2.0
        self.state_history = deque(maxlen=int(self.state_historyLen*self.numStepsPerSecond))
        self.recoverTime = 0.0
            #self.robotLib.executeCommand("cmd#testAction")
            #self.robotLib.executeCommand("cmd#initIdle")

    def move_robot(self, init_x, init_y, init_z, init_rx=0, init_ry=0, init_rz=0):
        "Used by multiplayer stadium to move sideways, to another running lane."
        self.cpp_robot.query_position()
        pose = self.cpp_robot.root_part.pose()
        pose.move_xyz(init_x, init_y, init_z)  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
        pose.set_rpy(init_rx, init_ry, init_rz)  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
        self.cpp_robot.set_pose(pose)
        self.start_pos_x, self.start_pos_y, self.start_pos_z = init_x, init_y, init_z

    def reset(self):
        if hasattr(self,"scene")==False or self.scene is None:
            self.scene = self.create_single_player_scene()
        if not self.scene.multiplayer:
            self.scene.episode_restart(self.advancedLevel,self.addObstacles)

        pose = cpp_household.Pose()
        #import time
        #t1 = time.time()
        self.urdf = self.scene.cpp_world.load_urdf(
            self.model_urdf,
            pose,
            self.fixed_base,
            self.self_collision)
        #t2 = time.time()
        #print("URDF load %0.2fms" % ((t2-t1)*1000))

        self.ordered_joints = []
        self.jdict = {}
        self.parts = {}
        self.frame = 0
        self.done = 0
        self.reward = 0
        dump = 0
        r = self.urdf
        self.cpp_robot = r
        if dump: print("ROBOT '%s'" % r.root_part.name)
        if r.root_part.name==self.robot_name:
            self.robot_body = r.root_part
        for part in r.parts:
            if dump: print("\tPART '%s'" % part.name)
            self.parts[part.name] = part
            if part.name==self.robot_name:
                self.robot_body = part
        for j in r.joints:
            if dump: print("\tALL JOINTS '%s' limits = %+0.2f..%+0.2f effort=%0.3f speed=%0.3f" % ((j.name,) + j.limits()) )
            if j.name[:6]=="ignore":
                j.set_motor_torque(0)
                continue
            j.power_coef, j.max_velocity = j.limits()[2:4]
            self.ordered_joints.append(j)
            self.jdict[j.name] = j
        #print("ordered_joints", len(self.ordered_joints))
        self.robot_specific_reset()
        self.cpp_robot.query_position()
        self.walk_target_dist_prev = self.getTargetDist()
        self.walk_target_dist = self.walk_target_dist_prev
        s = self.calc_state()
        self.potential = self.calc_potential()
        self.camera = self.scene.cpp_world.new_camera_free_float(self.VIDEO_W, self.VIDEO_H, "video_camera")
        #self.camera.move_and_look_at(0,1,0,0,0,0)
        self.end_reset()
        return s

    def cycleLeg(self,j0,index,target,minArea,maxArea):
        zCycleDeltaPerSec = 0.1 #  10 cm per sec 
        zCycleStepTime = 0.1 #1.0/25 # from -1 to 1 
        zCycleStepOffset = zCycleDeltaPerSec*zCycleStepTime
        allowCancelCycle = False
        if allowCancelCycle and ((target[index]>0.5 and j0.zCycleDir[index]<0.0) or (target[index]<-0.5 and j0.zCycleDir[index]>0.0)): 
            j0.zCycle[index] = 0.0
        if j0.zCycle[index]<=0.0:
            if target[index]>0.1: #up
                j0.localTargetCycle[index] = j0.localPos[index]+zCycleStepOffset
                target[index] = j0.localTargetCycle[index]
                j0.zCycle[index] = zCycleStepTime
                j0.zCycleDir[index] = 1.0
            elif target[index]<-0.1: #down
                j0.localTargetCycle[index] = j0.localPos[index]-zCycleStepOffset
                target[index] = j0.localTargetCycle[index]
                j0.zCycle[index] = zCycleStepTime
                j0.zCycleDir[index] = -1.0
            else:
                target[index] = j0.localTargetCycle[index]
                j0.zCycleDir[index]= 0.0
        else:
            target[index] = j0.localTargetCycle[index]
            j0.zCycle[index] -=self.physics_time_step
        target[index] = np.clip(target[index],minArea[index],maxArea[index])
        j0.localTargetCycle[index] = np.clip(j0.localTargetCycle[index],minArea[index],maxArea[index])
        j0.zCycle[index] = 0
        return target

    def moveLegAxis(self,j0,index,target,minArea,maxArea):
        zCycleDeltaPerSec = 0.1 #  10 cm per sec 
        zCycleStepOffset = zCycleDeltaPerSec*self.physics_time_step
        j0.localTargetCycle[index] = j0.localPos[index]+target[index]*zCycleStepOffset
        target[index] = j0.localTargetCycle[index]
        '''
        if target[index]>0.5: #up
            j0.localTargetCycle[index] = j0.localPos[index]+zCycleStepOffset
            target[index] = j0.localTargetCycle[index]
        elif target[index]<-0.5: #down
            j0.localTargetCycle[index] = j0.localPos[index]-zCycleStepOffset
            target[index] = j0.localTargetCycle[index]
        j0.localTargetCycle[index] = np.clip(j0.localTargetCycle[index],minArea[index],maxArea[index])
        '''
        return target[index]

    def doLegCycle(self,j0,target,minArea,maxArea):

        localPos = j0.localPos

        self.bodyRotInv = glm.inverse(self.bodyRotNoYaw)

        xyMaxDist = 0.1

        localXDir = glm.vec3(1.0,0.0,0.0)
        localXDir = glm.vec3(self.bodyRotInv * glm.vec4(localXDir,0.0))
        localYDir = glm.vec3(0.0,1.0,0.0)
        localYDir = glm.vec3(self.bodyRotInv * glm.vec4(localYDir,0.0))


        '''
        if self.feet_contact_time[j0.legIndex]<0.9:
            #localPos[0] = glm.mix(minArea[0],maxArea[0],target[0]*0.5+0.5)
            #localPos[1] = glm.mix(minArea[1],maxArea[1],target[1]*0.5+0.5)
            localPos = glm.mix(minArea[0],maxArea[0],target[0]*0.5+0.5)
            localPos[1] = glm.mix(minArea[1],maxArea[1],target[1]*0.5+0.5)
        '''
        # z cycle
        zCycleLen = 1.0
        zCycleZSpeed = 0.1 # up down speed
        localZDir = glm.vec3(0.0,0.0,1.0)
        localZDir = glm.vec3(self.bodyRotInv * glm.vec4(localZDir,0.0))
        if j0.zCycleTimeLeft<=0.0:
            if target[2]>0.0:
                # start cycle
                j0.zCycleTimeLeft = zCycleLen
                j0.targetXY = glm.vec2(target[0],target[1])*xyMaxDist*zCycleLen
                localPos = localPos+localZDir*zCycleZSpeed*self.physics_time_step
                factor = 1.0-(j0.zCycleTimeLeft-self.physics_time_step)
                target = localXDir*j0.targetXY[0]+j0.targetXY[1]*localYDir
                localPos = glm.mix(localPos, target, factor)
        else:
            #else:
#            if target[2]<-0.1:
#                # force stop cycle
#                j0.zCycleTimeLeft = 0.0
#            else:
                # cycle
            j0.zCycleTimeLeft-=self.physics_time_step
            if j0.zCycleTimeLeft>zCycleLen*0.5: # more than half
                localPos = localPos+localZDir*zCycleZSpeed*self.physics_time_step
                factor = (1.0-j0.zCycleTimeLeft)
                target = localXDir*j0.targetXY[0]+j0.targetXY[1]*localYDir
                localPos = glm.mix(localPos, target, factor)
            else:
                if self.feet_contact_time[j0.legIndex]>0.9:
                    # force stop cycle on leg ground
                    j0.zCycleTimeLeft = 0.0
                else:        
                    localPos = localPos-localZDir*zCycleZSpeed*self.physics_time_step
                    factor = (1.0-j0.zCycleTimeLeft)
                    target = localXDir*j0.targetXY[0]+j0.targetXY[1]*localYDir
                    localPos = glm.mix(localPos, target, factor)
        target = localPos
        #distance = glm.distance(j0.localPos,target)
        #speed = distance/self.physics_time_step
        #if speed>0.11:
        #    print(speed)
        return target


    def syncLegWordTM(self,chassisTM,lindex):
        j0 = self.ldict[lindex]
        j1 = j0.child
        j2 = j1.child
        j3 = j2.child
        angles = [j0.servo_target,j1.servo_target,j2.servo_target]
        j0.worldPos = self.getLeg1XYZFromAngles(angles,self.link1,self.link2,j0.jointsLocalPos,chassisTM)
        j1.worldPos = self.getLeg2XYZFromAngles(angles,self.link1,self.link2,j0.jointsLocalPos,j1.jointsLocalPos,chassisTM)
        j2.worldPos = self.getLeg3XYZFromAngles(angles,self.link1,self.link2,j0.jointsLocalPos,j1.jointsLocalPos,chassisTM)
        j3.worldPos = self.getLeg4XYZFromAngles(angles,self.link1,self.link2,j0.jointsLocalPos,j1.jointsLocalPos,chassisTM)

    def syncWordTM(self,chassisTM):
        self.syncLegWordTM(chassisTM,0)
        self.syncLegWordTM(chassisTM,1)
        self.syncLegWordTM(chassisTM,2)
        self.syncLegWordTM(chassisTM,3)

    def calculateChassisIK(self,currentChasisTM,desiredChasisTM,fixedLegs):
        self.syncWordTM(currentChasisTM)
        chassisTmInv = glm.inverse(currentChasisTM)
        #// remeber current legs pos
        legsWorldPos = [self.legs4Bodies[0].worldPos,self.legs4Bodies[1].worldPos,self.legs4Bodies[2].worldPos,self.legs4Bodies[3].worldPos]
        self.syncWordTM(desiredChasisTM)
        desiredChassisTmInv = glm.inverse(desiredChasisTM);
        reachability = [1,1,1,1]
        j2LocalPos = [self.jointsLocalPos["fl2"],self.jointsLocalPos["fr2"],self.jointsLocalPos["bl2"],self.jointsLocalPos["br2"]]

        for n in range(0, 4):
            if fixedLegs[n]<0.001:
                self.legs1Bodies[n].linkPos = self.legs1Bodies[n].localPos
                self.legs1Bodies[n].newCurTarget = self.legs1Bodies[n].servo_target
                self.legs1Bodies[n].newCurTargetReachable = 1
                self.legs1Bodies[n].child.newCurTarget = self.legs1Bodies[n].child.servo_target
                self.legs1Bodies[n].child.newCurTargetReachable = 1
                self.legs1Bodies[n].child.child.newCurTarget = self.legs1Bodies[n].child.child.servo_target
                self.legs1Bodies[n].child.child.newCurTargetReachable = 1
                continue
            legs4ChassisSpace = glm.vec3(desiredChassisTmInv * glm.vec4(legsWorldPos[n], 1.0))
            legs4ChassisSpacePrev = glm.vec3(chassisTmInv * glm.vec4(legsWorldPos[n], 1.0))

            legs1 = self.legs1Bodies[n].worldPos
            legs1ChassisSpace = glm.vec3(desiredChassisTmInv * glm.vec4(legs1, 1.0))

            # move chassis to zero
            legs4ChassisSpace = legs4ChassisSpace - legs1ChassisSpace
            legs4ChassisSpacePrev = legs4ChassisSpacePrev - legs1ChassisSpace

            newPos = legs4ChassisSpace # moveTo(legs4ChassisSpacePrev, legs4ChassisSpace, maxDiffClamp);
            # 0.4578671455383301 -0.01,        0.035,       -0.165 
            angles = self.solveLeg(j2LocalPos[n],self.len1,self.len2,newPos)
            reachability[n] = 0
            if angles.w > 0.0:
                reachability[n] = 1

            self.legs1Bodies[n].localPos = newPos
            self.legs1Bodies[n].linkPos = newPos
            self.legs1Bodies[n].newCurTarget = angles[0]
            self.legs1Bodies[n].newCurTargetReachable = angles[3]
            self.legs1Bodies[n].child.newCurTarget = angles[1]
            self.legs1Bodies[n].child.newCurTargetReachable = angles[3]
            self.legs1Bodies[n].child.child.newCurTarget = angles[2]
            self.legs1Bodies[n].child.child.newCurTargetReachable = angles[3]
        return reachability

    def moveChassis(self,dx,dy,dz):
        deltaTM = glm.mat4()
        deltaTM = glm.translate(deltaTM,glm.vec3(dx,dy,dz))
        #auto rotTm = glm.mat4()
        #rotTM = glm.translate()
        newChasisTM = deltaTM * self.bodyRotNoYaw
        return self.calculateChassisIK(self.bodyRotNoYaw,newChasisTM,self.feet_contact)


    def syncAngles(self,linkName, target, j0, j1, j2, minArea,maxArea, chassisSpace):
        target = np.clip(target, -1.0, +1.0)
        j2LocalPos = self.jointsLocalPos[linkName+"2"]
        if chassisSpace==0: #local space
            target[0] = glm.mix(minArea[0],maxArea[0],target[0]*0.5+0.5)
            target[1] = glm.mix(minArea[1],maxArea[1],target[1]*0.5+0.5)
            target[2] = glm.mix(minArea[2],maxArea[2],target[2]*0.5+0.5)
            targetWorld = target+self.jointsLocalPos[linkName+"1"]
            targetWorld = glm.vec3(self.bodyRot * glm.vec4(targetWorld[0],targetWorld[1],targetWorld[2],0.0))
            j = self.jdict[linkName+"1"]
            j.sphere = self.scene.cpp_world.debug_sphere(float(targetWorld[0]+self.body_xyz[0]),float(targetWorld[1]+self.body_xyz[1]),float(targetWorld[2]+self.body_xyz[2]),0.005,j.debug_color)
        elif chassisSpace==1: # chasis space
            target[0] = glm.mix(self.chassisSpaceMin[0],self.chassisSpaceMax[0],target[0]*0.5+0.5)
            target[1] = glm.mix(self.chassisSpaceMin[1],self.chassisSpaceMax[1],target[1]*0.5+0.5)
            target[2] = glm.mix(self.chassisSpaceMin[2],self.chassisSpaceMax[2],target[2]*0.5+0.5)
            target = target-self.jointsLocalPos[linkName+"1"]
        elif chassisSpace==2: #cycle space
            #target[0] = glm.mix(minArea[0],maxArea[0],target[0]*0.5+0.5)
            #target[1] = glm.mix(minArea[1],maxArea[1],target[1]*0.5+0.5)
            #target[0] = self.cycleLeg(j0,0,target,minArea,maxArea)[0]
            #target[1] = self.cycleLeg(j0,1,target,minArea,maxArea)[1]
            #target[2] = self.cycleLeg(j0,2,target,minArea,maxArea)[2]
            target[0] = self.moveLegAxis(j0,0,target,minArea,maxArea)
            target[1] = self.moveLegAxis(j0,1,target,minArea,maxArea)
            target[2] = self.moveLegAxis(j0,2,target,minArea,maxArea)
        elif chassisSpace==3: # world space
            target[0] = glm.mix(self.chassisRotSpaceMin[0],self.chassisRotSpaceMax[0],target[0]*0.5+0.5)
            target[1] = glm.mix(self.chassisRotSpaceMin[1],self.chassisRotSpaceMax[1],target[1]*0.5+0.5)
            target[2] = glm.mix(self.chassisRotSpaceMin[2],self.chassisRotSpaceMax[2],target[2]*0.5+0.5)
            j = self.jdict[linkName+"1"]
            j.sphere = self.scene.cpp_world.debug_sphere(float(target[0]+self.body_xyz[0]),float(target[1]+self.body_xyz[1]),float(target[2]+self.body_xyz[2]),0.005,j.debug_color)
            bodyRotNoYawInv = glm.inverse(self.bodyRotNoYaw)
            target = glm.vec3(bodyRotNoYawInv * glm.vec4(target[0],target[1],target[2],0.0))
            target = target-self.jointsLocalPos[linkName+"1"]
            #self.jdict[linkName+"1"].sphere = self.scene.cpp_world.debug_sphere(target[0],target[1],target[2],0.02,0xFF8080)
            #self.scene.cpp_world.debug_line(0,0,0,1,1,1,0xFF8080)
        elif chassisSpace==4: # world space
            target[0] = glm.mix(self.chassisRotSpaceMin[0],self.chassisRotSpaceMax[0],target[0]*0.5+0.5)
            target[1] = glm.mix(self.chassisRotSpaceMin[1],self.chassisRotSpaceMax[1],target[1]*0.5+0.5)
            target[2] = glm.mix(self.chassisRotSpaceMin[2],self.chassisRotSpaceMax[2],target[2]*0.5+0.5)
            bodyRotInv = glm.inverse(self.bodyRot)
            target = glm.vec3(bodyRotInv * glm.vec4(target[0],target[1],target[2],0.0))
        elif chassisSpace==5: #sin cycle space
            target = self.doLegCycle(j0,target,minArea,maxArea)
            targetWorld = target+self.jointsLocalPos[linkName+"1"]
            targetWorldFrom = j0.localPos+self.jointsLocalPos[linkName+"1"]
            targetWorld = glm.vec3(self.bodyRot * glm.vec4(targetWorld[0],targetWorld[1],targetWorld[2],0.0))
            targetWorldFrom = glm.vec3(self.bodyRot * glm.vec4(targetWorldFrom[0],targetWorldFrom[1],targetWorldFrom[2],0.0))
            delta = glm.normalize(targetWorld-targetWorldFrom)
            spherePos = targetWorldFrom+delta*0.05
            j = self.jdict[linkName+"1"]
            j.sphere = self.scene.cpp_world.debug_sphere(float(spherePos[0]+self.body_xyz[0]),float(spherePos[1]+self.body_xyz[1]),float(spherePos[2]+self.body_xyz[2]),0.005,j.debug_color)
        elif chassisSpace==6: #chassis space dir
            targetDir = glm.vec3(0.0)
            if glm.length(target)>0.0:
                targetDir = glm.normalize(target)
            
            localPos = j0.localPos #+self.jointsLocalPos[linkName+"1"]

            self.bodyRotInv = glm.inverse(self.bodyRotNoYaw)
            localXDir = glm.vec3(1.0,0.0,0.0)
            localXDir = glm.normalize(glm.vec3(self.bodyRotInv * glm.vec4(localXDir,0.0)))
            localYDir = glm.vec3(0.0,1.0,0.0)
            localYDir = glm.normalize(glm.vec3(self.bodyRotInv * glm.vec4(localYDir,0.0)))
            localZDir = glm.vec3(0.0,0.0,1.0)
            localZDir = glm.normalize(glm.vec3(self.bodyRotInv * glm.vec4(localZDir,0.0)))

            deltaSpeed = 0.1
            localDir = localXDir*targetDir[0]+localYDir*targetDir[1]+localZDir*targetDir[2]
            localDir = localDir*deltaSpeed*self.physics_time_step
            target=localPos+localDir

            targetWorld = target+self.jointsLocalPos[linkName+"1"]
            targetWorldFrom = j0.localPos+self.jointsLocalPos[linkName+"1"]
            targetWorld = glm.vec3(self.bodyRot * glm.vec4(targetWorld[0],targetWorld[1],targetWorld[2],0.0))
            targetWorldFrom = glm.vec3(self.bodyRot * glm.vec4(targetWorldFrom[0],targetWorldFrom[1],targetWorldFrom[2],0.0))
            delta = glm.normalize(targetWorld-targetWorldFrom)
            spherePos = targetWorldFrom+delta*0.05
            j = self.jdict[linkName+"1"]
            j.sphere = self.scene.cpp_world.debug_sphere(float(spherePos[0]+self.body_xyz[0]),float(spherePos[1]+self.body_xyz[1]),float(spherePos[2]+self.body_xyz[2]),0.005,j.debug_color)


        # limit z
        target[2] = np.clip(target[2],-(self.len1+self.len2),-0.1)
        
        angles = self.solveLeg(j2LocalPos,self.len1,self.len2,target)
        j0.linkPos = target
        j0.newCurTarget = angles[0]
        j0.newCurTargetReachable = angles[3]
        j1.newCurTarget = angles[1]
        j1.newCurTargetReachable = angles[3]
        j2.newCurTarget = angles[2]
        j2.newCurTargetReachable = angles[3]


    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        a = a * self.actionsMult
        '''
        a[0] = 0
        a[1] = 0.0
        a[2] = 1.0 #1.0
        #if self.frame>300:
        #    a[2] = 1.0

        a[3] = 0
        a[4] = 0
        a[5] = 0
        
        a[6] = 0
        a[7] = 0
        a[8] = 0
        
        a[9] = 0
        a[10] = 0
        a[11] = 0
        '''
        '''
        a[12] = 0
        a[13] = 0
        a[14] = 0
        if self.frame<300:
            a[14] = -1.0
        else:
            a[14] = 1.0
        '''
        #a[0],a[1],a[2] = self.rangeNormalize([0.0,0.195,-0.15])
        #a[3],a[4],a[5] = self.rangeNormalize([0.0,-0.195,-0.15])
        #a[6],a[7],a[8] = self.rangeNormalize([0.0,0.195,-0.15])
        #a[9],a[10],a[11] = self.rangeNormalize([0.0,-0.195,-0.15])

        #angles = float(np.clip([self.ordered_joints[0],self.ordered_joints[1],self.ordered_joints[2]], -1.0, +1.0))
        #angles[0] = glm.mix(angles[0]*0.5+0.5,self.minArea[0],self.minArea[0])
        #angles[1] = glm.mix(angles[1]*0.5+0.5,self.minArea[1],self.minArea[1])
        #angles[2] = glm.mix(angles[2]*0.5+0.5,self.minArea[2],self.minArea[2])

        #angles1 = self.solveLeg(self.jointsLocalPos["fl2"],self.len1,self.len2,[-0.07,0.035,-0.15-0.25])

        #angles = self.solveLeg(self.jointsLocalPos["fl2"],self.len1,self.len2,[0,0.-0.035,-0.15-0.25])

        #angles = self.solveLeg(self.jointsLocalPos["fl2"],self.len1,self.len2,[0.1,0.-0.035,-0.10])
        
        if self.ActionIsAngles:
            for n,j in enumerate(self.ordered_joints):
                if self.ActionsIsAdditive:
                    if self.AdditiveAnglesType==1:
                        j.newCurTarget = float(np.clip(a[n], -1, +1))*np.pi+j.type_1_servo_target
                        j.type_1_servo_target = j.newCurTarget
                    elif self.AdditiveAnglesType==2:
                        delta = float(np.clip(a[n], -self.AdditiveAnglesClip, self.AdditiveAnglesClip))
                        j.newCurTarget = delta+j.servo_target
                    elif self.AdditiveAnglesType==3:
                        delta = float(np.clip(a[n], -1, +1))*np.pi*0.5
                        j.newCurTarget = delta+j.servo_target
                    else:
                        delta = float(np.clip(a[n], -1, +1))*np.pi
                        j.newCurTarget = delta+j.servo_target
                else:
                    normalizedAngle = float(np.clip(a[n], -1, +1))
                    if self.ActionIsAnglesType==0:
                        j.newCurTarget = normalizedAngle*np.pi*0.5
                    elif self.ActionIsAnglesType==1:
                        j.newCurTarget = normalizedAngle*np.pi
                    elif self.ActionIsAnglesType==2:
                        limits = j.limits()
                        j.newCurTarget = glm.mix(limits[0],limits[1],normalizedAngle*0.5+0.5)
                j.newCurTargetReachable = 1
        else:
            a = np.clip(a, -1.0, +1.0)
            a = a * 1.0
            if self.ActionsIsAdditive:
                a = np.add(self.lastJointsStatesNonClipped,a).clip(-1.0,1.0)
                self.lastJointsStatesNonClipped = a
            #a = np.clip(a, -.0, +.0)
            if self.moveChassisActions:
                moveChassisStepTime = 0.1 #1.0/25 # from -1 to 1 
                moveChassisStepTime = moveChassisStepTime*self.physics_time_step
                '''
                dx = 0.0
                dy = 0.0
                dz = 0.0
                if a[12]>0.5:
                    dx = moveChassisStepTime
                elif a[12]<-0.5:
                    dx =-moveChassisStepTime
                #
                if a[13]>0.5:
                    dy = moveChassisStepTime
                elif a[13]<-0.5:
                    dy =-moveChassisStepTime
                #
                if a[14]>0.5:
                    dz = moveChassisStepTime
                elif a[14]<-0.5:
                    dz =-moveChassisStepTime
                '''
                dx = a[12]*moveChassisStepTime
                dy = a[13]*moveChassisStepTime
                dz = a[14]*moveChassisStepTime

                #dx = 0.0
                #dy = 0.0
                ##dz = 0.0
                self.moveChassis(dx,dy,dz) #0.0,0.0,-0.01)#a[12],a[13],a[14])
                
            self.syncAngles("fl",[a[0],a[1],a[2]],self.jdict['fl1'],self.jdict['fl2'],self.jdict['fl3'],self.flMinArea,self.flMaxArea, self.actionsSpace)
            self.syncAngles("fr",[a[3],a[4],a[5]],self.jdict['fr1'],self.jdict['fr2'],self.jdict['fr3'],self.frMinArea,self.frMaxArea, self.actionsSpace)
            self.syncAngles("bl",[a[6],a[7],a[8]],self.jdict['bl1'],self.jdict['bl2'],self.jdict['bl3'],self.blMinArea,self.blMaxArea, self.actionsSpace)
            self.syncAngles("br",[a[9],a[10],a[11]],self.jdict['br1'],self.jdict['br2'],self.jdict['br3'],self.brMinArea,self.brMaxArea, self.actionsSpace)


        if self.recoverTime>0.0:
            self.jdict["fl2"].newCurTarget = math.radians(45)
            self.jdict["fr2"].newCurTarget = math.radians(45)
            self.jdict["bl2"].newCurTarget = math.radians(45)
            self.jdict["br2"].newCurTarget = math.radians(45)
            self.jdict["fl3"].newCurTarget = math.radians(-45)
            self.jdict["fr3"].newCurTarget = math.radians(-45)
            self.jdict["bl3"].newCurTarget = math.radians(-45)
            self.jdict["br3"].newCurTarget = math.radians(-45)

        for n,j in enumerate(self.ordered_joints):
            curTarget = j.newCurTarget

            ''' Test pose 
            if(j.name=="fl1"):
                curTarget = math.radians(0);
            if(j.name=="fr1"):
                curTarget = math.radians(0);
            if(j.name=="bl1"):
                curTarget = math.radians(0);
            if(j.name=="br1"):
                curTarget = math.radians(0);

            if(j.name=="fl2"):
                curTarget = math.radians(45);
            if(j.name=="fr2"):
                curTarget = math.radians(45);
            if(j.name=="fl3"):
                curTarget = math.radians(-45);
            if(j.name=="fr3"):
                curTarget = math.radians(-45);
            if(j.name=="bl2"):
                curTarget = math.radians(45);
            if(j.name=="br2"):
                curTarget = math.radians(45);
            if(j.name=="bl3"):
                curTarget = math.radians(-45);
            if(j.name=="br3"):
                curTarget = math.radians(-45);
            '''

            j.targetAngle = curTarget
            j.targetAngleDelta = curTarget-j.servo_target
            #curTarget = math.radians(90)
            prevTarget = j.servo_target
            deltaAngle = curTarget-prevTarget
            maxServoAnglePerFrame = self.max_servo_speed*self.physics_time_step
            deltaAngleAbs = glm.clamp(deltaAngle,-maxServoAnglePerFrame,maxServoAnglePerFrame)
            curTarget = prevTarget+deltaAngleAbs
 
            #clamp angles to limits
            limits = j.limits()
            if curTarget<limits[0]:
                curTarget = limits[0]
                j.newCurTargetReachable = 0
            if curTarget>limits[1]:
                curTarget = limits[1]
                j.newCurTargetReachable = 0

            
            j.targetAngleDeltaFrame = curTarget-j.servo_target
            j.servo_target_prev = j.servo_target
            j.servo_target = curTarget

            #j.servo_target =  math.sin(self.time)*np.pi*0.3
            #print(j.servo_target," ",end='')
            j.set_servo_target(j.servo_target,self.servo_kp,self.servo_kd,self.servo_maxTorque)
            #j.set_motor_torque( self.power*j.power_coef*float(np.clip(a[n], -1, +1)) )
        #print('')

class QuadruppedWalker(RoboschoolForwardWalkerMujocoXML):
    '''
    3-D Quadruped walker similar to roboschool. 
    The task is to make the creature walk as fast as possible
    '''
    foot_list = ['fl4_link', 'fr4_link', 'bl4_link', 'br4_link']
    joint_to_foot  = { 
                    "fl1" : "fl4_link",
                    "fl2" : "fl4_link",
                    "fl3" : "fl4_link",

                    "fr1" : "fr4_link",
                    "fr2" : "fr4_link",
                    "fr3" : "fr4_link",
                    
                    "bl1" : "bl4_link",
                    "bl2" : "bl4_link",
                    "bl3" : "bl4_link",
                    
                    "br1" : "br4_link",
                    "br2" : "br4_link",
                    "br3" : "br4_link"
                }
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(self, "cat.urdf", "chassis")

    def alive_bonus(self,state):
        vecStep = [0,0,0]
        vecMovement = 0.0
        if self.prev_body_xyz is not None:
            vecStep = [self.prev_body_xyz[0] - self.body_xyz[0],self.prev_body_xyz[1] - self.body_xyz[1]] #, self.prev_body_xyz[2] - self.body_xyz[2]]
            vecMovement = np.linalg.norm( vecStep )
            self.walked_distance.append(vecMovement)
        if len(self.walked_distance)==self.walked_distance.maxlen:
            distLastInterval = np.sum(self.walked_distance)
            minDistPerSecToWalk = 0.02 # m per sec
            minAvgDist = minDistPerSecToWalk #/self.numStepsPerSecond
            #        '''
            if distLastInterval<minAvgDist:
                if self.debugStats:
                    print("Long wait without moving:",distLastInterval," ",minAvgDist)
                return -1.0 #0.1
        #'''
        #distToLine = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
        #if np.abs(self.body_xyz[1])>0.5:
        #    print("Far away from central line ",self.body_xyz[1] )
        #    return -1
        maxRoll = 40.0
        maxPitch = 40.0
        r, p, yaw = self.body_rpy
        # roll pitch angles check
        #if abs(r)>math.radians(maxRoll) or abs(p)>math.radians(maxPitch):
            #print(r,p,yaw)
        #    return -1
        # body height check
        avgZ = self.parts["fl4_link"].pose().xyz()[2]+self.parts["fr4_link"].pose().xyz()[2]+self.parts["bl4_link"].pose().xyz()[2]+self.parts["br4_link"].pose().xyz()[2]
        avgZ =avgZ/4.0
        if self.body_xyz[2]<avgZ+0.05:
            return -1
        legsBias = 1.0
        if not self.inputsIsIKTargets:
            # sync local position
            fl = [self.jdict["fl1"].servo_target,self.jdict["fl2"].servo_target,self.jdict["fl3"].servo_target]
            fr = [self.jdict["fr1"].servo_target,self.jdict["fr2"].servo_target,self.jdict["fr3"].servo_target]
            bl = [self.jdict["bl1"].servo_target,self.jdict["bl2"].servo_target,self.jdict["bl3"].servo_target]
            br = [self.jdict["br1"].servo_target,self.jdict["br2"].servo_target,self.jdict["br3"].servo_target]

            self.syncLocalFromXYZ(fl,fr,bl,br,self.inputsSpace)

        if True or self.inputsIsIKTargets:

            maxLocalValue = 0.07
            maxLocalValueDecrease = 0.04
            delta = maxLocalValue-maxLocalValueDecrease
            if self.flPosLocal[0]<-maxLocalValueDecrease and self.frPosLocal[0]<-maxLocalValueDecrease:
                legsBias = max((self.flPosLocal[0]+maxLocalValueDecrease)/delta,(self.frPosLocal[0]+maxLocalValueDecrease)/delta)
            if self.blPosLocal[0]>maxLocalValueDecrease and self.brPosLocal[0]>maxLocalValueDecrease:
                legsBias = -min((self.blPosLocal[0]-maxLocalValueDecrease)/delta,(self.brPosLocal[0]-maxLocalValueDecrease)/delta)

        flAngles = (self.jdict["fl2"].servo_target+self.jdict["fl3"].servo_target)
        if abs(flAngles)>glm.radians(90.0):
            return -1
        frAngles = (self.jdict["fr2"].servo_target+self.jdict["fr3"].servo_target)
        if abs(frAngles)>glm.radians(90.0):
            return -1
        blAngles = (self.jdict["bl2"].servo_target+self.jdict["bl3"].servo_target)
        if abs(blAngles)>glm.radians(90.0):
            return -1
        brAngles = (self.jdict["br2"].servo_target+self.jdict["br3"].servo_target)
        if abs(brAngles)>glm.radians(90.0):
            return -1
        rRatio = 0.5
        rRatioAngle = 0.5
        pRatio = 0.5
        rPitchAngle = 0.5
        angleReward = (1.0-abs(r)/math.radians(maxRoll*rRatioAngle))*rRatio+(1.0-abs(p)/math.radians(maxPitch*rPitchAngle))*pRatio
        aliveRes = min(legsBias,np.clip(angleReward,-1.0,1.0))
        if aliveRes<=-1.0:
            dddd = 0
        return aliveRes

    def __setstate__(self, state):
        self.__init__()

    def sample_tasks(self, num_tasks):
        positions = self.np_random.uniform(-3.0, 3.0, size=(num_tasks, 2))
        tasks = [{'position': position} for position in positions]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal_pos = task['position']

class QStadiumScene(Scene):
    zero_at_running_strip_start_line = True   # if False, center of coordinates (0,0,0) will be at the middle of the stadium
    stadium_halflen   = 105*0.25    # FOOBALL_FIELD_HALFLEN
    stadium_halfwidth = 50*0.25     # FOOBALL_FIELD_HALFWID

    def episode_restart(self,advancedLevel,addObstacles):
        Scene.episode_restart(self)   # contains cpp_world.clean_everything()
        stadium_pose = cpp_household.Pose()
        if self.zero_at_running_strip_start_line:
            stadium_pose.set_xyz(0, 0, 0.0)
        #self.stadium = self.cpp_world.load_thingy(
        #    os.path.join(os.path.dirname(__file__), "models_env/terrain.obj"),
        #    stadium_pose, 1.0, 0, 0xFFFFFF, True)
        #self.ground_plane_mjcf = self.cpp_world.load_mjcf(os.path.join(os.path.dirname(__file__), "models_env/ground_plane.xml"))
        #'''

        if advancedLevel:
            self.stadium = self.cpp_world.load_urdf(
                os.path.join(os.path.dirname(__file__), "models_env/terrain.urdf"),
                stadium_pose,True,False)

            if addObstacles:
                self.obstacles = []
                for i in range(30):
                    sphere_pose = cpp_household.Pose()
                    sphere_pose.set_xyz(0.5+random.uniform(0.0, 1.0), random.uniform(-0.3, 0.3), 0.0)
                    self.obstacles.append(self.cpp_world.load_urdf(
                        os.path.join(os.path.dirname(__file__), "models_env/sphere.urdf"),
                        sphere_pose,True,False))
        else:
            filename = os.path.join(os.path.dirname(__file__),"models_env/plane.urdf")
            if self.zero_at_running_strip_start_line:
                stadium_pose.set_xyz(0, 0, 0.0)
            self.stadium = self.cpp_world.load_urdf(
                filename,
                stadium_pose,
            True,
                False)
        '''
        self.boxes = []
        xOffset = 1.0
        for i in range(30):
            sphere_pose = cpp_household.Pose()
            sphere_pose.set_xyz(xOffset, 0, -0.045)
            sphere_pose.set_rpy(0.0, np.pi/4, 0.0)
            self.boxes.append(self.cpp_world.load_urdf(
                os.path.join(os.path.dirname(__file__), "models_env/box.urdf"),
                sphere_pose,True,False))
            xOffset+=0.3
        '''



class QSinglePlayerStadiumScene(QStadiumScene):
    "This scene created by environment, to work in a way as if there was no concept of scene visible to user."
    multiplayer = False

