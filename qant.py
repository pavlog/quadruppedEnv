import numpy as np
import gym
from gym import wrappers
import argparse
import pprint as pp
import roboschool
import math
from gym.envs.registration import register
from roboschool.scene_abstract import cpp_household
#from roboschool.scene_stadium import SinglePlayerStadiumScene
from roboschool.gym_forward_walker import RoboschoolForwardWalker
from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv
from roboschool.gym_urdf_robot_env import RoboschoolUrdfEnv
from roboschool.scene_stadium import SinglePlayerStadiumScene
from roboschool.gym_mujoco_walkers import RoboschoolForwardWalkerMujocoXML
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import os, sys
from collections import deque
from numpy import arccos, array, dot, pi, cross
from numpy.linalg import det, norm
from importlib import import_module
import time
from quadruppedEnv import settings
#from pyrr import Quaternion, Matrix44, Vector3
#import cProfile
#cp = cProfile.Profile()
import glm
import xml.etree.ElementTree as ET


class QAnt(RoboschoolForwardWalkerMujocoXML):
    '''
    3-D Quadruped walker similar to MuJoCo Ant. 
    The task is to make the creature walk as fast as possible
    '''
    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(self, "qant.xml", "torso", action_dim=8, obs_dim=28-3, power=2.5/100.0/10)
    def alive_bonus(self, z, r, p, yaw):
        if z < 0.07:
            return -1
        maxRoll = 30.0
        maxPitch = 25.0
        r, p, yaw = self.body_rpy
        # roll pitch angles check
        if abs(r)>math.radians(maxRoll) or abs(p)>math.radians(maxPitch):
            #print(r,p,yaw)
            return -1
        return 1


    def reset(self):
        if self.scene is None:
            self.scene = self.create_single_player_scene()
        if not self.scene.multiplayer:
            self.scene.episode_restart()
        self.mjcf = self.scene.cpp_world.load_mjcf(os.path.join(os.path.dirname(__file__), "mujoco_assets", self.model_xml))
        self.ordered_joints = []
        self.jdict = {}
        self.parts = {}
        self.frame = 0
        self.done = 0
        self.reward = 0
        dump = 0
        for r in self.mjcf:
            if dump: print("ROBOT '%s'" % r.root_part.name)
            if r.root_part.name==self.robot_name:
                self.cpp_robot = r
                self.robot_body = r.root_part
            for part in r.parts:
                if dump: print("\tPART '%s'" % part.name)
                self.parts[part.name] = part
                if part.name==self.robot_name:
                    self.cpp_robot = r
                    self.robot_body = part
            for j in r.joints:
                if dump: print("\tALL JOINTS '%s' limits = %+0.2f..%+0.2f effort=%0.3f speed=%0.3f" % ((j.name,) + j.limits()) )
                if j.name[:6]=="ignore":
                    j.set_motor_torque(0)
                    continue
                j.power_coef = 100.0
                self.ordered_joints.append(j)
                self.jdict[j.name] = j
        assert(self.cpp_robot)
        self.robot_specific_reset()
        self.body_xyz = [-0.3,0,0.25]
        RoboschoolForwardWalker.move_robot(self,self.body_xyz[0],self.body_xyz[1],self.body_xyz[2])
        for r in self.mjcf:
            r.query_position()
        s = self.calc_state()    # optimization: calc_state() can calculate something in self.* for calc_potential() to use
        self.potential = self.calc_potential()
        self.camera = self.scene.cpp_world.new_camera_free_float(self.VIDEO_W, self.VIDEO_H, "video_camera")
        return s

    def calc_state(self):
        j = np.array([j.current_relative_position() for j in self.ordered_joints], dtype=np.float32).flatten()
        # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        self.joint_speeds = j[1::2]
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

        body_pose = self.robot_body.pose()
        parts_xyz = np.array( [p.pose().xyz() for p in self.parts.values()] ).flatten()
        self.body_xyz = (parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2])  # torso z is more informative than mean z
        self.body_rpy = body_pose.rpy()
        z = self.body_xyz[2]
        r, p, yaw = self.body_rpy
        if self.initial_z==None:
            self.initial_z = z
        self.walk_target_theta = np.arctan2( self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0] )
        self.walk_target_dist  = np.linalg.norm( [self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]] )
        self.angle_to_target = self.walk_target_theta - yaw

        self.rot_minus_yaw = np.array(
            [[np.cos(-yaw), -np.sin(-yaw), 0],
             [np.sin(-yaw),  np.cos(-yaw), 0],
             [           0,             0, 1]]
            )
        vx, vy, vz = np.dot(self.rot_minus_yaw, self.robot_body.speed())  # rotate speed back to body point of view

        more = np.array([
            yaw,
            np.sin(self.angle_to_target), np.cos(self.angle_to_target),
            #0.3*vx, 0.3*vy, 0.3*vz,    # 0.3 is just scaling typical speed into -1..+1, no physical sense here
            r, p], dtype=np.float32)
        return np.clip( np.concatenate([more] + [j] + [self.feet_contact]), -5, +5)

    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        for n,j in enumerate(self.ordered_joints):
            torque = self.power*j.power_coef*float(np.clip(a[n], -1, +1))
            j.set_motor_torque( torque )

    def step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then step() for all robots with the same actions
            self.apply_action(a)
            self.scene.global_step()

        state = self.calc_state()  # also calculates self.joints_at_limit

        alive = float(self.alive_bonus(state[0]+self.initial_z, self.body_rpy[0],self.body_rpy[1],self.body_rpy[2]))   # state[0] is body height above ground, body_rpy[1] is pitch
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i,f in enumerate(self.feet):
            contact_names = set(x.name for x in f.contact_list())
            #print("CONTACT OF '%s' WITH %s" % (f.name, ",".join(contact_names)) )
            self.feet_contact[i] = 1.0 if (self.foot_ground_object_names & contact_names) else 0.0
            if contact_names - self.foot_ground_object_names:
                feet_collision_cost += self.foot_collision_cost

        electricity_cost  = self.electricity_cost  * float(np.abs(a*self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.joints_at_limit)

        self.rewards = [
            alive,
            progress,
            electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost
            ]

        self.frame  += 1
        if (done and not self.done) or self.frame==self.spec.max_episode_steps:
            self.episode_over(self.frame)
            done = True
        self.done   += done   # 2 == 1+True
        self.reward += sum(self.rewards)
        self.HUD(state, a, done)
        return state, sum(self.rewards), bool(done), {}

class QAntOrg(QAnt):
    '''
    3-D Quadruped walker similar to MuJoCo Ant. 
    The task is to make the creature walk as fast as possible
    '''
    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(self, "qantOrg.xml", "torso", action_dim=8, obs_dim=28-3, power=2.5)
    def alive_bonus(self, z, r, p, yaw):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground
