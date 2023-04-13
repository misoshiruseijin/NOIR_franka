import gym
import numpy as np
from gym import spaces
import cv2
import time
from PIL import Image, ImageDraw, ImageFont
from abc import abstractmethod

import sys
sys.path.append("..")
from primitive_skills_noir import PrimitiveSkill

from deoxys.utils.config_utils import (get_default_controller_config, verify_controller_config)
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import transform_utils
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.camera_redis_interface import CameraRedisSubInterface
from utils.camera_utils import get_camera_image

import pdb

class RealRobotEnvMulti(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(
        self,
        controller_type,
        general_cfg_file="config/charmander.yml",
        control_freq=20,
        workspace_limits={"x" : (0.35, 0.55), "y" : (-0.15, 0.25), "z" : (0.03, 0.45)},
        skill_config={
            "waypoint_height" : 0.25,
            "idx2skill" : None,
        },
        gripper_thresh=0.04, # gripper q below this value is considered "closed"
        normalized_params=True,
        reset_joint_pos=None,
    ): 

        super().__init__()
        
        self.workspace_limits = workspace_limits
        self.control_freq = control_freq

        self.gripper_thresh = gripper_thresh
        self.normalized_params = normalized_params

        # setup robot interface
        self.controller_type = controller_type
        self.controller_config = get_default_controller_config(self.controller_type)
        self.robot_interface = FrankaInterface(
            general_cfg_file=general_cfg_file,
            control_freq=self.control_freq,
        )

        # setup camera interfaces
        self.camera_interfaces = {
            0 : CameraRedisSubInterface(camera_id=0),
            1 : CameraRedisSubInterface(camera_id=1),
        }
        for id in self.camera_interfaces.keys():
            self.camera_interfaces[id].start()

        # setup skills
        if reset_joint_pos is None:
            reset_joint_pos = [0.07263956, -0.34306933, -0.01955571, -2.45878116, -0.01170808, 2.18055725, 0.84792026]
        self.skill = PrimitiveSkill(
            controller_type=self.controller_type,
            controller_config=self.controller_config,
            robot_interface=self.robot_interface,
            waypoint_height=skill_config["waypoint_height"],
            workspace_limits=self.workspace_limits,
            reset_joint_pos=reset_joint_pos,
        )

        self.num_skills = self.skill.num_skills

        # Define action and observation spaces - TODO this needs to be changed
        # if self.use_skills:
        #     max_num_params = self.skill.max_num_params
        #     print("max num params: ", max_num_params)
        #     high = np.ones(self.num_skills + max_num_params)
        #     low = -high
        #     self.action_space = spaces.Box(low=low, high=high)
        # else:
        #     high = np.ones(5) if use_yaw else np.ones(4)
        #     # high = np.ones(7)
        #     low = -high
        #     self.action_space = spaces.Box(low=low, high=high)

        self.current_infos = {}

        # send initial camera images to server
        self._upload_camera_images()
        

    def reward(self,):
        """
        Reard function for task. Returns environment reward only.
        Environment reward is given when task is successfully complete:
            pan placed on stove -> sausage placed on pan -> sausage placed in bread 
                in that order

        Returns:
            reward (float) : environment reward
        """
        
        return 0

    def step(self, action=[], execute=True): # TODO this should be changed to use skills only - we don't need low level control

        """
        Commands robot to execute skill corresponding to input action
        
        Args:
            action : skill selection vector concatenated with skill parameters
        """ 
        # receive action from server
        action = self._receive_action()

        # execute skill
        self.skill.execute_skill(action)
        reward, done, info = self._post_action(action)

        # send new camera images to server
        self._upload_camera_images()

        # receive observations
        obs = self._receive_object_states()  
        robot_state = self._get_current_robot_state()
        obs.update(robot_state)

        return obs, reward, done, info

    def _post_action(self, action): # TODO reset only when task is complete or if human wants it to??
            """
            Do any housekeeping after taking an action.
            Args:
                action (np.array): Action to execute within the environment
            Returns:
                3-tuple:
                    - (float) reward from the environment
                    - (bool) whether the current episode is completed or not
                    - (dict) empty dict to be filled with information by subclassed method
            """
            reward = self.reward()
            return reward, False, {}
        
    def reset(self): # TODO - flattening should happen in wrapper if needed
        """
        Reset robot to home joint configuration and returns observation
        """
        reset_joints_to(self.robot_interface, self.skill.reset_joint_positions)
        time.sleep(0.5)
        self._upload_camera_images()
        obs = self._receive_object_states()
        robot_state = self._get_current_robot_state()
        obs.update(robot_state)

        self.current_infos = {}
        return obs

    def _get_current_robot_state(self):
        """
        Return current proprio state of robot [position, quat, gripper state]
        Gripper state is binary : 1 = closed, -1 = opened
        """
        current_eef_pose = self.robot_interface.last_eef_quat_and_pos
        current_pos = current_eef_pose[1].flatten()
        current_quat = current_eef_pose[0]
        gripper_q = self.robot_interface.last_gripper_q
        gripper_state = 1 if gripper_q < self.gripper_thresh else -1

        # get current state
        current_eef_pose = self.robot_interface.last_eef_pose
        current_rot = current_eef_pose[:3, :3]
        current_euler = transform_utils.mat2euler(current_rot)
        current_yaw = current_euler[-1]

        robot_state = { 
            "eef_pos" : current_pos,
            "eef_quat" : current_quat,
            "eef_yaw" : current_yaw,
            "gripper_state" : gripper_state,
        }
        return robot_state

    def _upload_camera_images(self):

        raw_image0 = get_camera_image(self.camera_interfaces[0])
        rgb_image0 = raw_image0[:,:,::-1] # convert from bgr to rgb
        raw_image1 = get_camera_image(self.camera_interfaces[1])
        rgb_image1 = raw_image1[:,:,::-1] # convert from bgr to rgb
        
        # TODO - send rgb images to server

    def _receive_action(self):
        """
        Gets action from server
        TODO - decide between:
            get entire action vector : need to have skill dictionary on eeg side
            get skill name + param vector : no additional info needed on eeg side
        """
        action = [] # TODO get this from server
        return action

    def _receive_object_states(self):
        """
        Get 3d coordinate of objects 
        """
        obs = {} # TODO get this from server
        return obs

