"""
Version of environment to communicate with EEG side.
Sends image observations to and expects skill and parameter selection from EEG side.
"""

import gym
import numpy as np
from gym import spaces
import cv2
import time
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
        controller_type="OSC_POSE",
        general_cfg_file="config/charmander.yml",
        control_freq=20,
        workspace_limits={"x" : (0.35, 0.55), "y" : (-0.15, 0.25), "z" : (0.03, 0.45)},
        skill_config={
            "waypoint_height" : 0.25,
            "idx2skill" : None,
        },
        gripper_thresh=0.04, # gripper q below this value is considered "closed"
        normalized_params=True,
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
            2 : CameraRedisSubInterface(camera_id=2),
        }
        for id in self.camera_interfaces.keys():
            self.camera_interfaces[id].start()

        # setup skills
        self.skill = PrimitiveSkill(
            controller_type=self.controller_type,
            controller_config=self.controller_config,
            robot_interface=self.robot_interface,
            waypoint_height=skill_config["waypoint_height"],
            workspace_limits=self.workspace_limits,
        )

        self.num_skills = self.skill.num_skills
        self.current_infos = {}

        self.reset_q = None

        # reset joints
        print("--------- Resseting Joints -----------")
        self.skill._reset_joints(np.append(self.skill.reset_joint_positions["from_top"], -1.0))

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

    def step(self, action): 

        """
        Commands robot to execute skill corresponding to input action
        
        Args:
            action : skill selection vector concatenated with skill parameters
        """ 

        # execute skill
        self.skill.execute_skill(action)
        reward, done, info = self._post_action(action)
        # # upload images
        # self._upload_images()
        obs = {}

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
        self._upload_obj_selection_images()
        obs = self._receive_object_states()
        robot_state = self._get_current_robot_state()
        obs.update(robot_state)

        self.current_infos = {}
        return obs

    def get_image_observations(self, action, img_as_list=False, save_images=False):
        """
        Takes images for object selection and parameter selection, and uploads them to server
        """

        # take images for object selection
        obj_image0 = get_camera_image(self.camera_interfaces[0])
        obj_image1 = get_camera_image(self.camera_interfaces[1])

        # move robot out of camera view
        self._move_out_of_way(action)

        # take images for parameter selection (image 2 for x and y, image 0 or 1 for z)
        param_image0 = get_camera_image(self.camera_interfaces[0])
        param_image1 = get_camera_image(self.camera_interfaces[1])
        param_image2 = get_camera_image(self.camera_interfaces[2])

        if save_images:
            cv2.imwrite("obj_selection_img0.png", obj_image0)
            cv2.imwrite("obj_selection_img1.png", obj_image1)
            cv2.imwrite("param_selection_img0.png", param_image0)
            cv2.imwrite("param_selection_img1.png", param_image1)
            cv2.imwrite("param_selection_img2.png", param_image2)

        # move back to original position
        self._move_back_in_view()

        if img_as_list:
            data = {
                "obj_image0" : obj_image0.tolist(),
                # "obj_image1" : obj_image1.tolist(),
                "param_image0" : param_image0.tolist(),
                # "param_image1" : param_image1.tolist(),
                "param_image2" : param_image2.tolist(),
            }
        else:
            data = {
                "obj_image0" : obj_image0,
                # "obj_image1" : obj_image1,
                "param_image0" : param_image0,
                # "param_image1" : param_image1,
                "param_image2" : param_image2,
            }

        return data

    def _move_out_of_way(self, action):
        """
        Move the robot out of camera view (joint position depends on which skill was called)
        """
        # save current joint configuration
        self.reset_q = np.append(self.robot_interface.last_q, self.skill._get_gripper_state())

        skill_idx = np.argmax(action[:self.skill.num_skills])
        skill_name = self.skill.idx2skill[skill_idx]
        gripper_state = self.skill._get_gripper_state()
        if skill_name in ["pick_from_side", "pour_from_side"]:
            out_of_way_q = self.skill.reset_joint_positions["out_of_way_side"]
        else:
            out_of_way_q = self.skill.reset_joint_positions["out_of_way_top"]
        out_of_way_q = out_of_way_q + [gripper_state]

        self.skill._reset_joints(out_of_way_q)

    def _move_back_in_view(self):
        """
        Move arm back to joint configuration before self._move_out_of_way() is called
        """
        self.skill._reset_joints(self.reset_q)

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