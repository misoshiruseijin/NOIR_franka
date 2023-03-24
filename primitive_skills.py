"""Example script of moving robot joint positions."""
import time

import numpy as np

import sys
sys.path.insert(1, "/home/eeg/deoxys_control/deoxys")

from deoxys import config_root
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig, transform_utils
from deoxys.utils.config_utils import (get_default_controller_config, verify_controller_config)
from deoxys.utils.input_utils import input2action
from deoxys.utils.log_utils import get_deoxys_example_logger

import pdb

logger = get_deoxys_example_logger()

"""
At default reset joint angles:
    eef_ori = array([
        [ 0.99941218,  0.01790704,  0.02890312],
        [ 0.01801811, -0.9998216 , -0.00358691],
        [ 0.02883373,  0.00410558, -0.99957578]
    ])
       
    eef_pos = array([0.45775618, 0.03207872, 0.26534091])
    axis_angle = [3.13730597 0.02814467 0.04520512]

"""
class PrimitiveSkill:

    def __init__(
        self,
        # interface_config="config/charmander.yml",
        controller_type,
        controller_config,
        robot_interface,
        waypoint_height=0.25,
        workspace_limits=None,
        yaw_limits=None,
        idx2skill=None,
        aff_pos_thresh=None,
        use_yaw=True,
        reset_joint_pos=None,
        ):

        """
        Args: 
            interface_config (str) : path to robot interface config yaml file
            controller_type (str) : name of controller. defaults to OSC_POSE
            waypoint_height (float) : height of waypoint used in multi-step skills (pick, place, push)
            workspace_limits (dict) : {"x" : (x lower limit, x upper limit), "y" : (y lower limit, y upper limit), "z" : (z lower limit, z upper limit) }
            yaw_limits (tuple of floats) : (yaw lower limit, yaw upper limit)
            idx2skill (dict) : maps skill index (int) in one-hot skill-selection action vector to skill name (str). If unspecified, uses default settings
            aff_pos_thresh (dict) : maps skill name (str) to reaching position threshold for computing affordance scores. If unspecified, uses default settings
        """

        self.robot_interface = robot_interface
        self.controller_type = controller_type
        self.controller_config = controller_config

        # robot home position, waypoint height and workspace / yaw limits
        if reset_joint_pos is None:
            self.reset_joint_positions = [
                0.09162008114028396,
                -0.19826458111314524,
                -0.01990020486871322,
                -2.4732269941140346,
                -0.01307073642274261,
                2.30396583422025,
                0.8480939705504309,
            ]
        else:
            self.reset_joint_positions = reset_joint_pos
        self.reset_eef_pos = [0.45775618, 0.03207872, 0.26534091]
        self.reset_axis_angle_ori = [np.pi, 0.0, 0.0]
        self.reset_yaw = 0.0
        self.use_yaw = use_yaw
        
        self.waypoint_height = waypoint_height # height of waypoint in pick, place, push skills
        if workspace_limits is not None:
            self.workspace_limits = workspace_limits
        else: 
            self.workspace_limits = {
                "x" : (0.3, 0.55),
                "y" : (-0.15, 0.25),
                "z" : (0.03, 0.30)
            }
        if yaw_limits is not None:
            self.yaw_limits = yaw_limits
        else:
            self.yaw_limits = (-0.5*np.pi, 0.5*np.pi)

        # skill settings 
        self.skills = {
            "move_to" : {
                "max_steps" : 200,
                "num_params" : 5 if self.use_yaw else 4,
                "skill" : self._move_to,
                "default_idx" : 0,
                "default_aff_thresh" : 0.05,
            },

            "gripper_release" : {
                "max_steps" : 25,
                "num_params" : 0,
                "skill" : self._gripper_release,
                "default_idx" : 1,
                "default_aff_thresh" : None,
            },

            "gripper_close" : {
                "max_steps" : 25,
                "num_params" : 0,
                "skill" : self._gripper_close,
                "default_idx" : 2,
                "default_aff_thresh" : None,
            },

            "pick" : {
                "max_steps" : 300,
                "num_params" : 4 if self.use_yaw else 3,
                "skill" : self._pick,
                "default_idx" : 3,
                "default_aff_thresh" : 0.03,
            },

            "place" : {
                "max_steps" : 300,
                "num_params" : 4 if self.use_yaw else 3,
                "skill" : self._place,
                "default_idx" : 4,
                "default_aff_thresh" : 0.03,
            },

            "push" : {
                "max_steps" : 150,
                "num_params" : 8 if self.use_yaw else 7,
                "skill" : self._push,
                "default_idx" : 5,
                "default_aff_thresh" : 0.1,
            },

            "push_x" : {
                "max_steps" : 100,
                "num_params" : 5 if self.use_yaw else 4,
                "skill" : self._push_x,
                "default_idx" : 6,
                "default_aff_thresh" : 0.1,
            },

            "push_y" : {
                "max_steps" : 100,
                "num_params" : 5 if self.use_yaw else 4,
                "skill" : self._push_y,
                "default_idx" : 7,
                "default_aff_thresh" : 0.1,
            },
        }

        if idx2skill is None:
            self.idx2skill = { skill["default_idx"] : name for name, skill in self.skills.items()}
        else:
            for name in idx2skill.values():
                assert name in self.skills.keys(), f"Error with skill {name}. Skill name must be one of {self.skills.keys()}"
            self.idx2skill = idx2skill    
    
        self.num_skills = len(self.idx2skill)
        self.max_num_params = max([self.skills[skill_name]["num_params"] for skill_name in self.idx2skill.values()])
        # self.max_num_params = max([self.idx2skill[]["num_params"] for skill_name in self.skills.keys()])

        self.steps = 0 # number of steps spent so far on this skill
        self.grip_steps = 0 # number of consequtive steps spent on gripper close/releaes
        self.phase = 0 # keeps track of which phase the skill is in for multiple-step skills (pick, place, push)
        self.prev_success = False # whether previous phase succeeded 

        # affordance score settings
        if aff_pos_thresh is not None:
            assert all(key in aff_pos_thresh.keys() for key in self.idx2skill.values())
            self.aff_pos_thresh = aff_pos_thresh
        else:
            self.aff_pos_thresh = { name : skill["default_aff_thresh"] for name, skill in self.skills.items() }

        self.aff_tanh_scaling = 1.0

    def get_keypoints_dict(self):
        """
        Return dictionary with skill names as keys and all values None. Used for affordance reward calculation
        """
        keypoints = {key : None for key in self.idx2skill.values()}
        return keypoints
    
    def compute_affordance_reward(self, action, keypoint_dict):
        """
        Computes afforance reward given action and keypoints

        Args:
            action (array): action
            keypoint_dict (dict) : maps skill name to keypoints. keypoints can be None or list of coordinates
                "None" indicates that the skill is relevant at any position (choosing this skill is never penalized regardless of position parameters)
                Empty list indicates that the skill is not relevant at any position (choosing this skill is always penalized regardless of position parameters)

        Returns:
            affordance_reward (float) : affordance reward for choosing given action
        """
        print("keypoints", keypoint_dict)
        skill_idx = np.argmax(action[:self.num_skills])
        skill_name = self.idx2skill[skill_idx]
        keypoints = keypoint_dict[skill_name] # keypoints for chosen skill
        print("for chosen skill", keypoints)
        reach_pos = action[self.num_skills:self.num_skills + 3] # component of params corresponding to reach position
        if keypoints is None:
            return 1.0

        if len(keypoints) == 0:
            return 0.0

        aff_centers = np.stack(keypoints)
        dist = np.clip(np.abs(aff_centers - reach_pos) - self.aff_pos_thresh[skill_name], 0, None)
        min_dist = np.min(np.sum(dist, axis=1))
        aff_reward = 1.0 - np.tanh(self.aff_tanh_scaling * min_dist)
        return aff_reward

    def get_action(self, action):
        """
        Args:
            action (array): one-hot vector for skill selection concatenated with skill parameters
                one-hot vector dimension must be same as self.n_skills. skill parameter can have variable dimension
        Returns:
            action (7d-array): action commands for simulation environment - (position commands, orientation commands, gripper command)    
            skill_done (bool): True if goal skill completed successfully or if max allowed steps is reached
            skill_success (bool) : True if skill completed successfully
        """
        skill_idx = np.argmax(action[:self.num_skills])
        skill = self.skills[self.idx2skill[skill_idx]]["skill"]
        params = action[self.num_skills:]
        # print("Skill called", self.idx2skill[skill_idx])
        action, skill_done, skill_success = skill(params, return_n_steps=False)

        return action, skill_done, skill_success

    def execute_skill(self, action):

        skill_idx = np.argmax(action[:self.num_skills])
        skill = self.skills[self.idx2skill[skill_idx]]["skill"]
        params = action[self.num_skills:]
        skill_done = False
        
        while not skill_done:
            # print("step", self.steps)
            action_ll, skill_done, skill_success, n_steps = skill(
                params=params,
                return_n_steps=True,
            )

            if skill_done:
                break 

            # print("action ll", action_ll)
            self.robot_interface.control(
                controller_type=self.controller_type,
                action=action_ll,
                controller_cfg=self.controller_config,
            )

        self.steps = 0
        return skill_success, n_steps - 1

    def unnormalize_params(self, action): # TODO add case for push 1d's
        """
        Unnormalizes parameters from [-1, 1] to raw values

        Args:
            action : one-hot skill selection vector concatenated with params

        Returns: 
            action : action with unnormalized params
        """
        # find out which skill is called
        action = action.copy()
        skill_idx = np.argmax(action[:self.num_skills])
        skill_name = self.idx2skill[skill_idx]
        params = action[self.num_skills:]

        if skill_name == "push":
            params[0] = ( ((params[0] + 1) / 2 ) * (self.workspace_limits["x"][1] - self.workspace_limits["x"][0]) ) + self.workspace_limits["x"][0]
            params[1] = ( ((params[1] + 1) / 2 ) * (self.workspace_limits["y"][1] - self.workspace_limits["y"][0]) ) + self.workspace_limits["y"][0]
            params[2] = ( ((params[2] + 1) / 2 ) * (self.workspace_limits["z"][1] - self.workspace_limits["z"][0]) ) + self.workspace_limits["z"][0]
            params[3] = ( ((params[3] + 1) / 2 ) * (self.workspace_limits["x"][1] - self.workspace_limits["x"][0]) ) + self.workspace_limits["x"][0]
            params[4] = ( ((params[4] + 1) / 2 ) * (self.workspace_limits["y"][1] - self.workspace_limits["y"][0]) ) + self.workspace_limits["y"][0]
            params[5] = ( ((params[5] + 1) / 2 ) * (self.workspace_limits["z"][1] - self.workspace_limits["z"][0]) ) + self.workspace_limits["z"][0]
            if self.use_yaw:
                params[6] = ( ((params[6] + 1) / 2 ) * (self.yaw_limits[1] - self.yaw_limits[0]) ) + self.yaw_limits[0]

        elif skill_name == "push_x":
            params[0] = ( ((params[0] + 1) / 2 ) * (self.workspace_limits["x"][1] - self.workspace_limits["x"][0]) ) + self.workspace_limits["x"][0]
            params[1] = ( ((params[1] + 1) / 2 ) * (self.workspace_limits["y"][1] - self.workspace_limits["y"][0]) ) + self.workspace_limits["y"][0]
            params[2] = ( ((params[2] + 1) / 2 ) * (self.workspace_limits["z"][1] - self.workspace_limits["z"][0]) ) + self.workspace_limits["z"][0]
            params[3] = ( ((params[3] + 1) / 2 ) * (self.workspace_limits["x"][1] - self.workspace_limits["x"][0]) ) + self.workspace_limits["x"][0]
            if self.use_yaw:
                params[4] = ( ((params[6] + 1) / 2 ) * (self.yaw_limits[1] - self.yaw_limits[0]) ) + self.yaw_limits[0]

        elif skill_name == "push_y":
            params[0] = ( ((params[0] + 1) / 2 ) * (self.workspace_limits["x"][1] - self.workspace_limits["x"][0]) ) + self.workspace_limits["x"][0]
            params[1] = ( ((params[1] + 1) / 2 ) * (self.workspace_limits["y"][1] - self.workspace_limits["y"][0]) ) + self.workspace_limits["y"][0]
            params[2] = ( ((params[2] + 1) / 2 ) * (self.workspace_limits["z"][1] - self.workspace_limits["z"][0]) ) + self.workspace_limits["z"][0]
            params[3] = ( ((params[3] + 1) / 2 ) * (self.workspace_limits["y"][1] - self.workspace_limits["y"][0]) ) + self.workspace_limits["y"][0]
            if self.use_yaw:
                params[4] = ( ((params[6] + 1) / 2 ) * (self.yaw_limits[1] - self.yaw_limits[0]) ) + self.yaw_limits[0]

        else:
            params[0] = ( ((params[0] + 1) / 2 ) * (self.workspace_limits["x"][1] - self.workspace_limits["x"][0]) ) + self.workspace_limits["x"][0]
            params[1] = ( ((params[1] + 1) / 2 ) * (self.workspace_limits["y"][1] - self.workspace_limits["y"][0]) ) + self.workspace_limits["y"][0]
            params[2] = ( ((params[2] + 1) / 2 ) * (self.workspace_limits["z"][1] - self.workspace_limits["z"][0]) ) + self.workspace_limits["z"][0]
            if self.use_yaw:
                params[3] = ( ((params[3] + 1) / 2 ) * (self.yaw_limits[1] - self.yaw_limits[0]) ) + self.yaw_limits[0]

        return np.concatenate([action[:self.num_skills], params])

    def _move_to(self, params, pos_thresh=0.01, axis_angle_thresh=0.15, count_steps=True, return_n_steps=False):
        """
        Moves end effector to goal position and yaw

        Args: 
            params (7-tuple of floats) : [goal_pos, goal_axis_angle, gripper_state]
            pos_thresh (float) : how close to target position end effector must be in each axis for success
            axis_angle_thresh (float) : how close to target axis angle orientation end effector must be in each axis for success
            
        Returns: 
            action (7d array) : low level action command for given timestep
            skill_done (bool) : whether skill execution finished (goal position and orientation reached or max number of steps reached)
            skill_success (bool) : whether skill execution successfully completed
        """

        max_steps = self.skills["move_to"]["max_steps"]

        skill_success = False
        skill_done = False
        n_steps = self.steps

        # extract params
        target_pos = params[:3]
        if self.use_yaw:
            target_yaw = params[3]
        else:
            target_yaw = 0.0
        gripper_action = params[4]

        # how much to scale action
        pos_scale, ori_scale = 18, 1
        pos_near = 0.025

        while self.robot_interface.state_buffer_size == 0:
            logger.warn("Robot state not received")
            time.sleep(0.5)

        # get current state
        current_eef_pose = self.robot_interface.last_eef_pose
        current_pos = current_eef_pose[:3, 3:]
        current_rot = current_eef_pose[:3, :3]
        # current_quat = transform_utils.mat2quat(current_rot)
        current_euler = transform_utils.mat2euler(current_rot)
        current_yaw = current_euler[-1]

        # pos / ori error
        pos_error = target_pos - current_pos.flatten()
        yaw_error = target_yaw - current_yaw

        # reached goal pos / ori ?
        pos_reached = np.all(np.abs(pos_error) < pos_thresh)
        yaw_reached = np.abs(yaw_error) < axis_angle_thresh

        # print("current pos", current_pos.flatten())
        # print("pos, yaw reached", pos_reached, yaw_reached)
        # print("pos error", pos_error)
        # print("target yaw", target_yaw)
        # print("current yaw", current_yaw)
        # print("yaw error", yaw_error)
        # print("step", self.steps)
        # print('\n')

        # skill is done with success if pos and ori goals are reached
        if self.use_yaw:
            goal_reached = pos_reached and yaw_reached
        else:
            goal_reached = pos_reached

        if goal_reached:
            action = np.zeros(7)
            action[-1] = gripper_action
            skill_done = True
            skill_success = True
            if count_steps:
                self.steps = 0

        # skill is done with failure if number of steps exceed maximum allowed number of steps
        elif count_steps and self.steps > max_steps:
            action = np.zeros(7)
            action[-1] = gripper_action
            skill_done = True
            skill_success = False
            if count_steps:
                self.steps = 0
                print(f"max steps {max_steps} for move_to reached")

        else: # skill is not done yet - compute action

            # steady speed
            action_pos = pos_error / max(np.abs(pos_error))
            action_pos = np.zeros(3)
            action_axis_angle = np.zeros(3)

            if not pos_reached:
                pos_error_norm = np.linalg.norm(pos_error)
                action_scale = 1.0 # 0.6
                if pos_error_norm < pos_near: # slow down when near target
                    action_scale = 0.8 # 0.45
                action_pos = action_scale * pos_error / pos_error_norm

            if not yaw_reached:
                action_axis_angle = np.array([0, 0, yaw_error])
                action_axis_angle = np.clip(action_axis_angle, -0.5, 0.5)

            action = action_pos.tolist() + action_axis_angle.tolist() + [gripper_action]

        # print("action", action, "\n")

        if count_steps:
            self.steps += 1

        if return_n_steps:
            return action, skill_done, skill_success, n_steps
        
        return action, skill_done, skill_success


    # def _move_to(self, params, pos_thresh=0.01, axis_angle_thresh=0.15, count_steps=True, return_n_steps=False):
    #     """
    #     Moves end effector to goal position and yaw

    #     Args: 
    #         params (7-tuple of floats) : [goal_pos, goal_axis_angle, gripper_state]
    #         pos_thresh (float) : how close to target position end effector must be in each axis for success
    #         axis_angle_thresh (float) : how close to target axis angle orientation end effector must be in each axis for success
            
    #     Returns: 
    #         action (7d array) : low level action command for given timestep
    #         skill_done (bool) : whether skill execution finished (goal position and orientation reached or max number of steps reached)
    #         skill_success (bool) : whether skill execution successfully completed
    #     """

    #     max_steps = self.skills["move_to"]["max_steps"]

    #     skill_success = False
    #     skill_done = False
    #     n_steps = self.steps

    #     # extract params
    #     target_pos = params[:3]
    #     if self.use_yaw:
    #         target_yaw = params[3]
    #     else:
    #         target_yaw = 0.0
    #     gripper_action = params[4]

    #     # how much to scale action
    #     pos_scale, ori_scale = 18, 1
    #     pos_near = 0.025

    #     while self.robot_interface.state_buffer_size == 0:
    #         logger.warn("Robot state not received")
    #         time.sleep(0.5)

    #     # get current state
    #     current_eef_pose = self.robot_interface.last_eef_pose
    #     current_pos = current_eef_pose[:3, 3:]
    #     current_rot = current_eef_pose[:3, :3]
    #     # current_quat = transform_utils.mat2quat(current_rot)
    #     current_euler = transform_utils.mat2euler(current_rot)
    #     current_yaw = current_euler[-1]

    #     # pos / ori error
    #     pos_error = target_pos - current_pos.flatten()
    #     yaw_error = target_yaw - current_yaw

    #     # reached goal pos / ori ?
    #     pos_reached = np.all(np.abs(pos_error) < pos_thresh)
    #     yaw_reached = np.abs(yaw_error) < axis_angle_thresh

    #     # print("current pos", current_pos.flatten())
    #     # print("pos, yaw reached", pos_reached, yaw_reached)
    #     # print("pos error", pos_error)
    #     # print("target yaw", target_yaw)
    #     # print("current yaw", current_yaw)
    #     # print("yaw error", yaw_error)
    #     # print("step", self.steps)
    #     # print('\n')

    #     # skill is done with success if pos and ori goals are reached
    #     if self.use_yaw:
    #         goal_reached = pos_reached and yaw_reached
    #     else:
    #         goal_reached = pos_reached

    #     if goal_reached:
    #         action = np.zeros(7)
    #         action[-1] = gripper_action
    #         skill_done = True
    #         skill_success = True
    #         if count_steps:
    #             self.steps = 0

    #     # skill is done with failure if number of steps exceed maximum allowed number of steps
    #     elif count_steps and self.steps > max_steps:
    #         action = np.zeros(7)
    #         action[-1] = gripper_action
    #         skill_done = True
    #         skill_success = False
    #         if count_steps:
    #             self.steps = 0
    #             print(f"max steps {max_steps} for move_to reached")

    #     else: # skill is not done yet - compute action

    #         # steady speed
    #         action_pos = pos_error / max(np.abs(pos_error))
    #         action_pos = np.zeros(3)
    #         action_axis_angle = np.zeros(3)

    #         if not pos_reached:
    #             pos_error_norm = np.linalg.norm(pos_error)
    #             action_scale = 1.0 # 0.6
    #             if pos_error_norm < pos_near: # slow down when near target
    #                 action_scale = 0.8 # 0.45
    #             action_pos = action_scale * pos_error / pos_error_norm

    #         if not yaw_reached:
    #             action_axis_angle = np.array([0, 0, yaw_error])
    #             action_axis_angle = np.clip(action_axis_angle, -0.5, 0.5)

    #         action = action_pos.tolist() + action_axis_angle.tolist() + [gripper_action]

    #     # print("action", action, "\n")

    #     if count_steps:
    #         self.steps += 1

    #     if return_n_steps:
    #         return action, skill_done, skill_success, n_steps
        
    #     return action, skill_done, skill_success

    def _gripper_release(self, params={}, count_steps=True, return_n_steps=False):
        """
        Opens gripper in place

        Args:
            params : placeholder

        Returns: 
            action (7d array) : low level action command for given timestep
            skill_done (bool) : whether skill execution finished (max number of steps reached)
            skill_success (bool) : whether skill execution successfully completed 
        """

        max_steps = self.skills["gripper_release"]["max_steps"]
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        skill_done = False
        skill_success = False
        n_steps = self.steps

        if self.grip_steps >= max_steps:
            self.grip_steps = 0
            if count_steps:
                self.steps = 0
            skill_done = True
            skill_success = True

        self.grip_steps += 1

        if count_steps:
            self.steps += 1

        if return_n_steps:
            return action, skill_done, skill_success, n_steps

        return action, skill_done, skill_success

    def _gripper_close(self, params={}, count_steps=True, return_n_steps=False):
        """
        Closes gripper in place

        Args:
            params : placeholder 

        Returns: 
            action (7d array) : low level action command for given timestep
            skill_done (bool) : whether skill execution finished (max number of steps reached)
            skill_success (bool) : whether skill execution successfully completed 
        """

        max_steps = self.skills["gripper_release"]["max_steps"]
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        skill_done = False
        skill_success = False
        n_steps = self.steps

        if self.grip_steps >= max_steps:
            self.grip_steps = 0
            if count_steps:
                self.steps = 0
            skill_done = True
            skill_success = True

        self.grip_steps += 1

        if count_steps:
            self.steps += 1

        if return_n_steps:
            return action, skill_done, skill_success, n_steps

        return action, skill_done, skill_success

    def _pick(self, params, pos_thresh=0.02, axis_angle_thresh=0.1, return_n_steps=False):
        """
        Picks up an object at specified position and orientation, then rehomes

        Args: 
            params (6-tuple of floats) : [goal_pos, goal_axis_angle]
            pos_thresh (float) : how close to target position end effector must be in each axis for success
            axis_angle_thresh (float) : how close to target axis angle orientation end effector must be in each axis for success
            
        Returns: 
            action (7d array) : low level action command for given timestep
            skill_done (bool) : whether skill execution finished (goal position and orientation reached or max number of steps reached)
            skill_success (bool) : whether skill execution successfully completed
        """
        # print("phase",self.phase)
        max_steps = self.skills["pick"]["max_steps"]
        skill_done, skill_success = False, False
        n_steps = self.steps

        # extract params
        pick_pos = params[:3]
        if self.use_yaw:
            goal_yaw = params[3]
        else:  
            goal_yaw = 0.0

        # define params for waypoints (intermediate goal poses)
        above_wp_pose = np.concatenate([np.array([pick_pos[0], pick_pos[1], self.waypoint_height]), [goal_yaw]])
        pick_wp_pose = np.concatenate([pick_pos, [goal_yaw]])

        if self.prev_success:
            self.phase += 1
            self.prev_success = False
            time.sleep(0.5)

        # if max steps reached, rehome and terminate
        if self.steps > max_steps:
            action = np.zeros(7)
            skill_done, skill_success = True, False
            self.phase = 0
            self.steps = 0
            self.prev_success = False
            self._rehome(gripper_action=-1.0)
            print(f"max steps reached for pick {max_steps}")

            if return_n_steps:
                return action, skill_done, skill_success, n_steps
            return action, True, False

        # phase 0 : move to above pick position
        if self.phase == 0:
            params = np.concatenate([above_wp_pose, [-1]])
            action, _, self.prev_success = self._move_to(params=params, pos_thresh=pos_thresh, axis_angle_thresh=axis_angle_thresh, count_steps=False)

        # phase 1 : move down to pick position
        if self.phase == 1:
            params = np.concatenate([pick_wp_pose, [-1]])
            action, _, self.prev_success = self._move_to(params=params, pos_thresh=0.01, axis_angle_thresh=axis_angle_thresh, count_steps=False)

        # phase 2 : grip
        if self.phase == 2:
            action, _, self.prev_success = self._gripper_close(count_steps=False)

        # phase 3 : move up to above pick position
        if self.phase == 3:
            params = np.concatenate([above_wp_pose, [1]])
            action, _, self.prev_success = self._move_to(params=params, pos_thresh=pos_thresh, axis_angle_thresh=axis_angle_thresh, count_steps=False)

        # phase 4 : rehome after success
        if self.phase == 4:
            action = np.zeros(7)
            action[-1] = 1.0
            skill_done, skill_success = True, True
            self.phase = 0
            self.steps = 0
            self.prev_success = False
            print("pick success - rehoming")
            self._rehome(gripper_action=1.0)
            if return_n_steps:
                return action, skill_done, skill_success, n_steps
            return action, skill_done, skill_success

        self.steps += 1
        
        if return_n_steps:
            return action, skill_done, skill_success, n_steps

        return action, skill_done, skill_success

    def _place(self, params, pos_thresh=0.015, axis_angle_thresh=0.1, return_n_steps=False):
        """
        Places an object at specified position and orientation, then rehomes

        Args: 
            params (6-tuple of floats) : [goal_pos, goal_axis_angle]
            pos_thresh (float) : how close to target position end effector must be in each axis for success
            axis_angle_thresh (float) : how close to target axis angle orientation end effector must be in each axis for success
            
        Returns: 
            action (7d array) : low level action command for given timestep
            skill_done (bool) : whether skill execution finished (goal position and orientation reached or max number of steps reached)
            skill_success (bool) : whether skill execution successfully completed
        """

        max_steps = self.skills["place"]["max_steps"]
        skill_done, skill_success = False, False
        n_steps = self.steps

        # extract params
        place_pos = params[:3]
        if self.use_yaw:
            goal_yaw = params[3]
        else:
            goal_yaw = 0.0

        # define params for waypoints (intermediate goal poses)
        above_wp_pose = np.concatenate([np.array([place_pos[0], place_pos[1], self.waypoint_height]), [goal_yaw]])
        place_wp_pose = np.concatenate([place_pos, [goal_yaw]])

        if self.prev_success:
            self.phase += 1
            self.prev_success = False
            time.sleep(0.5)

        # if max steps reached, rehome and terminate
        if self.steps > max_steps:
            action = np.zeros(7)
            skill_done, skill_success = True, False
            self.phase = 0
            self.steps = 0
            self.prev_success = False
            self._rehome(gripper_action=-1.0)
            print(f"max steps reached for place {max_steps}")
            if return_n_steps:
                return action, skill_done, skill_success, n_steps
            return action, True, False

        # phase 0 : move to above place position
        if self.phase == 0:
            params = np.concatenate([above_wp_pose, [1]])
            action, _, self.prev_success = self._move_to(params=params, pos_thresh=0.015, axis_angle_thresh=axis_angle_thresh, count_steps=False)

        # phase 1 : move down to place position
        if self.phase == 1:
            params = np.concatenate([place_wp_pose, [1]])
            action, _, self.prev_success = self._move_to(params=params, pos_thresh=pos_thresh, axis_angle_thresh=axis_angle_thresh, count_steps=False)

        # phase 2 : release
        if self.phase == 2:
            action, _, self.prev_success = self._gripper_release(count_steps=False)

        # phase 3 : move up to above pick position
        if self.phase == 3:
            params = np.concatenate([above_wp_pose, [-1]])
            action, _, self.prev_success = self._move_to(params=params, pos_thresh=0.015, axis_angle_thresh=axis_angle_thresh, count_steps=False)

        # phase 4 : rehome after success
        if self.phase == 4:
            action = np.zeros(7)
            skill_done, skill_success = True, True
            self.phase = 0
            self.steps = 0
            self.prev_success = False
            print("place success - rehoming")
            self._rehome(gripper_action=-1.0)
            if return_n_steps:
                return action, skill_done, skill_success, n_steps
            return action, skill_done, skill_success

        self.steps += 1

        if return_n_steps:
            return action, skill_done, skill_success, n_steps
        
        return action, skill_done, skill_success

    def _push(self, params, pos_thresh=0.01, axis_angle_thresh=0.1, return_n_steps=False):
        """
        Executes pushing action given start position, end position, orientation to maintain, and gripper action.
        Rehomes after execution

        Args: 
            params (10-tuple of floats) : [start_pos, end_pos, goal_axis_angle, gripper_action]
            pos_thresh (float) : how close to target position end effector must be in each axis for success
            axis_angle_thresh (float) : how close to target axis angle orientation end effector must be in each axis for success
            
        Returns: 
            action (7d array) : low level action command for given timestep
            skill_done (bool) : whether skill execution finished (goal position and orientation reached or max number of steps reached)
            skill_success (bool) : whether skill execution successfully completed
        """

        max_steps = self.skills["push"]["max_steps"]
        skill_done, skill_success = False, False
        n_steps = self.steps

        # extract params
        push_start_pos = params[:3]
        push_end_pos = params[3:6]
        if self.use_yaw:
            goal_yaw = params[6]
            gripper_action = 1 if params[7] > 0 else -1
        else:
            goal_yaw = 0.0
            gripper_action = 1 if params[6] > 0 else -1

        # define params for waypoints (intermediate goal poses)
        above_start_pose = np.concatenate([np.array([push_start_pos[0], push_start_pos[1], self.waypoint_height]), [goal_yaw]])
        start_pose = np.concatenate([push_start_pos, [goal_yaw]])
        end_pose = np.concatenate([push_end_pos, [goal_yaw]])
        above_end_pose = np.concatenate([np.array([push_end_pos[0], push_end_pos[1], self.waypoint_height]), [goal_yaw]])

        if self.prev_success:
            self.phase += 1
            self.prev_success = False
            time.sleep(0.5)

        # print("phase", self.phase)
        # if max steps reached, rehome and terminate
        if self.steps > max_steps:
            action = np.zeros(7)
            skill_done, skill_success = True, False
            self.phase = 0
            self.steps = 0
            print(f"max steps reached for push {max_steps}")
            self._rehome(gripper_action=-1.0)
            if return_n_steps:
                return action, skill_done, skill_success, n_steps
            return action, True, False

        # phase 0 : move to above start position
        if self.phase == 0:
            params = np.concatenate([above_start_pose, [gripper_action]])
            action, _, self.prev_success = self._move_to(params=params, pos_thresh=pos_thresh, axis_angle_thresh=axis_angle_thresh, count_steps=False)

        # phase 1 : move down to start position
        if self.phase == 1:
            params = np.concatenate([start_pose, [gripper_action]])
            action, _, self.prev_success = self._move_to(params=params, pos_thresh=pos_thresh, axis_angle_thresh=axis_angle_thresh, count_steps=False)

        # phase 2 : move to end position
        if self.phase == 2:
            params = np.concatenate([end_pose, [gripper_action]])
            action, _, self.prev_success = self._move_to(params=params, pos_thresh=pos_thresh, axis_angle_thresh=axis_angle_thresh, count_steps=False)

        # phase 3 : move to end position
        if self.phase == 3:
            params = np.concatenate([above_end_pose, [gripper_action]])
            action, _, self.prev_success = self._move_to(params=params, pos_thresh=pos_thresh, axis_angle_thresh=axis_angle_thresh, count_steps=False)

        # phase 4 : rehome after success
        if self.phase == 4:
            action = np.zeros(7)
            skill_done, skill_success = True, True
            self.phase = 0
            self.steps = 0
            self.prev_success = False
            self._rehome(gripper_action=-1.0)
            if return_n_steps:
                return action, skill_done, skill_success, n_steps
            return action, skill_done, skill_success

        self.steps += 1

        if return_n_steps:
            return action, skill_done, skill_success, n_steps
        
        return action, skill_done, skill_success

    def _push_x(self, params, pos_thresh=0.02, axis_angle_thresh=0.1, return_n_steps=False):
        
        """
        Executes pushing action given start position, end position, orientation to maintain, and gripper action.
        Rehomes after execution

        Args: 
            params (10-tuple of floats) : [start_pos, end_pos, goal_axis_angle, gripper_action]
            pos_thresh (float) : how close to target position end effector must be in each axis for success
            axis_angle_thresh (float) : how close to target axis angle orientation end effector must be in each axis for success
            
        Returns: 
            action (7d array) : low level action command for given timestep
            skill_done (bool) : whether skill execution finished (goal position and orientation reached or max number of steps reached)
            skill_success (bool) : whether skill execution successfully completed
        """

        max_steps = self.skills["push"]["max_steps"]
        skill_done, skill_success = False, False
        n_steps = self.steps

        # extract params
        push_start_pos = params[:3]
        push_delta = params[3]
        if self.use_yaw:
            goal_yaw = params[4]
            # gripper_action = 1 if params[5] > 0 else -1
        else:
            goal_yaw = 0.0
            # gripper_action = 1 if params[4] > 0 else -1
        
        gripper_action = 1
        push_end_pos = push_start_pos.copy()
        push_end_pos[0] += push_delta

        # define params for waypoints (intermediate goal poses)
        above_start_pose = np.concatenate([np.array([push_start_pos[0], push_start_pos[1], self.waypoint_height]), [goal_yaw]])
        start_pose = np.concatenate([push_start_pos, [goal_yaw]])
        end_pose = np.concatenate([push_end_pos, [goal_yaw]])
        above_end_pose = np.concatenate([np.array([push_end_pos[0], push_end_pos[1], self.waypoint_height]), [goal_yaw]])

        if self.prev_success:
            self.phase += 1
            self.prev_success = False
            time.sleep(0.5)

        # print("phase", self.phase)
        # if max steps reached, rehome and terminate
        if self.steps > max_steps:
            action = np.zeros(7)
            action[-1] = 1.0
            skill_done, skill_success = True, False
            self.phase = 0
            self.steps = 0
            print(f"max steps reached for push {max_steps}")
            self._rehome(gripper_action=1.0)
            if return_n_steps:
                return action, skill_done, skill_success, n_steps
            return action, True, False

        # phase 0 : move to above start position
        if self.phase == 0:
            params = np.concatenate([above_start_pose, [gripper_action]])
            action, _, self.prev_success = self._move_to(params=params, pos_thresh=pos_thresh, axis_angle_thresh=axis_angle_thresh, count_steps=False)

        # phase 1 : move down to start position
        if self.phase == 1:
            params = np.concatenate([start_pose, [gripper_action]])
            action, _, self.prev_success = self._move_to(params=params, pos_thresh=pos_thresh, axis_angle_thresh=axis_angle_thresh, count_steps=False)

        # phase 2 : move to end position
        if self.phase == 2:
            params = np.concatenate([end_pose, [gripper_action]])
            action, _, self.prev_success = self._move_to(params=params, pos_thresh=pos_thresh, axis_angle_thresh=axis_angle_thresh, count_steps=False)

        # phase 3 : move to end position
        if self.phase == 3:
            params = np.concatenate([above_end_pose, [gripper_action]])
            action, _, self.prev_success = self._move_to(params=params, pos_thresh=pos_thresh, axis_angle_thresh=axis_angle_thresh, count_steps=False)

        # phase 4 : rehome after success
        if self.phase == 4:
            action = np.zeros(7)
            action[-1] = 1.0
            skill_done, skill_success = True, True
            self.phase = 0
            self.steps = 0
            self.prev_success = False
            self._rehome(gripper_action=1.0)
            if return_n_steps:
                return action, skill_done, skill_success, n_steps
            return action, skill_done, skill_success

        self.steps += 1

        if return_n_steps:
            return action, skill_done, skill_success, n_steps
        
        return action, skill_done, skill_success

    def _push_y(self, params, pos_thresh=0.02, axis_angle_thresh=0.1, return_n_steps=False):
        
        """
        Executes pushing action given start position, end position, orientation to maintain, and gripper action.
        Rehomes after execution

        Args: 
            params (10-tuple of floats) : [start_pos, end_pos, goal_axis_angle, gripper_action]
            pos_thresh (float) : how close to target position end effector must be in each axis for success
            axis_angle_thresh (float) : how close to target axis angle orientation end effector must be in each axis for success
            
        Returns: 
            action (7d array) : low level action command for given timestep
            skill_done (bool) : whether skill execution finished (goal position and orientation reached or max number of steps reached)
            skill_success (bool) : whether skill execution successfully completed
        """
        max_steps = self.skills["push"]["max_steps"]
        skill_done, skill_success = False, False
        n_steps = self.steps

        # extract params
        push_start_pos = params[:3]
        push_delta = params[3]
        if self.use_yaw:
            goal_yaw = params[4]
            # gripper_action = 1 if params[5] > 0 else -1
        else:
            goal_yaw = 0.0
            # gripper_action = 1 if params[4] > 0 else -1

        gripper_action = -1
        push_end_pos = push_start_pos.copy()
        push_end_pos[1] += push_delta

        # define params for waypoints (intermediate goal poses)
        above_start_pose = np.concatenate([np.array([push_start_pos[0], push_start_pos[1], self.waypoint_height]), [goal_yaw]])
        start_pose = np.concatenate([push_start_pos, [goal_yaw]])
        end_pose = np.concatenate([push_end_pos, [goal_yaw]])
        above_end_pose = np.concatenate([np.array([push_end_pos[0], push_end_pos[1], self.waypoint_height]), [goal_yaw]])

        if self.prev_success:
            self.phase += 1
            self.prev_success = False
            time.sleep(0.5)

        # print("phase", self.phase)
        # if max steps reached, rehome and terminate
        if self.steps > max_steps:
            action = np.zeros(7)
            skill_done, skill_success = True, False
            self.phase = 0
            self.steps = 0
            print(f"max steps reached for push {max_steps}")
            self._rehome(gripper_action=-1.0)
            if return_n_steps:
                return action, skill_done, skill_success, n_steps
            return action, True, False

        # phase 0 : move to above start position
        if self.phase == 0:
            params = np.concatenate([above_start_pose, [gripper_action]])
            action, _, self.prev_success = self._move_to(params=params, pos_thresh=pos_thresh, axis_angle_thresh=axis_angle_thresh, count_steps=False)

        # phase 1 : move down to start position
        if self.phase == 1:
            params = np.concatenate([start_pose, [gripper_action]])
            action, _, self.prev_success = self._move_to(params=params, pos_thresh=pos_thresh, axis_angle_thresh=axis_angle_thresh, count_steps=False)

        # phase 2 : move to end position
        if self.phase == 2:
            params = np.concatenate([end_pose, [gripper_action]])
            action, _, self.prev_success = self._move_to(params=params, pos_thresh=pos_thresh, axis_angle_thresh=axis_angle_thresh, count_steps=False)

        # phase 3 : move to end position
        if self.phase == 3:
            params = np.concatenate([above_end_pose, [gripper_action]])
            action, _, self.prev_success = self._move_to(params=params, pos_thresh=pos_thresh, axis_angle_thresh=axis_angle_thresh, count_steps=False)

        # phase 4 : rehome after success
        if self.phase == 4:
            action = np.zeros(7)
            skill_done, skill_success = True, True
            self.phase = 0
            self.steps = 0
            self.prev_success = False
            self._rehome(gripper_action=-1.0)
            if return_n_steps:
                return action, skill_done, skill_success, n_steps
            return action, skill_done, skill_success

        self.steps += 1

        if return_n_steps:
            return action, skill_done, skill_success, n_steps
        
        return action, skill_done, skill_success

    def _rehome(self, gripper_action=-1.0):
        print(f"---------rehoming with gripper {gripper_action}------------")
        params = np.concatenate([self.reset_eef_pos, [self.reset_yaw], [gripper_action]])
        skill_done = False
        rehome_steps = 0

        while not skill_done:
            if rehome_steps > 200:
                print("WARNING: REHOMING FAILED")
                break
            action, skill_done, skill_success = self._move_to(params=params, pos_thresh=0.01, count_steps=False)
            self.robot_interface.control(
                controller_type=self.controller_type,
                controller_cfg=self.controller_config,
                action=action,
            )

            # print("second stage", action)
            rehome_steps += 1
        rehome_steps = 0


# version with yaw as input
# class PrimitiveSkill:

#     def __init__(
#         self,
#         # interface_config="config/charmander.yml",
#         controller_type,
#         controller_config,
#         robot_interface,
#         waypoint_height=0.25,
#         workspace_limits=None,
#         yaw_limits=None,
#         idx2skill=None,
#         aff_pos_thresh=None,
#     ):

#         """
#         Args: 
#             interface_config (str) : path to robot interface config yaml file
#             controller_type (str) : name of controller. defaults to OSC_POSE
#             waypoint_height (float) : height of waypoint used in multi-step skills (pick, place, push)
#             workspace_limits (dict) : {"x" : (x lower limit, x upper limit), "y" : (y lower limit, y upper limit), "z" : (z lower limit, z upper limit) }
#             yaw_limits (tuple of floats) : (yaw lower limit, yaw upper limit)
#             idx2skill (dict) : maps skill index (int) in one-hot skill-selection action vector to skill name (str). If unspecified, uses default settings
#             aff_pos_thresh (dict) : maps skill name (str) to reaching position threshold for computing affordance scores. If unspecified, uses default settings
#         """

#         self.robot_interface = robot_interface
#         self.controller_type = controller_type
#         self.controller_config = controller_config

#         # robot home position, waypoint height and workspace / yaw limits
#         self.reset_joint_positions = [
#             0.09162008114028396,
#             -0.19826458111314524,
#             -0.01990020486871322,
#             -2.4732269941140346,
#             -0.01307073642274261,
#             2.30396583422025,
#             0.8480939705504309,
#         ]
#         self.reset_eef_pos = [0.45775618, 0.03207872, 0.26534091]
#         self.reset_axis_angle_ori = [np.pi, 0.0, 0.0]
#         self.reset_yaw = 0.0
        
#         self.waypoint_height = waypoint_height # height of waypoint in pick, place, push skills
#         if workspace_limits is not None:
#             self.workspace_limits = workspace_limits
#         else: 
#             self.workspace_limits = {
#                 "x" : (0.3, 0.55),
#                 "y" : (-0.15, 0.25),
#                 "z" : (0.03, 0.30)
#             }
#         if yaw_limits is not None:
#             self.yaw_limits = yaw_limits
#         else:
#             self.yaw_limits = (-0.5*np.pi, 0.5*np.pi)

#         # skill settings 
#         self.skills = {
#             "move_to" : {
#                 "max_steps" : 200,
#                 "num_params" : 5,
#                 "skill" : self._move_to,
#                 "default_idx" : 0,
#                 "default_aff_thresh" : 0.05,
#             },

#             "gripper_release" : {
#                 "max_steps" : 25,
#                 "num_params" : 0,
#                 "skill" : self._gripper_release,
#                 "default_idx" : 1,
#                 "default_aff_thresh" : None,
#             },

#             "gripper_close" : {
#                 "max_steps" : 25,
#                 "num_params" : 0,
#                 "skill" : self._gripper_close,
#                 "default_idx" : 2,
#                 "default_aff_thresh" : None,
#             },

#             "pick" : {
#                 "max_steps" : 300,
#                 "num_params" : 4,
#                 "skill" : self._pick,
#                 "default_idx" : 3,
#                 "default_aff_thresh" : 0.03,
#             },

#             "place" : {
#                 "max_steps" : 300,
#                 "num_params" : 4,
#                 "skill" : self._place,
#                 "default_idx" : 4,
#                 "default_aff_thresh" : 0.03,
#             },

#             "push" : {
#                 "max_steps" : 300,
#                 "num_params" : 8,
#                 "skill" : self._push,
#                 "default_idx" : 5,
#                 "default_aff_thresh" : 0.1,
#             },
#         }

#         if idx2skill is None:
#             self.idx2skill = { skill["default_idx"] : name for name, skill in self.skills.items()}
#         else:
#             for name in idx2skill.values():
#                 assert name in self.skills.keys(), f"Error with skill {name}. Skill name must be one of {self.skills.keys()}"
#             self.idx2skill = idx2skill    
    
#         self.num_skills = len(self.idx2skill)
#         self.max_num_params = max([self.skills[skill_name]["num_params"] for skill_name in self.idx2skill.values()])
#         # self.max_num_params = max([self.idx2skill[]["num_params"] for skill_name in self.skills.keys()])

#         self.steps = 0 # number of steps spent so far on this skill
#         self.grip_steps = 0 # number of consequtive steps spent on gripper close/releaes
#         self.phase = 0 # keeps track of which phase the skill is in for multiple-step skills (pick, place, push)
#         self.prev_success = False # whether previous phase succeeded 

#         # affordance score settings
#         if aff_pos_thresh is not None:
#             assert all(key in aff_pos_thresh.keys() for key in self.idx2skill.values())
#             self.aff_pos_thresh = aff_pos_thresh
#         else:
#             self.aff_pos_thresh = { name : skill["default_aff_thresh"] for name, skill in self.skills.items() }

#         self.aff_tanh_scaling = 1.0

#     def get_keypoints_dict(self):
#         """
#         Return dictionary with skill names as keys and all values None. Used for affordance reward calculation
#         """
#         keypoints = {key : None for key in self.idx2skill.values()}
#         return keypoints
    
#     def compute_affordance_reward(self, action, keypoint_dict):
#         """
#         Computes afforance reward given action and keypoints

#         Args:
#             action (array): action
#             keypoint_dict (dict) : maps skill name to keypoints. keypoints can be None or list of coordinates
#                 "None" indicates that the skill is relevant at any position (choosing this skill is never penalized regardless of position parameters)
#                 Empty list indicates that the skill is not relevant at any position (choosing this skill is always penalized regardless of position parameters)

#         Returns:
#             affordance_reward (float) : affordance reward for choosing given action
#         """
#         skill_idx = np.argmax(action[:self.n_skills])
#         skill_name = self.idx2skill[skill_idx]
#         keypoints = keypoint_dict[skill_name] # keypoints for chosen skill
#         reach_pos = action[self.n_skills:self.n_skills + 3] # component of params corresponding to reach position
#         if keypoints is None:
#             return 1.0

#         if len(keypoints) == 0:
#             return 0.0

#         aff_centers = np.stack(keypoints)
#         dist = np.clip(np.abs(aff_centers - reach_pos) - self.aff_pos_thresh[skill_name], 0, None)
#         min_dist = np.min(np.sum(dist, axis=1))
#         aff_reward = 1.0 - np.tanh(self.aff_tanh_scaling * min_dist)
#         return aff_reward

#     def get_action(self, action):
#         """
#         Args:
#             action (array): one-hot vector for skill selection concatenated with skill parameters
#                 one-hot vector dimension must be same as self.n_skills. skill parameter can have variable dimension
#         Returns:
#             action (7d-array): action commands for simulation environment - (position commands, orientation commands, gripper command)    
#             skill_done (bool): True if goal skill completed successfully or if max allowed steps is reached
#             skill_success (bool) : True if skill completed successfully
#         """
#         skill_idx = np.argmax(action[:self.num_skills])
#         skill = self.skills[self.idx2skill[skill_idx]]["skill"]
#         params = action[self.num_skills:]
#         action, skill_done, skill_success = skill(params, return_n_steps=False)

#         return action, skill_done, skill_success

#     def execute_skill(self, action):

#         skill_idx = np.argmax(action[:self.num_skills])
#         skill = self.skills[self.idx2skill[skill_idx]]["skill"]
#         params = action[self.num_skills:]
#         skill_done = False
        
#         while not skill_done:
#             # print("step", self.steps)
#             action_ll, skill_done, skill_success, n_steps = skill(
#                 params=params,
#                 return_n_steps=True,
#             )

#             if skill_done:
#                 break 

#             # print("action ll", action_ll)
#             self.robot_interface.control(
#                 controller_type=self.controller_type,
#                 action=action_ll,
#                 controller_cfg=self.controller_config,
#             )

#         self.steps = 0
#         return skill_success, n_steps - 1

#     def unnormalize_params(self, action):
#         """
#         Unnormalizes parameters from [-1, 1] to raw values

#         Args:
#             action : one-hot skill selection vector concatenated with params

#         Returns: 
#             action : action with unnormalized params
#         """
#         # find out which skill is called
#         action = action.copy()
#         skill_idx = np.argmax(action[:self.num_skills])
#         skill_name = self.idx2skill[skill_idx]
#         params = action[self.num_skills:]
#         if skill_name == "push":
#             params[0] = ( ((params[0] + 1) / 2 ) * (self.workspace_limits["x"][1] - self.workspace_limits["x"][0]) ) + self.workspace_limits["x"][0]
#             params[1] = ( ((params[1] + 1) / 2 ) * (self.workspace_limits["y"][1] - self.workspace_limits["y"][0]) ) + self.workspace_limits["y"][0]
#             params[2] = ( ((params[2] + 1) / 2 ) * (self.workspace_limits["z"][1] - self.workspace_limits["z"][0]) ) + self.workspace_limits["z"][0]
#             params[3] = ( ((params[3] + 1) / 2 ) * (self.workspace_limits["x"][1] - self.workspace_limits["x"][0]) ) + self.workspace_limits["x"][0]
#             params[4] = ( ((params[4] + 1) / 2 ) * (self.workspace_limits["y"][1] - self.workspace_limits["y"][0]) ) + self.workspace_limits["y"][0]
#             params[5] = ( ((params[5] + 1) / 2 ) * (self.workspace_limits["z"][1] - self.workspace_limits["z"][0]) ) + self.workspace_limits["z"][0]
#             params[6] = ( ((params[6] + 1) / 2 ) * (self.yaw_limits[1] - self.yaw_limits[0]) ) + self.yaw_limits[0]

#         else:
#             params[0] = ( ((params[0] + 1) / 2 ) * (self.workspace_limits["x"][1] - self.workspace_limits["x"][0]) ) + self.workspace_limits["x"][0]
#             params[1] = ( ((params[1] + 1) / 2 ) * (self.workspace_limits["y"][1] - self.workspace_limits["y"][0]) ) + self.workspace_limits["y"][0]
#             params[2] = ( ((params[2] + 1) / 2 ) * (self.workspace_limits["z"][1] - self.workspace_limits["z"][0]) ) + self.workspace_limits["z"][0]
#             params[3] = ( ((params[3] + 1) / 2 ) * (self.yaw_limits[1] - self.yaw_limits[0]) ) + self.yaw_limits[0]

#         return np.concatenate([action[:self.num_skills], params])

#     def _move_to(self, params, pos_thresh=0.01, axis_angle_thresh=0.15, count_steps=True, return_n_steps=False):
#         """
#         Moves end effector to goal position and yaw

#         Args: 
#             params (7-tuple of floats) : [goal_pos, goal_axis_angle, gripper_state]
#             pos_thresh (float) : how close to target position end effector must be in each axis for success
#             axis_angle_thresh (float) : how close to target axis angle orientation end effector must be in each axis for success
            
#         Returns: 
#             action (7d array) : low level action command for given timestep
#             skill_done (bool) : whether skill execution finished (goal position and orientation reached or max number of steps reached)
#             skill_success (bool) : whether skill execution successfully completed
#         """

#         max_steps = self.skills["move_to"]["max_steps"]

#         skill_success = False
#         skill_done = False
#         n_steps = self.steps

#         # extract params
#         target_pos = params[:3]
#         target_yaw = params[3]
#         gripper_action = params[4]

#         # how much to scale action
#         pos_scale, ori_scale = 18, 1
#         pos_near = 0.05

#         while self.robot_interface.state_buffer_size == 0:
#             logger.warn("Robot state not received")
#             time.sleep(0.5)

#         # get current state
#         current_eef_pose = self.robot_interface.last_eef_pose
#         current_pos = current_eef_pose[:3, 3:]
#         current_rot = current_eef_pose[:3, :3]
#         # current_quat = transform_utils.mat2quat(current_rot)
#         current_euler = transform_utils.mat2euler(current_rot)
#         current_yaw = current_euler[-1]

#         # pos / ori error
#         pos_error = target_pos - current_pos.flatten()
#         yaw_error = target_yaw - current_yaw

#         # reached goal pos / ori ?
#         pos_reached = np.all(np.abs(pos_error) < pos_thresh)
#         yaw_reached = np.abs(yaw_error) < axis_angle_thresh

#         # print("current pos", current_pos.flatten())
#         print("pos, yaw reached", pos_reached, yaw_reached)
#         print("pos error", pos_error)
#         # print("target yaw", target_yaw)
#         # print("current yaw", current_yaw)
#         print("yaw error", yaw_error)
#         # print("step", self.steps)
#         # print('\n')

#         # skill is done with success if pos and ori goals are reached
#         if pos_reached and yaw_reached:
#             action = np.zeros(7)
#             action[-1] = gripper_action
#             skill_done = True
#             skill_success = True
#             if count_steps:
#                 self.steps = 0

#         # skill is done with failure if number of steps exceed maximum allowed number of steps
#         elif count_steps and self.steps > max_steps:
#             action = np.zeros(7)
#             action[-1] = gripper_action
#             skill_done = True
#             skill_success = False
#             if count_steps:
#                 self.steps = 0
#                 print(f"max steps {max_steps} for move_to reached")

#         else: # skill is not done yet - compute action

#             # steady speed
#             action_pos = pos_error / max(np.abs(pos_error))
#             action_pos = np.zeros(3)
#             action_axis_angle = np.zeros(3)

#             if not pos_reached:
#                 pos_error_norm = np.linalg.norm(pos_error)
#                 action_scale = 1.0 # 0.6
#                 if pos_error_norm < pos_near: # slow down when near target
#                     action_scale = 0.8 # 0.45
#                 action_pos = action_scale * pos_error / pos_error_norm

#             if not yaw_reached:
#                 action_axis_angle = np.array([0, 0, yaw_error])
#                 action_axis_angle = np.clip(action_axis_angle, -0.5, 0.5)

#             action = action_pos.tolist() + action_axis_angle.tolist() + [gripper_action]

#         # print("action", action, "\n")

#         if count_steps:
#             self.steps += 1

#         if return_n_steps:
#             return action, skill_done, skill_success, n_steps
        
#         return action, skill_done, skill_success

#     def _gripper_release(self, params={}, count_steps=True, return_n_steps=False):
#         """
#         Opens gripper in place

#         Args:
#             params : placeholder

#         Returns: 
#             action (7d array) : low level action command for given timestep
#             skill_done (bool) : whether skill execution finished (max number of steps reached)
#             skill_success (bool) : whether skill execution successfully completed 
#         """

#         max_steps = self.skills["gripper_release"]["max_steps"]
#         action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
#         skill_done = False
#         skill_success = False
#         n_steps = self.steps

#         if self.grip_steps >= max_steps:
#             self.grip_steps = 0
#             if count_steps:
#                 self.steps = 0
#             skill_done = True
#             skill_success = True

#         self.grip_steps += 1

#         if count_steps:
#             self.steps += 1

#         if return_n_steps:
#             return action, skill_done, skill_success, n_steps

#         return action, skill_done, skill_success

#     def _gripper_close(self, params={}, count_steps=True, return_n_steps=False):
#         """
#         Closes gripper in place

#         Args:
#             params : placeholder 

#         Returns: 
#             action (7d array) : low level action command for given timestep
#             skill_done (bool) : whether skill execution finished (max number of steps reached)
#             skill_success (bool) : whether skill execution successfully completed 
#         """

#         max_steps = self.skills["gripper_release"]["max_steps"]
#         action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
#         skill_done = False
#         skill_success = False
#         n_steps = self.steps

#         if self.grip_steps >= max_steps:
#             self.grip_steps = 0
#             if count_steps:
#                 self.steps = 0
#             skill_done = True
#             skill_success = True

#         self.grip_steps += 1

#         if count_steps:
#             self.steps += 1

#         if return_n_steps:
#             return action, skill_done, skill_success, n_steps

#         return action, skill_done, skill_success

#     def _pick(self, params, pos_thresh=0.005, axis_angle_thresh=0.1, return_n_steps=False):
#         """
#         Picks up an object at specified position and orientation, then rehomes

#         Args: 
#             params (6-tuple of floats) : [goal_pos, goal_axis_angle]
#             pos_thresh (float) : how close to target position end effector must be in each axis for success
#             axis_angle_thresh (float) : how close to target axis angle orientation end effector must be in each axis for success
            
#         Returns: 
#             action (7d array) : low level action command for given timestep
#             skill_done (bool) : whether skill execution finished (goal position and orientation reached or max number of steps reached)
#             skill_success (bool) : whether skill execution successfully completed
#         """

#         max_steps = self.skills["pick"]["max_steps"]
#         skill_done, skill_success = False, False
#         n_steps = self.steps

#         # extract params
#         pick_pos = params[:3]
#         # goal_axis_angle = params[3:6]
#         goal_yaw = params[3]

#         # define params for waypoints (intermediate goal poses)
#         above_wp_pose = np.concatenate([np.array([pick_pos[0], pick_pos[1], self.waypoint_height]), [goal_yaw]])
#         pick_wp_pose = np.concatenate([pick_pos, [goal_yaw]])

#         if self.prev_success:
#             self.phase += 1
#             self.prev_success = False
#             time.sleep(0.5)

#         # if max steps reached, rehome and terminate
#         if self.steps > max_steps:
#             action = np.zeros(7)
#             skill_done, skill_success = True, False
#             self.phase = 0
#             self.steps = 0
#             self.prev_success = False
#             self._rehome(gripper_action=-1.0)
#             print(f"max steps reached for pick {max_steps}")
#             if return_n_steps:
#                 return action, skill_done, skill_success, n_steps
#             return action, True, False

#         # phase 0 : move to above pick position
#         if self.phase == 0:
#             params = np.concatenate([above_wp_pose, [-1]])
#             action, _, self.prev_success = self._move_to(params=params, pos_thresh=0.01, axis_angle_thresh=axis_angle_thresh, count_steps=False)

#         # phase 1 : move down to pick position
#         if self.phase == 1:
#             params = np.concatenate([pick_wp_pose, [-1]])
#             action, _, self.prev_success = self._move_to(params=params, pos_thresh=pos_thresh, axis_angle_thresh=axis_angle_thresh, count_steps=False)

#         # phase 2 : grip
#         if self.phase == 2:
#             action, _, self.prev_success = self._gripper_close(count_steps=False)

#         # phase 3 : move up to above pick position
#         if self.phase == 3:
#             params = np.concatenate([above_wp_pose, [1]])
#             action, _, self.prev_success = self._move_to(params=params, pos_thresh=0.01, axis_angle_thresh=axis_angle_thresh, count_steps=False)

#         # phase 4 : rehome after success
#         if self.phase == 4:
#             action = np.zeros(7)
#             skill_done, skill_success = True, True
#             self.phase = 0
#             self.steps = 0
#             self.prev_success = False
#             print("pick success - rehoming")
#             self._rehome(gripper_action=1.0)
#             if return_n_steps:
#                 return action, skill_done, skill_success, n_steps
#             return action, skill_done, skill_success

#         self.steps += 1
        
#         if return_n_steps:
#             return action, skill_done, skill_success, n_steps

#         return action, skill_done, skill_success

#     def _place(self, params, pos_thresh=0.005, axis_angle_thresh=0.1, return_n_steps=False):
#         """
#         Places an object at specified position and orientation, then rehomes

#         Args: 
#             params (6-tuple of floats) : [goal_pos, goal_axis_angle]
#             pos_thresh (float) : how close to target position end effector must be in each axis for success
#             axis_angle_thresh (float) : how close to target axis angle orientation end effector must be in each axis for success
            
#         Returns: 
#             action (7d array) : low level action command for given timestep
#             skill_done (bool) : whether skill execution finished (goal position and orientation reached or max number of steps reached)
#             skill_success (bool) : whether skill execution successfully completed
#         """

#         max_steps = self.skills["place"]["max_steps"]
#         skill_done, skill_success = False, False
#         n_steps = self.steps

#         # extract params
#         place_pos = params[:3]
#         goal_axis_angle = params[3:6]

#         # define params for waypoints (intermediate goal poses)
#         above_wp_pose = np.concatenate([np.array([place_pos[0], place_pos[1], self.waypoint_height]), goal_axis_angle])
#         place_wp_pose = np.concatenate([place_pos, goal_axis_angle])

#         if self.prev_success:
#             self.phase += 1
#             self.prev_success = False
#             time.sleep(0.5)

#         # if max steps reached, rehome and terminate
#         if self.steps > max_steps:
#             action = np.zeros(7)
#             skill_done, skill_success = True, False
#             self.phase = 0
#             self.steps = 0
#             self.prev_success = False
#             self._rehome(gripper_action=-1.0)
#             print(f"max steps reached for place {max_steps}")
#             if return_n_steps:
#                 return action, skill_done, skill_success, n_steps
#             return action, True, False

#         # phase 0 : move to above place position
#         if self.phase == 0:
#             params = np.concatenate([above_wp_pose, [1]])
#             action, _, self.prev_success = self._move_to(params=params, pos_thresh=0.01, axis_angle_thresh=axis_angle_thresh, count_steps=False)

#         # phase 1 : move down to place position
#         if self.phase == 1:
#             params = np.concatenate([place_wp_pose, [1]])
#             action, _, self.prev_success = self._move_to(params=params, pos_thresh=pos_thresh, axis_angle_thresh=axis_angle_thresh, count_steps=False)

#         # phase 2 : release
#         if self.phase == 2:
#             action, _, self.prev_success = self._gripper_release(count_steps=False)

#         # phase 3 : move up to above pick position
#         if self.phase == 3:
#             params = np.concatenate([above_wp_pose, [-1]])
#             action, _, self.prev_success = self._move_to(params=params, pos_thresh=0.01, axis_angle_thresh=axis_angle_thresh, count_steps=False)

#         # phase 4 : rehome after success
#         if self.phase == 4:
#             action = np.zeros(7)
#             skill_done, skill_success = True, True
#             self.phase = 0
#             self.steps = 0
#             self.prev_success = False
#             self._rehome(gripper_action=-1.0)
#             if return_n_steps:
#                 return action, skill_done, skill_success, n_steps
#             return action, skill_done, skill_success

#         self.steps += 1

#         if return_n_steps:
#             return action, skill_done, skill_success, n_steps
        
#         return action, skill_done, skill_success

#     def _push(self, params, pos_thresh=0.01, axis_angle_thresh=0.1, return_n_steps=False):
#         """
#         Executes pushing action given start position, end position, orientation to maintain, and gripper action.
#         Rehomes after execution

#         Args: 
#             params (10-tuple of floats) : [start_pos, end_pos, goal_axis_angle, gripper_action]
#             pos_thresh (float) : how close to target position end effector must be in each axis for success
#             axis_angle_thresh (float) : how close to target axis angle orientation end effector must be in each axis for success
            
#         Returns: 
#             action (7d array) : low level action command for given timestep
#             skill_done (bool) : whether skill execution finished (goal position and orientation reached or max number of steps reached)
#             skill_success (bool) : whether skill execution successfully completed
#         """

#         max_steps = self.skills["push"]["max_steps"]
#         skill_done, skill_success = False, False
#         n_steps = self.steps

#         # extract params
#         push_start_pos = params[:3]
#         push_end_pos = params[3:6]
#         goal_axis_angle = params[6:9]
#         gripper_action = 1 if params[9] > 0 else -1

#         # define params for waypoints (intermediate goal poses)
#         above_start_pose = np.concatenate([np.array([push_start_pos[0], push_start_pos[1], self.waypoint_height]), goal_axis_angle])
#         start_pose = np.concatenate([push_start_pos, goal_axis_angle])
#         end_pose = np.concatenate([push_end_pos, goal_axis_angle])
#         above_end_pose = np.concatenate([np.array([push_end_pos[0], push_end_pos[1], self.waypoint_height]), goal_axis_angle])

#         if self.prev_success:
#             self.phase += 1
#             self.prev_success = False
#             time.sleep(0.5)

#         # print("phase", self.phase)
#         # if max steps reached, rehome and terminate
#         if self.steps > max_steps:
#             action = np.zeros(7)
#             skill_done, skill_success = True, False
#             self.phase = 0
#             self.steps = 0
#             print(f"max steps reached for push {max_steps}")
#             self._rehome(gripper_action=-1.0)
#             if return_n_steps:
#                 return action, skill_done, skill_success, n_steps
#             return action, True, False

#         # phase 0 : move to above start position
#         if self.phase == 0:
#             params = np.concatenate([above_start_pose, [gripper_action]])
#             action, _, self.prev_success = self._move_to(params=params, pos_thresh=pos_thresh, axis_angle_thresh=axis_angle_thresh, count_steps=False)

#         # phase 1 : move down to start position
#         if self.phase == 1:
#             params = np.concatenate([start_pose, [gripper_action]])
#             action, _, self.prev_success = self._move_to(params=params, pos_thresh=pos_thresh, axis_angle_thresh=axis_angle_thresh, count_steps=False)

#         # phase 2 : move to end position
#         if self.phase == 2:
#             params = np.concatenate([end_pose, [gripper_action]])
#             action, _, self.prev_success = self._move_to(params=params, pos_thresh=pos_thresh, axis_angle_thresh=axis_angle_thresh, count_steps=False)

#         # phase 3 : move to end position
#         if self.phase == 3:
#             params = np.concatenate([above_end_pose, [gripper_action]])
#             action, _, self.prev_success = self._move_to(params=params, pos_thresh=pos_thresh, axis_angle_thresh=axis_angle_thresh, count_steps=False)

#         # phase 4 : rehome after success
#         if self.phase == 4:
#             action = np.zeros(7)
#             skill_done, skill_success = True, True
#             self.phase = 0
#             self.steps = 0
#             self.prev_success = False
#             self._rehome(gripper_action=-1.0)
#             if return_n_steps:
#                 return action, skill_done, skill_success, n_steps
#             return action, skill_done, skill_success

#         self.steps += 1

#         if return_n_steps:
#             return action, skill_done, skill_success, n_steps
        
#         return action, skill_done, skill_success

#     def _rehome(self, gripper_action=-1.0):
#         print(f"---------rehoming with gripper {gripper_action}------------")
#         params = np.concatenate([self.reset_eef_pos, [self.reset_yaw], [gripper_action]])
#         skill_done = False
#         rehome_steps = 0

#         while not skill_done:
#             if rehome_steps > 200:
#                 print("WARNING: REHOMING FAILED")
#                 break
#             action, skill_done, skill_success = self._move_to(params=params, pos_thresh=0.01, count_steps=False)
#             self.robot_interface.control(
#                 controller_type=self.controller_type,
#                 controller_cfg=self.controller_config,
#                 action=action,
#             )
#             rehome_steps += 1
#         rehome_steps = 0


