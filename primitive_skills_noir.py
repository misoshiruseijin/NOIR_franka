"""Example script of moving robot joint positions."""
import time

import numpy as np
import math

import sys
sys.path.insert(1, "/home/eeg/deoxys_control/deoxys")

from deoxys import config_root
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig, transform_utils
from deoxys.utils.config_utils import (get_default_controller_config, verify_controller_config)
from deoxys.utils.input_utils import input2action

from deoxys.utils.log_utils import get_deoxys_example_logger
import utils.transformation_utils as U

import pdb
import time

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
        idx2skill=None,
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

        # for now, just don't pass in idx2skill - TODO may need to change if we end up doing EF
        assert idx2skill is None, "Inputting idx2skill is not allowed at the moment"
        self.robot_interface = robot_interface
        self.controller_type = controller_type
        self.controller_config = controller_config

        # robot home position, waypoint height, etc.
        self.reset_eef_pos = [0.45775618, 0.03207872, 0.35534091]
        self.from_top_quat = [0.9998506, 0.00906314, 0.01459545, 0.00192735] # quat when eef is pointing straight down 
        # self.from_side_quat = [-0.70351493, -0.01353285, -0.71054333, 0.00344373] # quat when eef is pointing straight forward
        self.from_side_quat = [0.508257, 0.49478495, -0.49082687, 0.5059166] # quat when eef is pointing to right of robot 
        if reset_joint_pos is None:
            self.reset_joint_positions = [0.09162008114028396, -0.19826458111314524, -0.01990020486871322, -2.4732269941140346, -0.01307073642274261, 2.30396583422025, 0.8480939705504309]
        else:
            self.reset_joint_positions = reset_joint_pos

        self.waypoint_height = waypoint_height # height of waypoint in pick, place, push skills
        if workspace_limits is not None:
            self.workspace_limits = workspace_limits
        else: 
            self.workspace_limits = {
                "x" : (0.3, 0.55),
                "y" : (-0.15, 0.25),
                "z" : (0.03, 0.30)
            }

        # skill settings 
        self.skills = {
            "move_to" : {
                "num_params" : 8,
                "skill" : self._move_to,
                "default_idx" : 0,
            },
            "pick_from_top" : {
                "num_params" : 3,
                "skill" : self._pick_from_top,
                "default_idx" : 1,
            },
            "pick_from_side" : {
                "num_params" : 3,
                "skill" : self._pick_from_side,
                "default_idx" : 2,
            },
            "place_from_top" : {
                "num_params" : 3,
                "skill" : self._place_from_top,
                "default_idx" : 3,
            },
            "place_from_side" : {
                "num_params" : 3,
                "skill" : self._place_from_side,
                "default_idx" : 4,
            },
            "push_xy" : {
                "num_params" : 6,
                "skill" : self._push_xy,
                "default_idx" : 5,
            },
            "push_z" : {
                "num_params" : 5,
                "skill" : self._push_z,
                "default_idx" : 6,
            },
            "wipe_xy" : {
                "num_params" : 6,
                "skill" : self._wipe_xy,
                "default_idx" : 7,
            },
            "gripper_action" : {
                "num_params" : 1,
                "skill" : self._gripper_action,
                "default_idx" : 8,
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

    def execute_skill(self, action):
        """
        Executes skill
        Args:
            action : skill selection vector concatenated with params vector
        """
        # get the skill to execute
        skill_idx = np.argmax(action[:self.num_skills])
        skill = self.skills[self.idx2skill[skill_idx]]["skill"] 
        
        # extract params and execute
        params = action[self.num_skills:]
        skill(params)

    def _move_to(self, params, step_size=0.015, deg_step_size=5, finetune=True):
        """
        Moves end effector to goal position and orientation

        Args: 
            params (8-tuple of floats) : [goal_pos, goal_quat, gripper_state]
        """

        while self.robot_interface.state_buffer_size == 0:
            logger.warn("Robot state not received")
            time.sleep(0.5)

        # extract  params
        action = params[:-1]
        action = np.clip(action, -1, 1)
        goal_pos = action[:3]
        goal_orn = action[3:7]
        gripper_action = params[7]

        fine_tune_dist = 0.05 # start fine tuning after distance to goal is within this value

        tran_inter, ori_inter = self.interpolate_poses(goal_pos, goal_orn, step_size=step_size, step_size_deg=deg_step_size) # num_step should between [10, 30], the larger the slower

        for i in range(len(tran_inter)):
            # TODO - check if human wants to interrupt. if yes, stop execution and rehome. Make sure the other skills stop executing as well
            trans = tran_inter[i]
            ori = U.mat2quat(ori_inter[i])

            for _ in range(3): # how many times does one waypoint execute. 
                               # The smaller the faster but less accurate. 
                               # The larger the slower and more accurate but you will feel pausing between waypoints
                new_action = self.poses_to_action(trans, ori)
                new_action = np.concatenate((new_action, [gripper_action]))
                self.robot_interface.control(
                    controller_type=self.controller_type,
                    action=new_action,
                    controller_cfg=self.controller_config,
                )
            
            cur_quat, cur_pos = self.robot_interface.last_eef_quat_and_pos
            cur_pos = cur_pos.flatten()
            pos_error = goal_pos - cur_pos
            if np.linalg.norm(pos_error) < fine_tune_dist:
                break

        # fine tune
        if finetune:
            tran_inter, ori_inter = self.interpolate_poses(action[:3], action[3:7], step_size=0.005, step_size_deg=deg_step_size) # num_step should between [10, 30], the larger the slower
            for i in range(len(tran_inter)):
                trans = tran_inter[i]
                ori = U.mat2quat(ori_inter[i])

                for _ in range(3): # how many times does one waypoint execute. 
                                # The smaller the faster but less accurate. 
                                # The larger the slower and more accurate but you will feel pausing between waypoints
                    new_action = self.poses_to_action(trans, ori)
                    new_action = np.concatenate((new_action, [gripper_action]))
                    self.robot_interface.control(
                        controller_type=self.controller_type,
                        action=new_action,
                        controller_cfg=self.controller_config,
                    )

    def _pick_from_top(self, params):
        """
        Picks up object at specified position from top and rehomes

        Args:
            params (3-tuple of floats) : [goal_pos]
        """
        # define waypoints
        goal_pos = params[:3]
        waypoint = np.array([params[0], params[1], self.waypoint_height])

        # move to above pick position 
        params = np.concatenate([waypoint, self.from_top_quat, [-1]])
        self._move_to(params=params)

        # move down to pick position
        params = np.concatenate([goal_pos, self.from_top_quat, [-1]])
        self._move_to(params=params)
        final_quat, final_pos = self.robot_interface.last_eef_quat_and_pos
        print("pick pos error", goal_pos - final_pos.flatten())
        
        # close gripper
        self._gripper_action(params=[1])

        # move back up to waypoint
        params = np.concatenate([waypoint, self.from_top_quat, [1]])
        self._move_to(params=params, finetune=False)

        # rehome
        self._rehome(gripper_action=1, gripper_quat=self.from_top_quat)

    def _pick_from_side(self, params):
        """
        Picks up object at specified position from side and rehomes

        Args:
            params (3-tuple of floats) : [goal_pos]
        """
        # define waypoints
        goal_pos = params[:3]
        waypoint_side = np.array([params[0], params[1] + 0.2, params[2]])
        waypoint_above = np.array([params[0], params[1], self.waypoint_height])

        # move to beside pick position
        params = np.concatenate([waypoint_side, self.from_side_quat, [-1]])
        self._move_to(params=params, finetune=False)

        # move to pick position
        params = np.concatenate([goal_pos, self.from_side_quat, [-1]])
        self._move_to(params=params)
        
        # close gripper
        self._gripper_action(params=[1])

        # move up to waypoint
        params = np.concatenate([waypoint_above, self.from_side_quat, [1]])
        self._move_to(params=params, finetune=False)

        # rehome
        self._rehome(gripper_action=1, gripper_quat=self.from_side_quat)

    def _place_from_top(self, params):
        """
        Places object at specified location with gripper pointing down

        Args:
            params (3-tuple of floats) : [goal_pos]
        """
        # define waypoints
        goal_pos = params[:3]
        waypoint = np.array([params[0], params[1], self.waypoint_height])

        # move to above goal position 
        params = np.concatenate([waypoint, self.from_top_quat, [1]])
        self._move_to(params=params, finetune=False)

        # move down to goal position
        params = np.concatenate([goal_pos, self.from_top_quat, [1]])
        self._move_to(params=params)
        
        # release gripper
        self._gripper_action(params=[-1])

        # move back up to waypoint
        params = np.concatenate([waypoint, self.from_top_quat, [-1]])
        self._move_to(params=params, finetune=False)

        # rehome
        self._rehome(gripper_action=-1, gripper_quat=self.from_top_quat)

    def _place_from_side(self, params):
        """
        Places object at specified location with gripper pointing down

        Args:
            params (3-tuple of floats) : [goal_pos]
        """
        # define waypoints
        goal_pos = params[:3]
        waypoint = np.array([params[0], params[1], self.waypoint_height])

        # move to above goal position 
        params = np.concatenate([waypoint, self.from_side_quat, [1]])
        self._move_to(params=params, finetune=False)

        # move down to goal position
        params = np.concatenate([goal_pos, self.from_side_quat, [1]])
        self._move_to(params=params)
        
        # release gripper
        self._gripper_action(params=[-1])

        # move back up to waypoint
        params = np.concatenate([waypoint, self.from_side_quat, [-1]])
        self._move_to(params=params, finetune=False)

        # rehome
        self._rehome(gripper_action=-1, gripper_quat=self.from_top_quat)

    def _push_z(self, params):
        """
        Start from specified position with gripper pointing down, pushes in z direction by some delta, then rehomes

        Args: 
            params (5-tuple of floats) : [start_pos, dz, yaw_angle[deg]]
        """

        start_pos = params[:3]
        dz = params[3]
        yaw = params[4]
        gripper_action = 1 # gripper is closed

        goal_pos = [start_pos[0], start_pos[1], start_pos[2] + dz]
        waypoint_above = [start_pos[0], start_pos[1], self.waypoint_height]

        # convert euler to quat
        from_top_euler = U.mat2euler(U.quat2mat(self.from_top_quat)) # convert to euler
        goal_euler = np.array([from_top_euler[0], from_top_euler[1], np.radians(yaw)]) # update yaw component
        goal_quat = U.mat2quat(U.euler2mat(goal_euler)) # convert new orn back to quat
        print("goal quat", goal_quat)
        print("from top quat", self.from_top_quat)

        # move to start pos
        params = np.concatenate([start_pos, goal_quat, [gripper_action]])
        self._move_to(params=params)

        # move in z by dz
        params = np.concatenate([goal_pos, goal_quat, [gripper_action]])
        self._move_to(params=params, finetune=False)

        # pause 
        self._pause(gripper_action=1.0, sec=1)

        # move up to waypoint 
        params = np.concatenate([waypoint_above, goal_quat, [gripper_action]])
        self._move_to(params=params, finetune=False)
        
        # rehome
        self._rehome(gripper_action=gripper_action, gripper_quat=self.from_top_quat)

    def _push_xy(self, params):
        """
        Start from specified position with gripper pointing down, pushes in x and y direction by specified delta, then rehomes

        Args: 
            params (6-tuple of floats) : [start_pos, dx, dy, yaw_angle[deg]]
        """
        start_pos = params[:3]
        dx = params[3]
        dy = params[4]
        yaw = params[5]
        gripper_action = 1 # gripper is closed

        goal_pos = [start_pos[0] + dx, start_pos[1] + dy, start_pos[2]]
        waypoint_above = [start_pos[0], start_pos[1], self.waypoint_height]

        # convert euler to quat
        from_top_euler = U.mat2euler(U.quat2mat(self.from_top_quat)) # convert to euler
        goal_euler = np.array([from_top_euler[0], from_top_euler[1], np.radians(yaw)]) # update yaw component
        goal_quat = U.mat2quat(U.euler2mat(goal_euler)) # convert new orn back to quat
        print("goal quat", goal_quat)
        print("from top quat", self.from_top_quat)

        # move to start pos
        params = np.concatenate([start_pos, goal_quat, [gripper_action]])
        self._move_to(params=params)

        # move in xy by delta 
        params = np.concatenate([goal_pos, goal_quat, [gripper_action]])
        self._move_to(params=params)

        # move up to waypoint 
        params = np.concatenate([waypoint_above, goal_quat, [gripper_action]])
        self._move_to(params=params, finetune=False)
        
        # rehome
        self._rehome(gripper_action=gripper_action, gripper_quat=self.from_top_quat)

    def _wipe_xy(self, params):
        """
        Wipes a surface by starting at specified position, moving on the xy plane with specified dx, dy, returns to start position, then rehomes
        
        Args:
            params (7-tuple of floats) : [start_pos, dx, dy, yaw_angle[deg]]
        """
        start_pos = params[:3]
        dx = params[3]
        dy = params[4]
        yaw = params[5]
        gripper_action = 1 # gripper is closed

        end_pos = [start_pos[0] + dx, start_pos[1] + dy, start_pos[2]]
        waypoint_above = [start_pos[0], start_pos[1], self.waypoint_height]

        # convert euler to quat
        from_top_euler = U.mat2euler(U.quat2mat(self.from_top_quat)) # convert to euler
        goal_euler = np.array([from_top_euler[0], from_top_euler[1], np.radians(yaw)]) # update yaw component
        goal_quat = U.mat2quat(U.euler2mat(goal_euler)) # convert new orn back to quat
        print("goal quat", goal_quat)
        print("from top quat", self.from_top_quat)

        # move to start pos
        params = np.concatenate([start_pos, goal_quat, [gripper_action]])
        self._move_to(params=params)

        # move in xy by delta 
        params = np.concatenate([end_pos, goal_quat, [gripper_action]])
        self._move_to(params=params)

        # move back to start position
        params = np.concatenate([start_pos, goal_quat, [gripper_action]])
        self._move_to(params=params)

        # move up to waypoint
        params = np.concatenate([waypoint_above, goal_quat, [gripper_action]])
        self._move_to(params=params, finetune=False)

    def _gripper_action(self, params):
        """
        Closes or opens gripper
        
        Args:
            params : -1 to open, 1 to close
        """
        action = np.zeros(7)
        action[-1] = params[0]

        for _ in range(30):
            self.robot_interface.control(
                controller_type=self.controller_type,
                action=action,
                controller_cfg=self.controller_config,
            )
        
    def _rehome(self, gripper_action, gripper_quat, finetune=True, deg_step_size=None):
        """
        Returns to home position with gripper pointing down. Finetuning step is turned off as default

        Args:
            gripper_action (int) : -1 for open, 1 for closed
            gripper_direction (str) : "down" for pointing down, "front" for pointing forward
        """
        
        params = np.concatenate([self.reset_eef_pos, gripper_quat, [gripper_action]])
        
        if deg_step_size is not None:
            self._move_to(params=params, step_size=0.015, finetune=finetune, deg_step_size=deg_step_size)
        else:
            self._move_to(params=params, step_size=0.015, finetune=finetune)
            
        final_quat, final_pos = self.robot_interface.last_eef_quat_and_pos
        print("rehome pos error", self.reset_eef_pos - final_pos.flatten())

    def _reset(self):
        """
        Resets joints to home position
        """
        reset_joints_to(self.robot_interface, self.reset_joint_positions)

    def _pause(self, gripper_action, sec):
        """
        pause in place for specified number of seconds
        """
        action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_action]

        start_time = time.time()

        while time.time() < start_time + sec:
            self.robot_interface.control(
                controller_type=self.controller_type,
                action=action,
                controller_cfg=self.controller_config,
            )

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

    def interpolate_poses(self, target_pos, target_rot=None, num_steps=None, step_size=None, step_size_deg=5):
        assert num_steps is None or step_size is None
        last_robot_state = self.robot_interface._state_buffer[-1]
        last_gripper_state = self.robot_interface._gripper_state_buffer[-1]
        ee_pose = np.array(last_robot_state.O_T_EE).reshape((4, 4)).T
        if np.count_nonzero(ee_pose.reshape(-1)) == 0:
            raise Exception("GOT ZERO EE POSE")
        start_pos = ee_pose[:3, 3]
        start_rot = ee_pose[:3, :3]
        target_rot = U.quat2mat(target_rot)
    

        rot_step_size = np.radians(step_size_deg) # degrees per step

        if num_steps is None:
            # calculate number of steps in terms of translation
            delta_pos = target_pos - start_pos
            if np.linalg.norm(delta_pos) > 0:
                num_steps_pos = math.ceil(np.linalg.norm(delta_pos) / step_size)
            else:
                num_steps_pos = 1
            # calculate number of steps in terms of rotation
            rot_angle = np.arccos((np.trace(np.dot(start_rot, np.transpose(target_rot))) - 1) / 2)
            if rot_angle >= np.radians(rot_step_size):
                num_steps_rot = int(np.ceil(rot_angle / rot_step_size))  # 2 degree for one step
            else:
                num_steps_rot = 1
            
            num_steps = max(num_steps_rot, num_steps_pos)
            # print("rot angle", rot_angle)
            # print(f'num_steps_pos: {num_steps_pos}')
            # print(f'num_steps_rot: {num_steps_rot}')
            # print("num steps", num_steps)

        tran_inter = self.interpolate_tranlations(start_pos, target_pos, num_steps)
        ori_inter = self.interpolate_rotations(start_rot, target_rot, num_steps)

        return tran_inter, ori_inter

    def interpolate_tranlations(self, T1, T2, num_steps, perturb=False):
        delta_pos = T2 - T1
        pos_step_size = delta_pos / num_steps
        grid = np.arange(num_steps).astype(np.float64)
        if perturb:
            # move the interpolation grid points by up to a half-size forward or backward
            perturbations = np.random.uniform(
                low=-0.5,
                high=0.5,
                size=(num_steps - 2,),
            )
            grid[1:-1] += perturbations
        pos_steps = np.array([T1 + grid[i] * pos_step_size for i in range(num_steps)])

        # add in endpoint
        pos_steps = np.concatenate([pos_steps, T2[None]], axis=0)

        return pos_steps[1:]

    def interpolate_rotations(self, R1, R2, num_steps):
        """
        Interpolate between 2 rotation matrices.
        """
        q1 = U.mat2quat(R1)
        q2 = U.mat2quat(R2)
        rot_steps = np.array([U.quat2mat(self.quat_slerp(q1, q2, tau=(float(i) / num_steps))) for i in range(num_steps)])

        # add in endpoint
        rot_steps = np.concatenate([rot_steps, R2[None]], axis=0)

        return rot_steps[1:]
    
    def quat_slerp(self, q1, q2, tau):
        """
        Adapted from robosuite.
        """
        if tau == 0.0:
            return q1
        elif tau == 1.0:
            return q2
        d = np.dot(q1, q2)
        if abs(abs(d) - 1.0) < np.finfo(float).eps * 4.:
            return q1
        if d < 0.0:
            # invert rotation
            d = -d
            q2 *= -1.0
        angle = math.acos(np.clip(d, -1, 1))
        if abs(angle) < np.finfo(float).eps * 4.:
            return q1
        isin = 1.0 / math.sin(angle)
        q1 = q1 * math.sin((1.0 - tau) * angle) * isin
        q2 = q2 * math.sin(tau * angle) * isin
        q1 = q1 + q2
        return q1
    
    def poses_to_action(self, target_pos, target_rot=None, max_dpos=0.05, max_drot=0.2):
        """
        Takes a starting eef pose and target controller pose and returns a normalized action that
        corresponds to the desired controller target.
        """
        last_robot_state = self.robot_interface._state_buffer[-1]
        last_gripper_state = self.robot_interface._gripper_state_buffer[-1]
        ee_pose = np.array(last_robot_state.O_T_EE).reshape((4, 4)).T
        if np.count_nonzero(ee_pose.reshape(-1)) == 0:
            raise Exception("GOT ZERO EE POSE")
        start_pos = ee_pose[:3, 3]
        start_rot = ee_pose[:3, :3]

        delta_position = target_pos - start_pos
        delta_position = np.clip(delta_position / max_dpos, -1., 1.)
        if target_rot is None:
            return delta_position

        target_rot = U.quat2mat(target_rot)

        # use the OSC controller's convention for delta rotation
        delta_rot_mat = target_rot.dot(start_rot.T)
        delta_quat = U.mat2quat(delta_rot_mat)
        delta_rotation = U.quat2euler(delta_quat)
        delta_rotation = np.clip(delta_rotation / max_drot, -1., 1.)
        return np.concatenate([delta_position, delta_rotation])
