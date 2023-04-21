import numpy as np

import sys
from environments.solo_envs.realrobot_env_solo import RealRobotEnvSolo
import time
import pdb
from getkey import getkey, keys
from pynput import keyboard

class TablesettingEnv(RealRobotEnvSolo):
    """Custom Environment that follows gym interface."""

    def __init__(
        self,
        reward_scale=1.0,
        normalized_params=True,
        keys=None,
        reset_joint_pos=None,
        workspace_limits={"x" : (0.35, 0.55), "y" : (-0.15, 0.25), "z" : (0.0, 0.08)},
    ): 

        self.keys = keys
        if self.keys is None:
            self.keys = [
                "eef_pos", "gripper_state",
                "light blue bowl", 
                "red bowl",
                "red and blue spoon",
                "shiny silver cup",
            ]

        self.current_observations = {}

        self.current_observations = {key : None for key in self.keys}
        # self._update_current_observations()

        super().__init__(
            controller_type="OSC_POSE",
            general_cfg_file="config/charmander.yml",
            control_freq=20,
            workspace_limits=workspace_limits,
            skill_config={
                "waypoint_height" : 0.25,
                # "idx2skill" : {
                #     0 : "pick_from_top",
                #     1 : "place_from_top",
                # },
                # TODO - include objID2skillID dict here
            },
            detector_config={
                "texts" : ["shiny silver cup", "light blue bowl", "red bowl", "red and blue spoon"], # text description of objects of interest
                "thresholds" : [0.02, 0.02, 0.02, 0.02],
            },
            gripper_thresh=0.04,
            reset_joint_pos=reset_joint_pos, 
        )

        self.reward_given = False # TODO - should be able to eliminate this
        self.reward_scale = reward_scale
        self.num_skills = self.skill.num_skills
        self.normalized_params = normalized_params
        
        self.gripper_state = -1

        # update object positions using camera input
        self.update()

    def reward(self): # TODO
        """
        Reard function for task. Returns environment reward only.
        Environment reward is given when task is successfully complete:
            pan placed on stove -> sausage placed on pan -> sausage placed in bread 
                in that order

        Returns:
            reward (float) : environment reward
        """
        
        reward = 0.0

        # check for task completion
        if self._check_success() and not self.reward_given:
            reward = 10.0
            if not self.use_skills:
                self.reward_given = True
                print("~~~~~~~~~~~~ TASK COMPLETE ~~~~~~~~~~~~~~~")

        reward = self.reward_scale * reward / 10.0
        return reward

    def _get_reward(self): # TODO

        reward = 0.0

        # check for task completion
        if self._check_success() and not self.reward_given:
            reward = 10.0
            self.reward_given = True
            print("~~~~~~~~~~~~ TASK COMPLETE ~~~~~~~~~~~~~~~")

        reward = self.reward_scale * reward / 10.0
        return reward

    def _check_success(self): # TODO

        return False

    def _update_task_status(self): # TODO
        """
        Updates task status using camera input
        """
        # print("\nUpdate task status")
        # robot_state = self._get_current_robot_state()
        # eef_pos = robot_state["eef_pos"]
        
        # self.grasping_sausage = self.sausage_pos[2] > 0.1
        # self.grasping_pan = self.pan_pos[2] > 0.12
        # # these 2 only relevant when place_in_bread = True TODO - test this
        # sausage_bread_error = np.abs(self.bread_position[:-1] - self.sausage_pos[:-1])
        # self.sausage_in_bread = sausage_bread_error[0] < 0.09 and sausage_bread_error[1] < 0.02
        
        # self.pan_on_stove = self.pan_pos[-1] > 0.04 and not self.grasping_pan
        # # self.pan_on_stove = np.linalg.norm(self.pan_pos[:-1] - self.stove_position[:-1]) < 0.06
        # # self.sausage_on_pan = np.linalg.norm(self.sausage_pos[:-1] - self.stove_position[:-1]) < 0.08
        # self.sausage_on_pan = (
        #     abs(self.sausage_pos[0] - self.stove_position[0]) < 0.13
        #     and abs(self.sausage_pos[1] - self.stove_position[1]) < 0.08
        #     and self.pan_on_stove
        # )
        # # sausage_on_pan = np.linalg.norm(self.sausage_pos[:-1] - self.stove_position[:-1]) < 0.05
        
        # if not self.sausage_cooked and self.sausage_on_pan and self.pan_on_stove:
        #     self.sausage_cooked = True 

        return        

    def update(self, wait=True):
        """
        Updates keypoints, object positions, and task state
        """
        self._update_task_status()
        self._update_current_observations(wait=wait)

    def human_reward(self, action): # TODO - just here for reference
        print("=====human reward call=======")
        global human_feedback_reward
        global go_signal
        human_feedback_reward = 0.0
        go_signal = False
        if self.visualize_params:
            self.human_feedback_request(action)

        human_feedback_value = ''

        while human_feedback_value not in ['g', 'b', 'e', 's']:
            print("Input your feedback:")
            human_feedback_value = getkey()
            # print("key we get {}".format(human_feedback_value))

            if human_feedback_value == "g":
                human_feedback_reward = 1.0
                go_signal = False
            elif human_feedback_value == "b":
                human_feedback_reward = -1.0
                go_signal = False
            elif human_feedback_value == "e":
                human_feedback_reward = 0.0
                go_signal = True
            elif human_feedback_value == "s":
                human_feedback_reward = 19.0
                go_signal = False
            else:
                human_feedback_reward = 0.0
                go_signal = False

        return human_feedback_reward, go_signal        

    def step(self, action):

        global go_signal
        print("=======call to step=========")
        print("action", action)
        # print("go signal", go_signal)
        info = {}

        # self.update()
       
        if self.normalized_params:
            action = self.skill.unnormalize_params(action)
        
        super().step(action)

        self.update()
        obs = self.current_observations

        # process rewards
        reward = self._get_reward()

        done = False # TODO - if there are termination conditions, add it here

        # check success
        if self._check_success(): # if success (including accidental success), terminate
            done = True
        
        return obs, reward, done, info   
       
    def _update_current_observations(self, wait=True):
        """
        Updates self.current_observations dict with environment-specific observations
        """
        obj_positions = self.get_object_pos(wait=wait)
        # TODO - if needed, adjust obj position observations (e.g. set every z value < eps to zero)
        
        self.current_observations.update(obj_positions)
        print("updated current observations", self.current_observations)

    def reset(self):
        
        print("Resetting...")
        # reset flags
        self.reward_given = False

        super().reset()
        self._update_current_observations()       
        
        # add some delay to get time to physically reset the environment
        time.sleep(5)
        # return flattened_obs
        return self.current_observations