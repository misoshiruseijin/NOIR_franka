import numpy as np

import sys
sys.path.insert(1, "/home/eeg/MAPLE-EF")
from environments.realrobot_env_noir import RealRobotEnv
import time
import pdb
import cv2
from getkey import getkey, keys
from pynput import keyboard

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.filterByColor = False
detector_params.filterByCircularity = False
detector_params.filterByConvexity = False
detector_params.filterByInertia = False
detector_params.minArea = 150.0
detector_params.maxArea = 15000.0

class TablesettingEnv(RealRobotEnv):
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
                "bowl_l_pos", 
                "bowl_s_pos",
                "spoon_pos",
                "cup_pos"
            ]
        self.current_observations = {}
        self.bowl_l_pos = np.zeros(3)
        self.bowl_s_pos = np.zeros(3)
        self.spoon_pos = np.zeros(3)
        self.cup_pos = np.zeros(3)

        # # flags for reward computation
        # self.reward_given = False
        # self.pan_on_stove = False
        # self.sausage_on_pan = False
        # self.sausage_cooked = False
        # self.sausage_in_bread = False
        # self.grasping_pan = False
        # self.grasping_sausage = False

        # # HSV thresholds for objects
        # self.hsv_thresh = {
        #     "sausage" : {
        #         "low" : np.array([0, 120, 20]),
        #         "high" : np.array([5, 255, 255]),
        #     },
        #     "pan" : {
        #         "low" : np.array([100, 130, 50]), # blue tape handle
        #         "high" : np.array([120, 255, 255]),
        #     },
        # }

        self.current_observations = {key : None for key in self.keys}
        self._update_current_observations()

        super().__init__(
            controller_type="OSC_POSE",
            general_cfg_file="config/charmander.yml",
            control_freq=20,
            workspace_limits=workspace_limits,
            skill_config={
                "waypoint_height" : 0.25,
                "idx2skill" : {
                    0 : "pick_from_top",
                    1 : "place_from_top",
                },
            },
            gripper_thresh=0.04,
            reset_joint_pos=reset_joint_pos, 
        )

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
        # # TODO - make sure this works
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

        # # TODO - make sure this works
        return

    def _update_obj_positions(self, wait=True): # TODO
        """
        Use camera input to update object positions

        Args:
            wait_until_obj_found (bool) :
                if True waits until each object is found
                if Flase, tries to find each object once. if not found, use previs locations
        """
        # if wait:
        #     self.sausage_pos = self.get_object_pos(self.hsv_thresh["sausage"]["low"], self.hsv_thresh["sausage"]["high"], "sausage")        
        #     # print(f"Found sausage at {self.sausage_pos}")
        #     if self.sausage_pos[-1] < 0.02: # if estimated height is below some threshold, assume sausage is on the table
        #         self.sausage_pos[-1] = 0.005

        #     self.pan_pos = self.get_object_pos(self.hsv_thresh["pan"]["low"], self.hsv_thresh["pan"]["high"], "pan")
        #     if self.pan_pos[-1] < 0.02:
        #         self.pan_pos[-1] = 0.01
        #     # print(f"Found pan at {self.pan_pos}")
        # else:
        #     sausage_pos = self.get_object_pos(self.hsv_thresh["sausage"]["low"], self.hsv_thresh["sausage"]["high"], "sausage", wait=False)        
        #     pan_pos = self.get_object_pos(self.hsv_thresh["pan"]["low"], self.hsv_thresh["pan"]["high"], "pan", wait=False)
        #     if sausage_pos is not None:
        #         if sausage_pos[-1] < 0.02:
        #             sausage_pos[-1] = 0.005
        #         self.sausage_pos = sausage_pos
        #     if pan_pos is not None:
        #         if pan_pos[-1] < 0.02:
        #             pan_pos[-1] = 0.01
        #         self.pan_pos = pan_pos
        pos = self.get_object_pos()
        return pos

    def update(self, wait=True):
        """
        Updates keypoints, object positions, and task state
        """
        self._update_obj_positions(wait=wait)
        self._update_task_status()
        self._update_current_observations()

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

        self.update()
        num_timesteps = 0
        done, skill_done, skill_success = False, False, False

        if self.normalized_params:
            action = self.skill.unnormalize_params(action)
        
        # self._update_current_observations()
        while not done and not skill_done:
            action_ll, skill_done, skill_success = self.skill.get_action(action)
            obs, _, done, info = super().step(action_ll)
            num_timesteps += 1
        # self._update_current_observations()
        
        info["num_ll_steps"] = num_timesteps
        info["num_hl_steps"] = 1
    
        if done: # horizon exceeded
            print(f"-----------Horizon {self.timestep} Reached--------------")

        # # if action is executed, wait for a few seconds to update the state
        # if go_signal:
        #     print("------- If moving objects, do it now. State update in 5 seconds ----------")
        #     time.sleep(0.005)
        
        self.update()
        obs = self.current_observations

        # process rewards
        reward = self._get_reward()

        # check success
        if self._check_success(): # if success (including accidental success), terminate
            done = True
        
        return obs, reward, done, info   
       
    def _update_current_observations(self): # TODO - get object position from OWL-VIT
        """
        Updates self.current_observations dict with environment-specific observations
        """

        env_obs = {
            "bowl_l_pos" : self.bowl_l_pos, 
            "bowl_s_pos" : self.bowl_s_pos,
            "spoon_pos" : self.spoon_pos,
            "cup_pos" : self.cup_pos,
        }
        self.current_observations.update(env_obs)
        print("updated current observations", self.current_observations)

    def reset(self):
        
        print("Resetting...")
        # reset flags
        self.reward_given = False
        # self.pan_on_stove, self.sausage_on_pan, self.sausage_cooked, self.sausage_in_bread = False, False, False, False
        # self.grasping_pan = False
        # self.grasping_sausage = False

        super().reset()
        self._update_current_observations()       
        
        # add some delay to get time to physically reset the environment
        time.sleep(5)
        # return flattened_obs
        return self.current_observations
