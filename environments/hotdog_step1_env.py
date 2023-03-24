import numpy as np

import sys
sys.path.insert(1, "/home/eeg/MAPLE-EF")
from environments.realrobot_env import RealRobotEnv
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

class HotDogStep1Env(RealRobotEnv):
    """Custom Environment that follows gym interface."""

    def __init__(
        self,
        reward_scale=1.0,
        horizon=5000,
        use_skills=True,
        normalized_params=True,
        use_aff_rewards=False,
        stove_position=(0.5, 0.2, 0.05),
        bread_position=(0.5, -0.1, 0.05), 
        use_yaw=True,
        keys=None,
        detector_params=detector_params, # whether skill parameter visualization is used
        visualize_params=True,
        reset_joint_pos=None,
        workspace_limits={"x" : (0.35, 0.55), "y" : (-0.15, 0.25), "z" : (0.0, 0.08)},
        use_human_feedback=True,
    ): 

        self.keys = keys
        if self.keys is None:
            self.keys = [
                # "eef_pos", "gripper_state",
                "sausage_pos", "grasping_sausage", "sausage_cooked", "sausage_in_bread", "sausage_on_pan",
                "pan_pos", "grasping_pan", "pan_on_stove",
            ]
        self.current_observations = {}
        self.sausage_pos = np.zeros(3)
        self.pan_pos = np.zeros(3)

        self.visualize_params = visualize_params
        self.use_human_feedback = use_human_feedback

        # flags for reward computation
        self.reward_given = False
        self.pan_on_stove = False
        self.sausage_on_pan = False
        self.sausage_cooked = False
        self.sausage_in_bread = False
        self.grasping_pan = False
        self.grasping_sausage = False

        # HSV thresholds for objects
        self.hsv_thresh = {
            "sausage" : {
                "low" : np.array([0, 120, 20]),
                "high" : np.array([5, 255, 255]),
            },
            "pan" : {
                "low" : np.array([100, 130, 50]), # blue tape handle
                "high" : np.array([120, 255, 255]),
            },
        }

        self.current_observations = {key : None for key in self.keys}
        self._update_current_observations()

        super().__init__(
            horizon=horizon,
            use_skills=use_skills,
            controller_type="OSC_YAW",
            general_cfg_file="config/charmander.yml",
            control_freq=20,
            workspace_limits=workspace_limits,
            skill_config={
                "waypoint_height" : 0.25,
                "yaw_limits" : (-0.25*np.pi, 0.25*np.pi),
                "idx2skill" : {
                    0 : "pick",
                    1 : "place",
                    # 2 : "push"
                },
                "aff_pos_thresh" : {
                    "pick" : 0.03,
                    "place" : 0.03,
                    # "push" : 0.05
                },
            },
            gripper_thresh=0.04,
            use_yaw=use_yaw,
            detector_params=detector_params,
            reset_joint_pos=reset_joint_pos,
        )

        self.reward_scale = reward_scale
        self.use_skills = use_skills
        self.num_skills = self.skill.num_skills
        self.normalized_params = normalized_params
        self.use_aff_rewards = use_aff_rewards
        
        self.gripper_state = -1
        self.num_neutral_feedbacks = 0

        # fix object positions
        self.stove_position = np.array(stove_position)
        self.bread_position = np.array(bread_position)
        self.pan_place_pos = np.array([0.455, 0.20, 0.06])
        self.sausage_place_pos1 = self.stove_position + np.array([0, 0, 0.04])
        self.sausage_place_pos2 = self.bread_position

        # update object positions using camera input
        self.update()


    def reward(self,):
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

    def _get_reward(self):

        reward = 0.0

        # check for task completion
        if self._check_success() and not self.reward_given:
            reward = 10.0
            self.reward_given = True
            print("~~~~~~~~~~~~ TASK COMPLETE ~~~~~~~~~~~~~~~")

        reward = self.reward_scale * reward / 10.0
        return reward

    def _aff_penalty(self, action):
        """
        Computes affordance penalty
        """
        aff_penalty_factor = 1.0
        aff_reward = self.skill.compute_affordance_reward(action, self.keypoints)
        assert 0.0 <= aff_reward <= 1.0
        aff_penalty = 1.0 - aff_reward
        aff_penalty *= aff_penalty_factor

        return aff_penalty

    def _check_success(self):

        # success when the pan is picked up
        success = False
        if self.grasping_pan:
            success = True
        if success and not self.reward_given:
            print("========= Task Success!! ===========")
        return success

    def _update_keypoints(self):
        """
        Update self.keypoints dic according to current task state 
        """
        # if holding the pan, the best action is to place it on the stove
        if self.grasping_pan:
            self.keypoints["pick"] = []
            self.keypoints["place"] = [self.pan_place_pos]
        
        # if holding the sausage
        elif self.grasping_sausage:
            if self.pan_on_stove:
                if self.sausage_cooked: # sausage cooked, pan on stove
                    self.keypoints["pick"] = []
                    self.keypoints["place"] = [self.sausage_place_pos2]
                else: # sausage not cooked, pan on stove
                    self.keypoints["pick"] = []
                    self.keypoints["place"] = [self.sausage_place_pos1]
            else: # holding sausage but pan is not on stove -> should put down the sausage somewhere
                self.keypoints["pick"] = []
                self.keypoints["place"] = None

        # if not holding anything
        else:
            if not self.pan_on_stove: # pan is not on stove -> pick up the pan
                self.keypoints["pick"] = [self.pan_pos]
                self.keypoints["place"] = []
    

    def _update_task_status(self): 
        """
        Updates task status using camera input
        """
        # TODO - make sure this works
        print("\nUpdate task status")
        robot_state = self._get_current_robot_state()
        eef_pos = robot_state["eef_pos"]
        
        self.grasping_sausage = self.sausage_pos[2] > 0.1
        self.grasping_pan = self.pan_pos[2] > 0.12
        # these 2 only relevant when place_in_bread = True TODO - test this
        sausage_bread_error = np.abs(self.bread_position[:-1] - self.sausage_pos[:-1])
        self.sausage_in_bread = sausage_bread_error[0] < 0.09 and sausage_bread_error[1] < 0.02
        
        self.pan_on_stove = self.pan_pos[-1] > 0.04 and not self.grasping_pan
        # self.pan_on_stove = np.linalg.norm(self.pan_pos[:-1] - self.stove_position[:-1]) < 0.06
        # self.sausage_on_pan = np.linalg.norm(self.sausage_pos[:-1] - self.stove_position[:-1]) < 0.08
        self.sausage_on_pan = (
            abs(self.sausage_pos[0] - self.stove_position[0]) < 0.13
            and abs(self.sausage_pos[1] - self.stove_position[1]) < 0.08
            and self.pan_on_stove
        )
        # sausage_on_pan = np.linalg.norm(self.sausage_pos[:-1] - self.stove_position[:-1]) < 0.05
        
        if not self.sausage_cooked and self.sausage_on_pan and self.pan_on_stove:
            self.sausage_cooked = True 

        # TODO - make sure this works

    def _update_obj_positions(self, wait=True):
        """
        Use camera input to update object positions

        Args:
            wait_until_obj_found (bool) :
                if True waits until each object is found
                if Flase, tries to find each object once. if not found, use previs locations
        """
        if wait:
            self.sausage_pos = self.get_object_pos(self.hsv_thresh["sausage"]["low"], self.hsv_thresh["sausage"]["high"], "sausage")        
            # print(f"Found sausage at {self.sausage_pos}")
            if self.sausage_pos[-1] < 0.02: # if estimated height is below some threshold, assume sausage is on the table
                self.sausage_pos[-1] = 0.005

            self.pan_pos = self.get_object_pos(self.hsv_thresh["pan"]["low"], self.hsv_thresh["pan"]["high"], "pan")
            if self.pan_pos[-1] < 0.02:
                self.pan_pos[-1] = 0.01
            # print(f"Found pan at {self.pan_pos}")
        else:
            sausage_pos = self.get_object_pos(self.hsv_thresh["sausage"]["low"], self.hsv_thresh["sausage"]["high"], "sausage", wait=False)        
            pan_pos = self.get_object_pos(self.hsv_thresh["pan"]["low"], self.hsv_thresh["pan"]["high"], "pan", wait=False)
            if sausage_pos is not None:
                if sausage_pos[-1] < 0.02:
                    sausage_pos[-1] = 0.005
                self.sausage_pos = sausage_pos
            if pan_pos is not None:
                if pan_pos[-1] < 0.02:
                    pan_pos[-1] = 0.01
                self.pan_pos = pan_pos

    def update(self, wait=True):
        """
        Updates keypoints, object positions, and task state
        """
        self._update_obj_positions(wait=wait)
        self._update_task_status()
        self._update_keypoints()
        self._update_current_observations()

    def compute_affordance_score(self, action):

        thresholds = np.array([
            np.array([0.03, 0.03, 0.01]), # pan handle position
            # np.array([0.03, 0.01, 0.03]), # pan place position
            np.array([0.03, 0.03, 0.005]), # sausage position
            np.array([0.13, 0.08, 0.02]), # sausage place position
            np.array([0.01, 0.03, 0.02]), # bread position
        ])
        good_params = np.array([ # TODO define these
            self.pan_pos, # pan handle position
            # self.pan_place_pos, # pan place position
            self.sausage_pos, # sausage position 
            self.sausage_place_pos1, # sausage place position
            self.sausage_place_pos2, # bread position
        ])
        
        # unnormalize params
        if self.normalized_params:
            action = self.skill.unnormalize_params(action)
        params = action[self.num_skills:]

        good = np.any(np.all(np.abs(good_params - params) < thresholds, axis=1))
        print("!!!!!----Params good???----!!!!", good)
        # print("\n", np.abs(good_params - params), "\n", np.abs(params - good_params) < thresholds)
        # return 1 if good else 0 # NOTE change this
        return 0.0 if good else -0.1

    def human_reward(self, action):
        print("=====human reward call=======")
        global human_feedback_reward
        global go_signal
        human_feedback_reward = 0.0
        go_signal = False
        if self.visualize_params:
            self.human_feedback_request(action)

        # get affordance reward
        if self.use_aff_rewards and self.use_skills:
            affordance_reward = self.compute_affordance_score(action)
        else:
            affordance_reward = 0.0 

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

        # add affordance reward
        if not human_feedback_value == "s":
            human_feedback_reward += affordance_reward 
            print(human_feedback_reward)
        return human_feedback_reward, go_signal        
    
    def human_reward_after_execution(self):
        print("=====human reward after execution call=======")
        global human_feedback_reward
        global go_signal
        human_feedback_reward = 0.0
        go_signal = False

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
                self.num_neutral_feedbacks += 1
            elif human_feedback_value == "s":
                human_feedback_reward = 19.0
                go_signal = False
            else:
                human_feedback_reward = 0.0
                go_signal = False

        print("----# of neutral feedbacks:", self.num_neutral_feedbacks)
        return human_feedback_reward, go_signal

    def step(self, action):

        global go_signal
        print("=======call to step=========")
        print("action", action)
        # print("go signal", go_signal)
        info = {}

        if not self.use_human_feedback:
            go_signal = True
            
        # if using skills
        if self.use_skills:
            self.update()
            num_timesteps = 0
            done, skill_done, skill_success = False, False, False

            aff_penalty = self._aff_penalty(action)
            print("penalty", aff_penalty)
            
            # if action should be executed
            if go_signal:
                print("------Go Signal Received------")
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
            
            else: # just update the observations

                info["num_ll_steps"] = 0
                info["num_hl_steps"] = 0
            
            if done: # horizon exceeded
                print(f"-----------Horizon {self.timestep} Reached--------------")

            # if action is executed, wait for a few seconds to update the state
            if go_signal:
                print("------- If moving objects, do it now. State update in 5 seconds ----------")
                time.sleep(0.005)
            
            self.update()
            obs = self._flatten_obs(self.current_observations)
            
            # process rewards
            reward = self._get_reward()
            print("reward without affordance", reward)

            if self.use_aff_rewards:
                reward -= self._aff_penalty(action)
                print("reward after affordance", reward)
            # check success
            if self._check_success(): # if success (including accidental success), terminate
                done = True
            
            print("current timestep: ", self.timestep)
            return obs, reward, done, info   

        # # if using low level action commands
        else:
            self.update(wait=False)
            if action.shape[0] == 7:
                # ignore roll and pitch
                action[3:5] = 0
            elif action.shape[0] == 5:
                action = np.concatenate([action[:3], np.zeros(2), action[3:]])
            elif action.shape[0] == 4:
                action = np.concatenate([action[:3], np.zeros(3), action[3:]])

            # scale action for higher precision
            action[:-1] *= 1.0
 
            # make sure action does not send robot out of workspace bounds
            action_in_bounds = self._check_action_in_ws_bounds(action)
            if not action_in_bounds:
                action[:-1] = 0.0
                print("Action out of bounds")
            
            obs, reward, done, info = super().step(action)
            self.update(wait=False)
            obs = self._flatten_obs(self.current_observations)

            # if self.use_aff_rewards: # TODO - decide design
            #     reward += self._aff_penalty(action)

            # check success
            if self._check_success(): # if success (including accidental success), terminate
                done = True
            
            print("current timestep: ", self.timestep)
            return obs, reward, done, info  
       
    def _update_current_observations(self):

        self.current_observations["sausage_pos"] = self.sausage_pos#[:-1]
        self.current_observations["grasping_sausage"] = self.grasping_sausage
        self.current_observations["sausage_cooked"] = self.sausage_cooked
        self.current_observations["sausage_in_bread"] = self.sausage_in_bread
        self.current_observations["sausage_on_pan"] = self.sausage_on_pan
        
        self.current_observations["pan_pos"] = self.pan_pos#[:-1]
        self.current_observations["grasping_pan"] = self.grasping_pan
        self.current_observations["pan_on_stove"] = self.pan_on_stove
        print("updated current observations", self.current_observations)

    def reset(self):
        
        print("Resetting...")
        # reset flags
        self.reward_given = False
        self.pan_on_stove, self.sausage_on_pan, self.sausage_cooked, self.sausage_in_bread = False, False, False, False
        self.grasping_pan = False
        self.grasping_sausage = False

        super().reset()
        self._update_current_observations()       
        
        # add some delay to get time to physically reset the environment
        time.sleep(5)
        # return flattened_obs
        return self._flatten_obs(self.current_observations)

    # def render(self, mode="human"):
    #     ...

    # def close(self):
    #     ...