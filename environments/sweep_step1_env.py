import numpy as np
import time

import sys
sys.path.insert(1, "/home/eeg/MAPLE-EF")
from environments.realrobot_env import RealRobotEnv
# from detection_utils import get_object_world_coords, get_obj_pixel_coord

import pdb
import cv2
from getkey import getkey

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.filterByColor = False
detector_params.filterByCircularity = False
detector_params.filterByConvexity = False
detector_params.filterByInertia = False
detector_params.minArea = 75.0
detector_params.maxArea = 15000.0
#### NOTE - This environment is unfinished!


class SweepStep1Env(RealRobotEnv):
    """Custom Environment that follows gym interface."""

    # metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        reward_scale=1.0,
        horizon=5000,
        use_skills=True,
        normalized_params=True,
        use_aff_rewards=False,
        use_yaw=True,
        keys=None,
        detector_params=detector_params,
        visualize_params=True,
        reset_joint_pos=None,
        workspace_limits={"x" : (0.40, 0.55), "y" : (-0.10, 0.25), "z" : (0.13, 0.2)}, 
        use_human_feedback=True,
    ): 
        self.keys = keys
        if self.keys is None:
            self.keys = [
                "eef_pos", "gripper_state",
                "broom_pos", "grasping_broom",
                "trash_pos",
            ] 

        # initialize observations
        self.current_observations = {}
        self.broom_pos = np.zeros(3)
        self.grasping_broom = False
        self.trash_pos = np.zeros(3) 
        self.trash_cleaned = False

        self.visualize_params = visualize_params
        self.use_human_feedback = use_human_feedback

        # HSV thresholds for objects
        self.hsv_thresh = { 
            "trash" : { # brown elephant toy
                "low" : np.array([0, 160, 50]),
                "high" : np.array([15, 255, 255]),
            },
            "broom_handle" : {
                "low" : np.array([90, 110, 50]), # blue tape handle 
                "high" : np.array([120, 255, 255]),
            }
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
                    1 : "push_x",
                },
                "aff_pos_thresh" : {
                    "pick" : 0.03,
                    "push_x" : 0.05
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
        self.dust_bin_pos = np.array([0.57, 0.04, 0.0]) # center of dust bin in xy
        self.update()

    def reward(self,):
        """
        Reard function for task. Returns environment reward only.

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
        return self.grasping_broom

    def _update_keypoints(self): 
        """
        Update self.keypoints dic according to current task state 
        """

        # if holding broom, should push near the toy
        if self.grasping_broom:
            self.keypoints["pick"] = []
            self.keypoints["push"] = [self.trash_pos]

        # otherwise, should pick up broom
        else:
            self.keypoints["pick"] = [self.broom_pos]
            self.keypoints["push"] = []
        
    def _update_task_status(self, wait=True): # TODO - make sure this works
        """
        Updates task status using camera input
        """
        print("\nUpdate task status")
        if wait:
            self.broom_pos = self.get_object_pos(self.hsv_thresh["broom_handle"]["low"], self.hsv_thresh["broom_handle"]["high"], "broom_handle", wait=True)
            self.grasping_broom = self.broom_pos[2] > 0.2
            self.trash_pos = self.get_object_pos(self.hsv_thresh["trash"]["low"], self.hsv_thresh["trash"]["high"], "trash", wait=True)
        
        else: # for algorithms where occlusions happen often (e.g. SAC, TAMER, etc.), use this option
            broom_pos = self.get_object_pos(self.hsv_thresh["broom_handle"]["low"], self.hsv_thresh["broom_handle"]["high"], "broom_handle", wait=False)
            trash_pos = self.get_object_pos(self.hsv_thresh["trash"]["low"], self.hsv_thresh["trash"]["high"], "trash", wait=False)
            if broom_pos is not None:
                self.broom_pos = broom_pos
            if trash_pos is not None:
                self.trash_pos = trash_pos
            self.grasping_broom = self.broom_pos[2] > 0.2

        self.trash_cleaned = np.linalg.norm(self.trash_pos - self.dust_bin_pos) < 0.05 # TODO tune this
        # TODO - make sure this works

    def update(self, wait=True):
        """
        Updates keypoints, object positions, and task state
        """
        self._update_task_status(wait=wait)
        self._update_keypoints()
        self._update_current_observations()

    def compute_affordance_score(self, action): 

        thresholds = np.array([ # TODO - discuss this
            np.array([0.05, 0.05, 0.05]), # broom handle
            np.array([0.1, 0.1, 1]), # trash position
            # np.array([0, 0, 0]), # dust bin position
        ])
        good_params = np.array([ 
            self.broom_pos,
            self.trash_pos,
            # self.dust_bin_pos,
        ])
        
        # unnormalize params
        if self.normalized_params:
            action = self.skill.unnormalize_params(action)
        params = action[self.num_skills:]

        good = np.any(np.all(np.abs(good_params - params[:3]) < thresholds, axis=1))
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

        self.human_feedback_request(action)

        # get affordance reward
        affordance_reward = self.compute_affordance_score(action)

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
            human_feedback_reward += affordance_reward # NOTE
            print(human_feedback_reward)
        return human_feedback_reward, go_signal

    def human_reward_after_execution(self):
        print("=====human reward call=======")
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
            print("reward before affordance", reward)
            if self.use_aff_rewards:
                reward -= aff_penalty
            print("reward after affordance", reward)

            # check termination condition
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
            action[:-1] *= 0.8
 
            # make sure action does not send robot out of workspace bounds
            action_in_bounds = self._check_action_in_ws_bounds(action)
            if not action_in_bounds:
                action[:-1] = 0.0
                print("Action out of bounds")
            
            obs, reward, done, info = super().step(action)
            self.update(wait=False)
            obs = self._flatten_obs(self.current_observations)

            # check termination condition
            if self._check_success(): # if success (including accidental success), terminate
                done = True
            
            print("current timestep: ", self.timestep)
            return obs, reward, done, info  
   
    def _update_current_observations(self):
        self.current_observations["broom_pos"] = self.broom_pos
        self.current_observations["grasping_broom"] = self.grasping_broom
        self.current_observations["trash_pos"] = self.trash_pos[:2]
        
        print("updated current observations", self.current_observations)

    def reset(self):
        
        # reset flags
        self.reward_given = False
        self.grasping_broom = False
        self.trash_cleaned = False

        super().reset()        
        self._update_current_observations()

        # add some delay to physically reset the environment
        time.sleep(5)
        return self._flatten_obs(self.current_observations)
