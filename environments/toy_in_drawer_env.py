
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


class ToyInDrawerEnv(RealRobotEnv):
    """Custom Environment that follows gym interface."""

    # metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        reward_scale=1.0,
        horizon=5000,
        use_skills=True,
        normalized_params=True,
        use_aff_rewards=False,
        # num_toys=1,
        use_yaw=True,
        keys=None,
        detector_params=detector_params,
        visualize_params=True,
        reset_joint_pos=None,
        workspace_limits={"x" : (0.40, 0.55), "y" : (-0.10, 0.25), "z" : (0.01, 0.2)}, # adjust this
        use_human_feedback=True,
    ): 
        self.keys = keys
        if self.keys is None:
            self.keys = [
                "eef_pos", "gripper_state",
                "toy_pos", "grasping_toy", "toy_in_drawer",
                "drawer_delta", #"drawer_handle_pos"
            ] 

        # initialize observations
        self.current_observations = {}
        self.toy_pos = np.zeros(3)
        self.grasping_toy = False
        self.toy_in_drawer = False
        self.drawer_delta = 0.1
        self.drawer_closed = False
        self.missing_toy = False

        self.visualize_params = visualize_params
        self.use_human_feedback = use_human_feedback

        # HSV thresholds for objects
        self.hsv_thresh = { 
            "toy1" : { # pink toy
                "low" : np.array([140, 100, 10]), 
                "high" : np.array([240, 255, 255]),
            },
            "toy2" : { # brown toy
                "low" : np.array([0, 160, 50]),
                "high" : np.array([40, 255, 255]),
            },
            "drawer_handle" : {
                "low" : np.array([100, 130, 50]), # blue tape handle 
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
                    1 : "place",
                    2 : "push_y"
                },
                "aff_pos_thresh" : {
                    "pick" : 0.03,
                    "place" : 0.03,
                    "push_y" : 0.05
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
        # self.drawer_pos = drawer_position
        self.drawer_push_start_pos = np.array([0.475, 0.16, 0.025]) # good position to start pushing
        self.toy_place_pos = np.array([0.5, -0.035, 0.16]) # ideal position to drop the object
        self.drawer_handle_closed_pos = -0.055 # y-position of drawer handle when drawer is closed
        self.drawer_handle_pos = 0.0
        self.drawer_closed_thresh = 0.02 

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

        return self.toy_in_drawer and self.drawer_closed

    def _check_failure(self):
        """
        Task failed if:
            - drawer is closed and toy is still visible
            # - drawer is opened and toy is not inside but toy is missing
        """
        cond1 = self.drawer_closed and not self.missing_toy
        # cond2 = self.missing_toy and not self.toy_in_drawer
        failure = cond1 #or cond2
        if failure:
            print(f"~~~~Task fail :(~~~~")
        return failure

    def _update_keypoints(self): 
        """
        Update self.keypoints dic according to current task state 
        """

        # if holding toy, should place it in drawer
        if np.any(self.grasping_toy):
            self.keypoints["pick"] = []
            self.keypoints["place"] = [self.toy_place_pos]
            self.keypoints["push"] = []

        # if all toys in drawer, should close drawer
        elif self.toy_in_drawer:
            self.keypoints["pick"] = []
            self.keypoints["place"] = []
            self.keypoints["push"] = [np.concatenate([self.drawer_push_start_pos, [self.drawer_delta]])]
            
        # if toy is not in drawer, should pick toy
        else:
            self.keypoints["pick"] = [self.toy_pos]
            self.keypoints["place"] = []
            self.keypoints["push"] = []

    def _update_task_status(self, wait=True): # TODO - make sure this works
        """
        Updates task status using camera input
        """
        print("\nUpdate task status")
        
        if wait:
            # get drawer handle position and update drawer_closed state
            self.drawer_handle_pos = self.get_object_pos(self.hsv_thresh["drawer_handle"]["low"], self.hsv_thresh["drawer_handle"]["high"], "drawer_handle", wait=True)
            self.drawer_delta = self.drawer_handle_pos[1] - self.drawer_handle_closed_pos # scalar (y direction)
            self.drawer_closed = self.drawer_delta < self.drawer_closed_thresh

            # if toy is in drawer, only look for it once   
            if self.toy_in_drawer:
                self.toy_pos = self.get_object_pos(self.hsv_thresh["toy1"]["low"], self.hsv_thresh["toy1"]["high"], "toy1", wait=False, dilation=True) # apply dilation to mask to account for nonuniform colors
                self.grasping_toy = False
                if self.toy_pos is None:
                    self.missing_toy = True
                    self.toy_pos = np.array([0.45, -0.2, 0.15])
                else:
                    self.missing_toy = False
            
            # if toy is not in drawer, look until we find it        
            else:
                self.toy_pos = self.get_object_pos(self.hsv_thresh["toy1"]["low"], self.hsv_thresh["toy1"]["high"], "toy1", wait=True, dilation=True)
                err = self.toy_place_pos - self.toy_pos
                if not self.toy_in_drawer and abs(err[1]) < 0.03 and abs(err[0]) < 0.12:
                    self.toy_in_drawer = True
                # self.toy_in_drawer = abs(err[1]) < 0.03 and abs(err[0]) < 0.12
                self.grasping_toy = self.toy_pos[2] > 0.2
        else:
            # get drawer handle position and update drawer_closed state
            drawer_handle_pos = self.get_object_pos(self.hsv_thresh["drawer_handle"]["low"], self.hsv_thresh["drawer_handle"]["high"], "drawer_handle", wait=False)
            if drawer_handle_pos is not None:
                self.drawer_handle_pos = drawer_handle_pos
            self.drawer_delta = self.drawer_handle_pos[1] - self.drawer_handle_closed_pos # scalar (y direction)
            self.drawer_closed = self.drawer_delta < self.drawer_closed_thresh
            toy_pos = self.get_object_pos(self.hsv_thresh["toy1"]["low"], self.hsv_thresh["toy1"]["high"], "toy1", wait=False, dilation=True)
            if toy_pos is not None:
                self.toy_pos = toy_pos
            err = self.toy_place_pos - self.toy_pos
            if not self.toy_in_drawer and abs(err[1]) < 0.03 and abs(err[0]) < 0.12:
                self.toy_in_drawer = True
            # self.toy_in_drawer = abs(err[1]) < 0.03 and abs(err[0]) < 0.12
            self.grasping_toy = self.toy_pos[2] > 0.2
        # TODO - make sure this works

    def update(self, wait=True):
        """
        Updates keypoints, object positions, and task state
        """
        self._update_task_status(wait=wait)
        self._update_keypoints()
        self._update_current_observations()

    def compute_affordance_score(self, action): 

        thresholds = np.array([
            np.array([0.05, 0.05, 0.01]), # toy pick position
            np.array([0.12, 0.05, 0.04]), # toy place position
            # np.array([0.075, 0.05, 0.025]), # drawer push position
        ])
        good_params = np.array([ 
            self.toy_pos, # toy position
            self.toy_place_pos, # toy place position
            # self.drawer_push_start_pos, # drawer push position
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
            human_feedback_reward += affordance_reward # NOTE - change this 
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
            self.update(wait=True)
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
                time.sleep(0.05)
            
            self.update()
            obs = self._flatten_obs(self.current_observations)
            
            # process rewards
            reward = self._get_reward()
            print("reward before affordance", reward)
            if self.use_aff_rewards:
                reward -= aff_penalty
            print("reward after affordance", reward)

            # check termination condition
            if self._check_success() or self._check_failure(): # if success (including accidental success), terminate
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

            # check termination condition
            if self._check_success() or self._check_failure(): # if success (including accidental success), terminate
                done = True
            
            print("current timestep: ", self.timestep)
            return obs, reward, done, info  
   
    def _update_current_observations(self):
 
        self.current_observations["toy_pos"] = self.toy_pos
        self.current_observations["grasping_toy"] = self.grasping_toy
        self.current_observations["toy_in_drawer"] = self.toy_in_drawer
        self.current_observations["drawer_delta"] = self.drawer_delta
        
        print("updated current observations", self.current_observations)

    def reset(self):
        
        # reset flags
        self.reward_given = False
        self.toy_in_drawer = False
        self.drawer_closed = False
        self.grasping_toy = False
        self.missing_toy = False

        super().reset()        
        self._update_current_observations()

        # add some delay to physically reset the environment
        time.sleep(5)
        return self._flatten_obs(self.current_observations)

