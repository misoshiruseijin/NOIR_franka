import numpy as np

import sys
sys.path.insert(1, "/home/eeg/MAPLE-EF")
from environments.realrobot_env import RealRobotEnv

from getkey import getkey, keys

import pdb

class ReachingEnv(RealRobotEnv):
    """Custom Environment that follows gym interface."""

    # metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        reward_scale=1.0,
        horizon=1000,
        use_skills=True,
        normalized_params=True,
        use_aff_rewards=False,
        target_pos=(0.45, 0.15),
    ): 
        self.keys = ["eef_pos"]
        super().__init__(
            horizon=horizon,
            use_skills=use_skills,
            controller_type="OSC_YAW",
            general_cfg_file="config/charmander.yml",
            control_freq=20,
            skill_config={
                "waypoint_height" : 0.25,
                "workspace_limits" : {"x" : (0.3, 0.55), "y" : (-0.15, 0.25), "z" : (0.03, 0.3)},
                "yaw_limits" : (-0.1*np.pi, 0.1*np.pi),
                "idx2skill" : {
                    0 : "move_to",
                    # 0 : "pick",
                    # 1 : "place",
                    # 2 : "push"
                },
                "aff_pos_thresh" : {
                    "move_to" : 0.03,
                    # "pick" : 0.03,
                    # "place" : 0.03,
                    # "push" : 0.05
                },
            },
            gripper_thresh=0.04,
            normalized_params=True,
        )
        
        self.reward_scale = reward_scale
        self.use_skills = use_skills
        self.num_skills = self.skill.num_skills
        self.normalized_params = normalized_params
        self.use_aff_rewards = use_aff_rewards

        self.target_pos = target_pos

        self.gripper_state = -1

        # flags for reward computation
        self.reward_given = False


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

        eef_pos = self.current_observations["eef_pos"]
        in_target = np.all(np.abs(eef_pos[:-1] - self.target_pos) < 0.05)
        return in_target

    def _update_keypoints(self):
        """
        Update self.keypoints dic according to current task state 
        """
        pass

    def step(self, action):

        # if using skills
        if self.use_skills:

            num_timesteps = 0
            
            done, skill_done, skill_success = False, False, False

            if self.normalized_params:
                action = self.skill.unnormalize_params(action)

            while not done and not skill_done:
                action_ll, skill_done, skill_success = self.skill.get_action(action)
                obs, _, done, info = super().step(action_ll)
                num_timesteps += 1

            info["num_timesteps"] = num_timesteps

            if done: # horizon exceeded
                print(f"-----------Horizon {self.timestep} Reached--------------")

            # process rewards
            reward = self._get_reward()
            if reward > 0 and not skill_success: # zero the reward if skill failed or was terminated due to horizon
                print("Success on accident. Reward = 0")
                reward = 0.0

            if self.use_aff_rewards:
                reward += self._aff_penalty(action)

            # check success
            if self._check_success(): # if success (including accidental success), terminate
                done = True

        # if using low level action commands
        else:
            if action.shape[0] == 7:
                # ignore roll and pitch
                action[3:5] = 0
            elif action.shape[0] == 5:
                action = np.concatenate([action[:3], np.zeros(2), action[3:]])

            # make sure action does not send robot out of workspace bounds
            action_in_bounds = self._check_action_in_ws_bounds(action)
            if not action_in_bounds:
                action[:-1] = 0.0
                print("Action out of bounds")
            
            obs, reward, done, info = super().step(action)
        
        return obs, reward, done, info

    def reset(self):
        
        # reset flags
        self.reward_given = False

        # reset robot joint positoins and get eef_proprio (position, axis angle orientation, gripper_state)
        observation = super().reset()        

        # TODO - add some delay so there is time to physically reset object positions?

        return observation

    # def render(self, mode="human"):
    #     ...

    # def close(self):
    #     ...