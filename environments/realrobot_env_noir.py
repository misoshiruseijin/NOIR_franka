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
from camera_utils import project_points_from_base_to_camera, get_camera_image
from utils.detection_utils import DetectionUtils
from getkey import getkey, keys

import pdb
from pynput import keyboard

class RealRobotEnv(gym.Env):
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
        detector_config={
            "texts" : [],
            "thresholds" : []
        },
        ignore_done=False,
        gripper_thresh=0.04, # gripper q below this value is considered "closed"
        normalized_params=True,
        reset_joint_pos=None,
    ): 

        super().__init__()
        
        self.timestep = 0
        self.ignore_done = ignore_done

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
        
        # setup detection-related things
        self.detection_utils = DetectionUtils()
        self.texts = detector_config["texts"]
        self.thresholds = detector_config["thresholds"]

        # setup skills
        self.skill = PrimitiveSkill(
            controller_type=self.controller_type,
            controller_config=self.controller_config,
            robot_interface=self.robot_interface,
            waypoint_height=skill_config["waypoint_height"],
            workspace_limits=self.workspace_limits,
            idx2skill=skill_config["idx2skill"],
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

        self._add_robot_state_to_cur_obs()
        self.current_infos = {}
        obs_to_use = {key : self.current_observations[key] for key in self.keys}
        flattened_obs = self._flatten_obs(obs_to_use)
        self.obs_dim = flattened_obs.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)

    def reward(self,):
        """
        Reard function for task. Returns environment reward only.
        Environment reward is given when task is successfully complete:
            pan placed on stove -> sausage placed on pan -> sausage placed in bread 
                in that order

        Returns:
            reward (float) : environment reward
        """
        
        raise NotImplementedError

    def step(self, action, execute=True): # TODO this should be changed to use skills only - we don't need low level control

        """
        Commands robot to execute skill corresponding to input action
        
        Args:
            action : skill selection vector concatenated with skill parameters
        """ 

        self.skill.execute_skill(action)

        reward, done, info = self._post_action(action)        
        self._add_robot_state_to_cur_obs()
        obs = {key : self.current_observations[key] for key in self.keys}
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

            # if self.done: # NOTE there is no check for num steps - this flag should be set in child class if needed
            #     self.reset()
            return reward, False, {}

    def _reset_internal(self):
        """
        Reset internal parameters
        """
        self.timestep = 0 # TODO - don't need?
        self.done = False
        
    def reset(self): # TODO - flattening should happen in wrapper if needed
        """
        Reset robot to home joint configuration and returns observation
        """
        self._reset_internal()
        reset_joints_to(self.robot_interface, self.skill.reset_joint_positions)
        time.sleep(0.5)
        self._add_robot_state_to_cur_obs()
        obs = {key : self.current_observations[key] for key in self.keys}
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

    # def _check_action_in_ws_bounds(self, action): # shouldn't need this if not using low level control
    #     """
    #     Checks if action will keep robot within workspace boundaries
    #     """
    #     sf = 3 # TODO - tune this: higher is more cautious
    #     scale = self.controller_config["action_scale"]["translation"] # controller internally scales translation action inputs by this value
    #     scaled_action = scale * action
    #     eef_pos = self.current_observations["eef_pos"]
        
    #     cur_x_in_bounds = self.workspace_limits["x"][0] < eef_pos[0] < self.workspace_limits["x"][1]
    #     cur_y_in_bounds = self.workspace_limits["y"][0] < eef_pos[1] < self.workspace_limits["y"][1]
    #     cur_z_in_bounds = self.workspace_limits["z"][0] < eef_pos[2] < self.workspace_limits["z"][1]

    #     # if already outside workspace, allow action if it brings it inside workspace
    #     if not cur_x_in_bounds:
    #         print("EEF already out of bounds in x")
    #         x_good = (
    #             eef_pos[0] < self.workspace_limits["x"][0] and action[0] > 0
    #             or eef_pos[0] > self.workspace_limits["x"][1] and action[0] < 0
    #         )
    #     else:
    #         x_good = self.workspace_limits["x"][0] < eef_pos[0] + sf * scaled_action[0] / self.control_freq < self.workspace_limits["x"][1]

    #     if not cur_y_in_bounds:
    #         print("EEF already out of bounds in y")
    #         y_good = (
    #             eef_pos[1] < self.workspace_limits["y"][0] and action[1] > 0
    #             or eef_pos[1] > self.workspace_limits["y"][1] and action[1] < 0
    #         )
    #     else:
    #         y_good = self.workspace_limits["y"][0] < eef_pos[1] + sf * scaled_action[1] / self.control_freq < self.workspace_limits["y"][1]

    #     if not cur_z_in_bounds:
    #         print("EEF already out of bounds in z")
    #         z_good = (
    #             eef_pos[2] < self.workspace_limits["z"][0] and action[2] > 0
    #             or eef_pos[2] > self.workspace_limits["z"][1] and action[2] < 0
    #         )
    #     else:
    #         z_good = self.workspace_limits["z"][0] < eef_pos[2] + sf * scaled_action[2] / self.control_freq < self.workspace_limits["z"][1]
        
    #     print("action is ok in x y z", x_good, y_good, z_good)
    #     return x_good and y_good and z_good

    def _update_param_visualization(self, action): # TODO - check the case where multiple positoins are given and fix colors
        """
        Given coordinates in 3d world, updates visualization of 3d points in camera image.
        First point is displayed in red and the second point is displayed in blue.
        Yaw angle is shown as a green line, where the direction of the line is the direction the front of the end effector will point to
        Args:
            action : one-hot skill selection vector concatentated with params
        """
        # get skill 
        if self.normalized_params:
            action = self.skill.unnormalize_params(action)
        skill_idx = np.argmax(action[:self.num_skills])
        skill_name = self.skill.idx2skill[skill_idx]
        params = action[self.num_skills:]
        print("params", params)

        if skill_name == "push":
            points = np.vstack([params[0:3], params[3:6]])
            yaw = params[6] if self.use_yaw else 0.0
        else:
            points = params[0:3].reshape((1,3))
            yaw = params[3] if self.use_yaw else 0.0

        points_on_table = points.copy()
        points_on_table[:,-1] = 0
        u_yaw = points.copy()
        u_yaw[:,0] += 0.05 * np.cos(yaw)
        u_yaw[:,1] += 0.05 * np.sin(yaw)
        # if visualize: update camera image with dot and arrow
        for camera_id in range(2):
            # get 2d projection of points in pos_list
            pixels = project_points_from_base_to_camera(
                points=points,
                camera_id=camera_id,
                camera_height=480,
                camera_width=640,
            )
            pixels_on_table = project_points_from_base_to_camera(
                points=points_on_table,
                camera_id=camera_id,
                camera_height=480,
                camera_width=640,
            )
            pixels_for_yaw = project_points_from_base_to_camera(
                points=u_yaw,
                camera_id=camera_id,
                camera_height=480,
                camera_width=640,
            )

            # get image and plot red circle at the pixel location
            im = get_camera_image(self.camera_interfaces[camera_id])
            image = Image.fromarray(im)
            im_raw = im.copy()
            draw = ImageDraw.Draw(image)

            # write skill name on image
            # specified font size
            font = ImageFont.truetype('arial.ttf', 48)             
            # drawing text size
            txt = skill_name
            if skill_name in ["push_x", "push_y"]:
                txt += f" {str(params[3])[:5]}"
            draw.text((15, 15), txt, font = font, align ="left") 
            
            r = 10 # size of circle

            colors = ["red", "blue"]
            for i, (pix, pix0, pix_yaw) in enumerate(zip(pixels, pixels_on_table, pixels_for_yaw)):
                x, y = pix[0], pix[1]
                y, x = x, y
                x0, y0 = pix0[0], pix0[1]
                y0, x0 = x0, y0
                x_yaw, y_yaw = pix_yaw[0], pix_yaw[1]
                y_yaw, x_yaw = x_yaw, y_yaw
                draw.line([x0, y0, x, y], fill=colors[i], width=2)
                draw.line([x_yaw, y_yaw, x, y], fill="green", width=10)
                draw.ellipse((x - r, y - r, x + r, y + r), fill=colors[i])
                im = np.array(image).astype(np.uint8)

            # save image
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            im_raw = cv2.cvtColor(im_raw, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(f'param_vis_images/camera{camera_id}_with_params.png', im[:, :, ::-1])
            cv2.imwrite(f'param_vis_images/camera{camera_id}_raw.png', im_raw[:, :, ::-1])

    # def human_feedback_request(self, action): # don't need this?

    #     """
    #     Algorithm calls this function to ask for human feedback. A visualization of parameters is updated
    #     Args:
    #         pos_list : list of positions to display (at most 2 elements)
    #         yaw : yaw angle
    #     """
    #     self._update_param_visualization(action)

    # TODO - need new function that: presents objects -> human choose obj -> present relevant skills -> human chooses skill -> human chooses param -> execute

    def _add_robot_state_to_cur_obs(self):
        robot_state = self._get_current_robot_state()
        self.current_observations.update(robot_state)

    @abstractmethod
    def update(self):
        raise NotImplementedError()    

    def get_object_pos(self, obj_name="", wait=True): # TODO - this needs to be updated with OWL-VIT
        """
        Finds position of an object in world coordinates
        
        Args:
            hsv_low : lower bound of HSV values of object of interest
            hsv_high : upper bound of HSV values of object of interest
        """
        return self.detection_utils.get_object_world_coords(self.camera_interfaces[0], self.camera_interfaces[1], texts=self.texts, thresholds=self.thresholds, wait=wait)
    
    


    """
    TODO - below functions should go in wrapper if needed
    """
    def _flatten_obs(self, obs_dict, verbose=False): 
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        # import pdb; pdb.set_trace()
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)

    def seed(self, seed=None): 
        """
        Utility function to set numpy seed

        Args:
            seed (None or int): If specified, numpy seed to set

        Raises:
            TypeError: [Seed must be integer]
        """
        # Seed the generator
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")