
import numpy as np
import sys
import time
sys.path.append("..")
sys.path.insert(1, "/home/eeg/deoxys_control/deoxys")
from primitive_skills_noir import PrimitiveSkill
from environments.solo_envs.tablesetting_env import TablesettingEnv
from environments.solo_envs.realrobot_env_general import RealRobotEnvGeneral
from environments.realrobot_env_multi import RealRobotEnvMulti

from deoxys.utils.config_utils import (get_default_controller_config, verify_controller_config)
from deoxys.franka_interface import FrankaInterface
from deoxys.experimental.motion_utils import reset_joints_to

from utils.camera_utils import project_points_from_base_to_camera, get_camera_image
from deoxys.camera_redis_interface import CameraRedisSubInterface
import utils.transformation_utils as T
from utils.detection_utils import DetectionUtils
# from utils.detection_utils_eeg import DetectionUtils

from PIL import Image, ImageDraw, ImageFont, ImageTk
import cv2
import pdb
from getkey import getkey

import json

import tkinter as tk

camera_interfaces = {
    0 : CameraRedisSubInterface(camera_id=0),
    1 : CameraRedisSubInterface(camera_id=1),
    2 : CameraRedisSubInterface(camera_id=2),
}
for id in camera_interfaces.keys():
    camera_interfaces[id].start()

################# TEST MULTI ENV ####################
# env = RealRobotEnvMulti(
#     controller_type="OSC_POSE",
# )
# skill_selection_vec = np.zeros(env.skill.num_skills)
# skill_selection_vec[0] = 1
# params = [0.4, 0.0, 0.1]
# action = np.concatenate([skill_selection_vec, params])
# obs, reward, done, info = env.step(action)


################## TEST OBJ 3D POS ESTIMATE ######################
env = RealRobotEnvGeneral(
    env_name="Sweeping",
)
obs = env.reset()
handle_pos = obs["world_coords"]["red lego block"]
skill_selection_vec = np.zeros(env.num_skills)
skill_selection_vec[0] = 1
params = handle_pos
action = np.concatenate([skill_selection_vec, params])
obs, reward, done, info = env.step(action)


################## TEST Z DISCRETIZATION #####################
# # get world xy from top down view
# detection_utils = DetectionUtils()
# pix = (158., 315.)
# # world_xy = detection_utils.get_world_xy_from_topdown_view(pix)
# world_xy = np.array([0.5, 0.0, 0.25])
# print(world_xy)
# camera_id = 1
# detection_utils.get_points_on_z(world_xy=world_xy, camera_interface=camera_interfaces[camera_id], camera_id=camera_id, max_height=0.3)

############### TEST EEG SIDE DETECTION UTILS #################
# detection_utils = DetectionUtils()
# img0 = get_camera_image(camera_interface=camera_interfaces[0])
# img1 = get_camera_image(camera_interface=camera_interfaces[1])

# trim_low = [90, 130]
# trim_high = [450, 370]
# img2 = get_camera_image(camera_interface=camera_interfaces[2])
# img2 = img2[trim_low[1]:trim_high[1], trim_low[0]:trim_high[0]]

# pix0 = detection_utils.get_obj_pixel_coord(
#     img_array=img0,
#     texts=["red mug"],
#     save_filename="testing",
# )
# breakpoint()

# pix1 = detection_utils.get_obj_pixel_coord(
#     img_array=img1,
#     texts=["red mug"],
#     save_filename="testing",
# )
# breakpoint()

# pix2 = detection_utils.get_obj_pixel_coord(
#     img_array=img2,
#     texts=["red mug"],
#     save_filename="testing",
# )
# breakpoint()

# world = detection_utils.get_object_world_coords(
#     cam0_img=img0,
#     cam1_img=img1,
#     texts=["red mug"],
# )
# breakpoint()

# xy = detection_utils.get_world_xy_from_topdown_view(
#     pix_coords=(11.0, 35.0),
#     img_array=img2,
# )
# breakpoint()

################### CAMERA TEST ########################
# while True:
#     raw_image = get_camera_image(camera_interfaces[2])
#     rgb_image = raw_image[:,:,::-1] # convert from bgr to rgb
#     image = Image.fromarray(np.uint8(rgb_image))

#     # Create a draw object
#     draw = ImageDraw.Draw(image)

#     # Set the grid size and color
#     grid_size = 10
#     grid_color = (0, 255, 0)

#     # Draw the grid
#     width, height = image.size
#     for x in range(0, width, grid_size):
#         draw.line((x, 0, x, height), fill=grid_color)
#     for y in range(0, height, grid_size):
#         draw.line((0, y, width, y), fill=grid_color)

#     # Save the image with the grid
#     image.save("image_with_grid.jpg")
#     print("saved image")
#     time.sleep(0.5)


################### SKILLS WITHOUT ENVIRONMENT ####################
# setup robot interface
# controller_type = "OSC_POSE"
# controller_config = get_default_controller_config(controller_type)
# robot_interface = FrankaInterface(
#     general_cfg_file="config/charmander.yml",
#     control_freq=20,
# )

# # setup skills
# skill = PrimitiveSkill(
#     controller_type=controller_type,
#     controller_config=controller_config,
#     robot_interface=robot_interface,
#     waypoint_height=0.25,
#     workspace_limits={"x" : (0.35, 0.55), "y" : (-0.15, 0.25), "z" : (0.03, 0.45)},
# )

# skill._rehome_pos_quat(params=np.concatenate([skill.from_top_reset_eef_pos, skill.from_side_quat, [1, 1]]))
# skill._pick_from_top(params=np.array([0.5, 0.0, 0.2]))
# skill._pick_from_side(params=np.array([0.5, 0.0, 0.2]))
# skill._gripper_action([1])
# skill._rehome(params=np.append(skill.reset_joint_positions["from_top"], 0.0))
# skill._rehome(params=np.append(skill.reset_joint_positions["from_top"], 1.0))
# skill._rehome(params=np.append(skill.reset_joint_positions["from_top"], -1.0))
# skill._move_to(params=np.concatenate([[0.5, 0.0, 0.2], skill.from_top_quat, [1, 0]]))
# skill._reset_joints()







####################### TEST ENVIRONMENT ##########################
# env = RealRobotEnvGeneral(
#     env_name="TableSetting",
#     controller_type="OSC_POSE"
# )

# env.skill._gripper_action([1])
# move_params = np.concatenate([[0.5, 0.2, 0.2], env.skill.from_top_quat, [1.0, 0.0]])
# env.skill._move_to(params=move_params)
# robot_interface = env.robot_interface
# reset_joint_positions = env.skill.reset_joint_positions
# controller_type = "JOINT_IMPEDANCE"
# controller_cfg = get_default_controller_config(controller_type)
# action = reset_joint_positions + [1.0]
# time.sleep(1)
# while True:
#     if len(robot_interface._state_buffer) > 0:
#         print(robot_interface._state_buffer[-1].q)
#         print(robot_interface._state_buffer[-1].q_d)
#         print("-----------------------")

#         if (
#             np.max(
#                 np.abs(
#                     np.array(robot_interface._state_buffer[-1].q)
#                     - np.array(reset_joint_positions)
#                 )
#             )
#             < 1e-3
#         ):
#             break
#     robot_interface.control(controller_type=controller_type, action=action, controller_cfg=controller_cfg)

# breakpoint()
# env.robot_interface.control(
#     controller_type="JOINT_IMPEDANCE",
#     action=env.skill.reset_joint_positions + [1.0],
#     controller_cfg=env.controller_config,
# )
# breakpoint()

# obs = env.reset()
# print("initial obs\n", obs)
# world_coords = obs["world_coords"]

# # pick_pos = [cup_pos[0], cup_pos[1]+0.05, cup_pos[2]]
# skill_selection_vec = np.zeros(env.num_skills)

# skill_selection_vec[env.skill.skills["pick_from_top"]["default_idx"]] = 1

# obs, reward, done, info = env.step(action=np.concatenate([skill_selection_vec, pick_pos]))
# pdb.set_trace()



# reset_joint_positions = [
#     0.09162008114028396,
#     -0.19826458111314524,
#     -0.01990020486871322,
#     -2.4732269941140346,
#     -0.01307073642274261,
#     2.30396583422025,
#     0.8480939705504309,
# ]


# env = TablesettingEnv(
#     normalized_params=False
# )
# obs = env.reset()
# cup_pos = obs["shiny silver cup"]
# pick_pos = [cup_pos[0], cup_pos[1]+0.05, cup_pos[2]]
# pick_pos = [0.54657, 0.01063, 0.15]
# skill_selection_vec = np.zeros(env.num_skills)

# skill_selection_vec[env.skill.skills["pick_from_top"]["default_idx"]] = 1

# obs, reward, done, info = env.step(action=np.concatenate([skill_selection_vec, pick_pos]))
# pdb.set_trace()

# controller_type = "OSC_POSE"
# robot_interface = FrankaInterface(
#         general_cfg_file="config/charmander.yml",
#         control_freq=20,
# )

# reset_joints_to(robot_interface, reset_joint_positions)
# quat, pos = robot_interface.last_eef_quat_and_pos

# print("initial pos, quat", pos, quat)
# skills = PrimitiveSkill(
#     controller_type=controller_type,
#     controller_config=get_default_controller_config(controller_type),
#     robot_interface=robot_interface,
# )

# pos1 = [0.46871761, -0.12582393, 0.005]
# place_pos = [0.48755184, 0.13649393, 0.2]
# skills._pick_from_top(params=pos1)
# skills._pick_from_side(params=handle_pos)
# skills._pour_from_side(params=place_pos)

# pdb.set_trace()
# skills._place_from_top(params=place_pos)

# pos = [0.5, -0.1, 0.1]
# while True:
#     skills._pick_from_top(pos)
#     print("finished executing skill")
#     pdb.set_trace()

### Test everything in series ###
# pos = [0.5, 0.0, 0.1]
# skills._reset_joints()
# skills._pick_from_top(pos)
# pdb.set_trace()
# skills._place_from_top(pos)
# pdb.set_trace()
# skills._reset_joints()
# pdb.set_trace()
# skills._pick_from_side(pos)
# pdb.set_trace()
# skills._place_from_side(pos) 
# pdb.set_trace()
# skills._reset_joints()
# pdb.set_trace()
# skills._push_z(np.concatenate([pos, [-0.1, 90]]))
# pdb.set_trace()
# skills._push_xy(np.concatenate([pos, [0.05, -0.1, 45]]))
# pdb.set_trace()
# skills._reset_joints()
# pdb.set_trace()
# skills._wipe_xy(np.concatenate([pos, [0.05, -0.1, 0.0]]))
# pdb.set_trace()
# skills._draw_x(pos)
# pdb.set_trace()
# skills._reset_joints()
# pdb.set_trace()
# skills._screw(np.concatenate([pos, [45]]))
# pdb.set_trace()
# skills._reset_joints()
# pdb.set_trace()
# skills._pour_from_top(pos)
# pdb.set_trace()
# skills._pour_from_side(pos)
# pdb.set_trace()

# #### Test draw ####
# skills._pick_from_top([0.5, 0.0, 0.25])
# skills._draw_x([0.5, 0.0, 0.11])

#### Test screw ####
# skills._screw([0.5, 0.0, 0.2, 180])

#### Test pour ####
# skills._pick_from_top(params=[0.5, 0.0, 0.2])
# skills._pour_from_top(params=[0.5, 0.0, 0.2])

