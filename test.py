
import numpy as np
import sys
import time
sys.path.append("..")
sys.path.insert(1, "/home/eeg/deoxys_control/deoxys")
from primitive_skills_noir import PrimitiveSkill
from environments.tablesetting_env import TablesettingEnv

from deoxys.utils.config_utils import (get_default_controller_config, verify_controller_config)
from deoxys.franka_interface import FrankaInterface
from deoxys.experimental.motion_utils import reset_joints_to

from camera_utils import project_points_from_base_to_camera, get_camera_image
from deoxys.camera_redis_interface import CameraRedisSubInterface

from PIL import Image, ImageDraw
import cv2
import pdb
from getkey import getkey

reset_joint_positions = [
    0.09162008114028396,
    -0.19826458111314524,
    -0.01990020486871322,
    -2.4732269941140346,
    -0.01307073642274261,
    2.30396583422025,
    0.8480939705504309,
]


env = TablesettingEnv()
pos = env._update_obj_positions()




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

# # pos = [0.43794176, -0.04834167, 0.3170166]
# pos = [0.40395209, -0.15580851, 0.10464783] # teapot on drawer
# # pos = [0.44395209, -0.10580851, 0.05464783] # teapot on table

# skills._move_to(params=np.concatenate([pos, quat, [-1]]))
# final_quat, final_pos = robot_interface.last_eef_quat_and_pos
# error = pos - final_pos.flatten()
# print("goal pos", pos)
# print("final pos", final_pos)
# print("error", error)

# skills._pick_from_side(params=pos)
# skills._place_from_side(params=pos)
# skills._push_z(params=np.concatenate([pos, [-0.1, 45.0]]))

# skills._pick(params=np.concatenate([pos, left_quat]))
# skills._rehome(gripper_action=1, gripper_quat=left_quat)

# skills._rehome(gripper_action=-1, gripper_direction="down")

# goal_pos = [0.5764149, 0.1, 0.16015941]
# goal_quat = [0.9998497, 0.00907074, 0.01465143, 0.00190753]

# skills._move_to(
#     params = np.concatenate([goal_pos, goal_quat, [-1]])
# )

# final_quat, final_pos = robot_interface.last_eef_quat_and_pos
# final_pos = final_pos.flatten()
# print("final eef pos and quat", final_pos, final_quat)
# print("position error", final_pos - goal_pos)