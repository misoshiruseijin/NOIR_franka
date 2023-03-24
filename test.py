
import numpy as np
import sys
import time
sys.path.append("..")
sys.path.insert(1, "/home/eeg/deoxys_control/deoxys")
from primitive_skills_noir import PrimitiveSkill

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

controller_type = "OSC_POSE"
robot_interface = FrankaInterface(
        general_cfg_file="config/charmander.yml",
        control_freq=20,
)

reset_joints_to(robot_interface, reset_joint_positions)

skills = PrimitiveSkill(
    controller_type=controller_type,
    controller_config=get_default_controller_config(controller_type),
    robot_interface=robot_interface,
)

goal_pos = [0.5764149, -0.12984779, 0.56015941]
goal_quat = [0.7913314, 0.5174134, 0.28115153, 0.16441339]

skills._move_to(
    params = np.concatenate([goal_pos, goal_quat, [-1]])
)

print("final eef pos and quat", robot_interface.last_eef_quat_and_pos)