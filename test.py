
import numpy as np
import sys
import time
sys.path.append("..")
sys.path.insert(1, "/home/eeg/deoxys_control/deoxys")
from primitive_skills_noir import PrimitiveSkill
from environments.solo_envs.tablesetting_env import TablesettingEnv
from environments.solo_envs.realrobot_env_general import RealRobotEnvGeneral

from deoxys.utils.config_utils import (get_default_controller_config, verify_controller_config)
from deoxys.franka_interface import FrankaInterface
from deoxys.experimental.motion_utils import reset_joints_to

from utils.camera_utils import project_points_from_base_to_camera, get_camera_image
from deoxys.camera_redis_interface import CameraRedisSubInterface
import utils.transformation_utils as T

from PIL import Image, ImageDraw
import cv2
import pdb
from getkey import getkey


env = RealRobotEnvGeneral(
    env_name="TableSetting",
    controller_type="OSC_POSE"
)

obs = env.reset()
print("initial obs\n", obs)
world_coords = obs["world_coords"]

# pick_pos = [cup_pos[0], cup_pos[1]+0.05, cup_pos[2]]
skill_selection_vec = np.zeros(env.num_skills)

skill_selection_vec[env.skill.skills["pick_from_top"]["default_idx"]] = 1

obs, reward, done, info = env.step(action=np.concatenate([skill_selection_vec, pick_pos]))
pdb.set_trace()



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

