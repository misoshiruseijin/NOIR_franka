
import numpy as np
import sys
import time
import argparse
import json

sys.path.append("..")
sys.path.insert(1, "/home/eeg/deoxys_control/deoxys")
from primitive_skills_noir import PrimitiveSkill
from environments.tablesetting_env import TablesettingEnv
from environments.tofu_env import TofuEnv
from environments.whiteboard_env import WhiteboardEnv
from environments.realrobot_env_noir import RealRobotEnv


from deoxys.utils.config_utils import (get_default_controller_config, verify_controller_config)
from deoxys.franka_interface import FrankaInterface
from deoxys.experimental.motion_utils import reset_joints_to

from camera_utils import project_points_from_base_to_camera, get_camera_image
from deoxys.camera_redis_interface import CameraRedisSubInterface

from PIL import Image, ImageDraw
import cv2
import pdb
from getkey import getkey

import pdb

reset_joint_positions = [
    0.09162008114028396,
    -0.19826458111314524,
    -0.01990020486871322,
    -2.4732269941140346,
    -0.01307073642274261,
    2.30396583422025,
    0.8480939705504309,
]

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

# # Get relevant skill names dictionary from json file
# with open('config/obj2skill.json') as json_file:
#     obj2skill = json.load(json_file)
    
obj2skill = {
    "red bowl" : ["pick_from_top", "pick_from_side", "push_xy"],
    "light blue bowl" : ["pick_from_top", "pick_from_side", "push_xy"], 
    "shiny silver cup" : ["pick_from_top", "pick_from_side", "push_xy"],
    "red and blue spoon" : ["pick_from_top", "pick_from_side", "push_xy"],
    "red spatula" : ["pick_from_top", "pick_from_side", "push_xy"],
    "dark blue eraser" : ["pick_from_top", "pick_from_side", "push_xy"],
    "none_tablesetting" : ["place_from_top", "place_from_side", "reset_joints"],
    "none_tofu" : ["push_z", "place_from_top", "reset_joints"],
    "none_whiteboard" : ["wipe_xy", "place_from_top", "reset_joints"]
}

task2env = {
    "tablesetting" : TablesettingEnv,
    "tofu" : TofuEnv,
    "whiteboard" : WhiteboardEnv,
}

def terminal_shared_autonomy(env:RealRobotEnv, obj_names):
    
    obs = env.reset()
    
    # ask for human input
    while True:
        obj_name = ""
        skill_name = ""

        # object selection
        while obj_name not in obj_names:
            obj_name = input(f"select object from {obj_names}:\n")

        # skill selection
        skill_options = obj2skill[obj_name]
        while skill_name not in skill_options:
            skill_name = input(f"choose skill from {skill_options}:\n")

        # param input
        params = []
        param_dim = env.skill.skills[skill_name]["num_params"]
        while not len(params) == param_dim:
            if "none" not in obj_name:
                print(f"Position of {obj_name} is {obs[obj_name]}")
            params = input(f"input skill parameters ({param_dim} floats separated by spaces):\n")
            params = [float(x) for x in params.split()]

        # execute the skill
        print(f"Executing skill {skill_name}({params})")
        skill_idx = env.skill.skills[skill_name]["default_idx"]
        skill_selection_vec = np.zeros(env.num_skills)
        skill_selection_vec[skill_idx] = 1
        action = np.concatenate([skill_selection_vec, params])
        env.step(action)

def run_demo(task_name):
    env_class = task2env[task_name]
    env = env_class(
        normalized_params=False,
    )

    # get object names
    obj_names = env.texts.copy()
    obj_names.append(f"none_{task_name}")
    terminal_shared_autonomy(env, obj_names)

# def table_setting_demo():
#     # intialize environment and get initial observation
#     env = TablesettingEnv(
#         normalized_params=False,
#     )
#     obs = env.reset()
    
#     # get object names
#     obj_names = env.texts.copy()
#     obj_names.append("none_tablesetting")

# def whiteboard_demo():
#     pass

# def tofu_demo():
#     pass

# def whiteboard_demo():
#     eraser_pos = [0.46666203, 0.15260639, 0.008]
#     start_pos = [0.45430818, 0.12086698, 0.0]
#     dx = 0.1
#     dy = -0.25
#     yaw = 0.0
    
#     # pick up eraser
#     skills._pick_from_top(eraser_pos)

#     # erase 
#     params = np.concatenate([start_pos, [dx, dy, yaw]])
#     # skills._push_xy(params=params)
#     skills._wipe_xy(params=params)

# def table_setting_demo():
#     bowl_l_pick_pos = [0.35273809, -0.09305739, 0.03344466]
#     bowl_s_pick_pos = [0.34883716, -0.09842549, 0.01824088]
#     spoon_pick_pos = [0.38930392, -0.10407848, 0.0]
#     cup_pick_pos = [0.55421162, -0.20033619, 0.05052035]
#     bowl_l_place_pos = [0.49800597, 0.08944026, 0.05197067]
#     bowl_s_place_pos = [0.37402136, 0.19934355, 0.02147385]
#     spoon_place_pos = [0.43709598, 0.05446209, 0.10071782]
#     cup_place_pos = [0.3500154, -0.10689394, 0.03797645]

#     # pick place bowl L
#     skills._pick_from_top(params=bowl_l_pick_pos)
#     skills._place_from_top(params=bowl_l_place_pos)

#     # pick_place bowl S
#     skills._pick_from_top(params=bowl_s_pick_pos)
#     skills._place_from_top(params=bowl_s_place_pos)

#     # pick place spoon
#     skills._pick_from_top(params=spoon_pick_pos)
#     skills._place_from_top(params=spoon_place_pos)

#     # pick place cup
#     skills._pick_from_top(params=cup_pick_pos)
#     skills._place_from_top(params=cup_place_pos)

# def tofu():
#     knife_pos = [0.53986897, -0.16281398, 0.11506687]
#     cut_pos = [0.47002734, 0.04665355, 0.25096985]
#     cut_depth = 0.2

#     # pick knife
#     skills._pick_from_top(params=knife_pos)

#     # cut 0 deg
#     skills._push_z(params=np.concatenate([cut_pos, [-cut_depth, 0.0]]))

#     # cut 90 deg
#     skills._push_z(params=np.concatenate([cut_pos, [-cut_depth, 90.0]]))

def main(args):

    task_name = args.task
    assert task_name is not None, "task not specified"
    assert task_name in task2env.keys(), f"task must be one of {task2env.keys()}"

    run_demo(task_name)
  
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    args = parser.parse_args()

    main(args)