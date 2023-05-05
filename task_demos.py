
import numpy as np
import sys
import time
import argparse
import json

sys.path.append("..")
sys.path.insert(1, "/home/eeg/deoxys_control/deoxys")
from primitive_skills_noir import PrimitiveSkill
from environments.solo_envs.realrobot_env_general import RealRobotEnvGeneral


from deoxys.utils.config_utils import (get_default_controller_config, verify_controller_config)
from deoxys.franka_interface import FrankaInterface
from deoxys.experimental.motion_utils import reset_joints_to

from utils.camera_utils import project_points_from_base_to_camera, get_camera_image
from deoxys.camera_redis_interface import CameraRedisSubInterface

from PIL import Image, ImageDraw
import cv2
import pdb
from getkey import getkey
import json

def terminal_shared_autonomy(env_name):
    with open('config/task_obj_skills.json') as json_file:
        task_dict = json.load(json_file)
        assert env_name in task_dict.keys(), f"Unrecognized environment name. Choose from {task_dict.keys()}"
        obj2skills = task_dict[env_name]
        obj_names = list(obj2skills.keys())
    
    env = RealRobotEnvGeneral(
        env_name=env_name,
        controller_type="OSC_POSE",
    )
    obs = env.reset()
    
    # ask for human input
    while True:
        obj_name = ""
        skill_name = ""

        # object selection
        while obj_name not in obj_names:
            obj_name = input(f"select object from {obj_names}:\n")

        # skill selection
        skill_options = obj2skills[obj_name]
        while skill_name not in skill_options:
            skill_name = input(f"choose skill from {skill_options}:\n")

        # param input
        params = []
        param_dim = env.skill.skills[skill_name]["num_params"]
        while not len(params) == param_dim:
            if "none" not in obj_name:
                # print(f"Position of {obj_name} is {obs[obj_name]}")
                world_coords = obs["world_coords"]
                print(f"Position of {obj_name} is {world_coords[obj_name]}")

            params = input(f"input skill parameters ({param_dim} floats separated by spaces):\n")
            params = [float(x) for x in params.split()]

        # execute the skill
        print(f"Executing skill {skill_name}({params})")
        skill_idx = env.skill.skills[skill_name]["default_idx"]
        skill_selection_vec = np.zeros(env.num_skills)
        skill_selection_vec[skill_idx] = 1
        action = np.concatenate([skill_selection_vec, params])
        obs, _, _, _ = env.step(action)
        print("object positions\n", obs["world_coords"])
        
def main(args):
    terminal_shared_autonomy(env_name=args.env)   
  
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)
    args = parser.parse_args()

    main(args)



# import numpy as np
# import sys
# import time
# import argparse
# import json

# sys.path.append("..")
# sys.path.insert(1, "/home/eeg/deoxys_control/deoxys")
# from primitive_skills_noir import PrimitiveSkill
# from environments.tablesetting_env import TablesettingEnv
# from environments.tofu_env import TofuEnv
# from environments.whiteboard_env import WhiteboardEnv
# from environments.realrobot_env_noir import RealRobotEnv


# from deoxys.utils.config_utils import (get_default_controller_config, verify_controller_config)
# from deoxys.franka_interface import FrankaInterface
# from deoxys.experimental.motion_utils import reset_joints_to

# from camera_utils import project_points_from_base_to_camera, get_camera_image
# from deoxys.camera_redis_interface import CameraRedisSubInterface

# from PIL import Image, ImageDraw
# import cv2
# import pdb
# from getkey import getkey

# import pdb

# reset_joint_positions = [
#     0.09162008114028396,
#     -0.19826458111314524,
#     -0.01990020486871322,
#     -2.4732269941140346,
#     -0.01307073642274261,
#     2.30396583422025,
#     0.8480939705504309,
# ]
    
# obj2skill = {
#     "red bowl" : ["pick_from_top", "pick_from_side", "push_xy"],
#     "light blue bowl" : ["pick_from_top", "pick_from_side", "push_xy"], 
#     "shiny silver cup" : ["pick_from_top", "pick_from_side", "push_xy"],
#     "red and blue spoon" : ["pick_from_top", "pick_from_side", "push_xy"],
#     "red spatula" : ["pick_from_top", "pick_from_side", "push_xy"],
#     "dark blue eraser" : ["pick_from_top", "pick_from_side", "push_xy"],
#     "none_tablesetting" : ["place_from_top", "place_from_side", "reset_joints"],
#     "none_tofu" : ["push_z", "place_from_top", "reset_joints"],
#     "none_whiteboard" : ["wipe_xy", "place_from_top", "reset_joints"]
# }

# task2env = {
#     "tablesetting" : TablesettingEnv,
#     "tofu" : TofuEnv,
#     "whiteboard" : WhiteboardEnv,
# }

# def terminal_shared_autonomy(env:RealRobotEnv, obj_names):
    
#     obs = env.reset()
    
#     # ask for human input
#     while True:
#         obj_name = ""
#         skill_name = ""

#         # object selection
#         while obj_name not in obj_names:
#             obj_name = input(f"select object from {obj_names}:\n")

#         # skill selection
#         skill_options = obj2skill[obj_name]
#         while skill_name not in skill_options:
#             skill_name = input(f"choose skill from {skill_options}:\n")

#         # param input
#         params = []
#         param_dim = env.skill.skills[skill_name]["num_params"]
#         while not len(params) == param_dim:
#             if "none" not in obj_name:
#                 print(f"Position of {obj_name} is {obs[obj_name]}")
#             params = input(f"input skill parameters ({param_dim} floats separated by spaces):\n")
#             params = [float(x) for x in params.split()]

#         # execute the skill
#         print(f"Executing skill {skill_name}({params})")
#         skill_idx = env.skill.skills[skill_name]["default_idx"]
#         skill_selection_vec = np.zeros(env.num_skills)
#         skill_selection_vec[skill_idx] = 1
#         action = np.concatenate([skill_selection_vec, params])
#         env.step(action)

# def run_demo(task_name):
#     env_class = task2env[task_name]
#     env = env_class(
#         normalized_params=False,
#     )

#     # get object names
#     obj_names = env.texts.copy()
#     obj_names.append(f"none_{task_name}")
#     terminal_shared_autonomy(env, obj_names)

# def main(args):

#     task_name = args.task
#     assert task_name is not None, "task not specified"
#     assert task_name in task2env.keys(), f"task must be one of {task2env.keys()}"

#     run_demo(task_name)
  
# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--task", type=str)
#     args = parser.parse_args()

#     main(args)