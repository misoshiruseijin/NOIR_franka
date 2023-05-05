from utils.detection_utils import DetectionUtils
from deoxys.camera_redis_interface import CameraRedisSubInterface
from utils.camera_utils import get_camera_image, get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, pose_inv
from environments.solo_envs.realrobot_env_general import RealRobotEnvGeneral

import argparse
import os
import cv2
import json
import numpy as np

import time
import pdb

detection_utils = DetectionUtils()


def collect_data(env_name, save_dir):

    """
    Collect (image, skill, parameter) data for a task

    Args:

    Returns:
        data (dict) : 
            {
                datapoint # :
                    2d coord in cam0 : [],
                    2d coord in cam1 : [],
                    img0 : np.array,
                    img1 : np.array,
                    img0_boxes : np.array,
                    img1_boxes : np.array,
                    3d coord : [],
                    params : [],
            }
    """
    env = RealRobotEnvGeneral(
        env_name=env_name,
        controller_type="OSC_POSE"
    )
    obs = env.reset()
    obj_names = env.texts
    obj2skill = env.obj2skills

    # save_dir = f"il_data/{env_name}"
    os.makedirs(save_dir)

    full_data = {}
    
    step = 0
    while True:
        # get camera images
        cam0_raw = get_camera_image(camera_interface=env.camera_interfaces[0])
        cam1_raw = get_camera_image(camera_interface=env.camera_interfaces[1])
        cam0_box = cv2.imread("camera0.png")
        cam1_box = cv2.imread("camera1.png")
        
        obj_name = ""
        skill_name = ""

        # done with this iteration?
        done = input("task done? (y/n)")
        if done == "y":
            break

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

        # save data
        cv2.imwrite(f"{save_dir}/cam0_raw_{step}.png", cam0_raw)
        cv2.imwrite(f"{save_dir}/cam1_raw_{step}.png", cam1_raw)
        cv2.imwrite(f"{save_dir}/cam0_box_{step}.png", cam0_box)
        cv2.imwrite(f"{save_dir}/cam1_box_{step}.png", cam1_box)

        data = {
            "object" : obj_name,
            "skill" : skill_name,
            "params" : params,
            "obs" : obs,
        }
        full_data[step] = data

        # execute the skill
        print(f"Executing skill {skill_name}({params})")
        skill_idx = env.skill.skills[skill_name]["default_idx"]
        skill_selection_vec = np.zeros(env.num_skills)
        skill_selection_vec[skill_idx] = 1
        action = np.concatenate([skill_selection_vec, params])
        obs = env.step(action)

        step += 1

    # save data to json
    with open(f"{save_dir}/data.json", "x") as outfile:
        json.dump(full_data, outfile)


def main(args):

    ### test to make sure all data is readable from json ###    
    # with open("affordance_data/single_obj_fixed_pos_fixed_orn/data.json") as json_file:
    #     data = json.load(json_file)
    #     pdb.set_trace()

    for i in range(args.start_idx, args.start_idx + args.n_iter):
        collect_data(
            env_name=args.env,
            save_dir=f"il_data/{args.env}/traj{i}",
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--env", type=str)
    parser.add_argument("--start_idx", type=int, default=0) # save file index
    args = parser.parse_args()
    main(args)