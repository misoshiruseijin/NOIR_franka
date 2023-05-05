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

with open('config/obj2skillparams.json') as json_file:
    PARAM_CONFIG = json.load(json_file)

SEQUENCE = {
    "TableSetting" : [
        ("shiny silver cup", "pick_from_top"),
        ("shiny silver cup", "place_from_top"),
        ("light blue bowl", "pick_from_top"),
        ("light blue bowl", "place_from_top"),
        ("green handle", "pick_from_top"),
        ("green handle", "place_from_top"),
    ]
}

def get_skill_param(env_name, obj_name, skill_name, world_coords, perturb=0.0):
    param_config = PARAM_CONFIG[env_name][obj_name]
    if "pick" in skill_name:
        world_coord = world_coords[obj_name]
        pick_params = param_config[skill_name]
        params = [world_coord[0] + pick_params[0], world_coord[1] + pick_params[1], pick_params[2]]
        return params
    elif "place" in skill_name:
        place_params = param_config[skill_name]
        ref = param_config["place_ref"]
        if ref == "world":
            params = place_params
        else:
            ref_coord = world_coords[ref]
            params = [ref_coord[0] + place_params[0], ref_coord[1] + place_params[1], ref_coord[2] + place_params[2]]
        # add some noise to place position
        noise = np.random.uniform(-perturb, perturb, 3)
        params = [params[0] + noise[0], params[1] + noise[1], params[2]]
        return params

def collect_data_manual(env_name, save_dir):

    """
    Collect (image, skill, parameter) data for a task.
    Manual: terminal input for object, skill, parameter for each step
    """

    # save_dir = f"il_data/{env_name}"
    os.makedirs(save_dir)
    
    env = RealRobotEnvGeneral(
        env_name=env_name,
        controller_type="OSC_POSE"
    )
    obs = env.reset()
    obj2skill = env.obj2skills
    obj_names = obj2skill.keys()

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

        # get object world coordinates
        world_coords = obs["world_coords"]

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
                print(f"Position of {obj_name} is {world_coords[obj_name]}")
            params = input(f"input skill parameters ({param_dim} floats separated by spaces):\n")
            params = [float(x) for x in params.split()]

        # save data
        cv2.imwrite(f"{save_dir}/cam0_raw_{step}.png", cam0_raw)
        cv2.imwrite(f"{save_dir}/cam1_raw_{step}.png", cam1_raw)
        cv2.imwrite(f"{save_dir}/cam0_box_{step}.png", cam0_box)
        cv2.imwrite(f"{save_dir}/cam1_box_{step}.png", cam1_box)

        for key, value in world_coords.items():
            if isinstance(value, np.ndarray):
                world_coords[key] = value.tolist()
        data = {
            "object" : obj_name,
            "skill" : skill_name,
            "params" : params,
            "world_coords" : world_coords,
            "pixel_coords0" : obs["pixel_coords0"],
            "pixel_coords1" : obs["pixel_coords1"],
        }
        
        full_data[step] = data

        # execute the skill
        print(f"Executing skill {skill_name}({params})")
        skill_idx = env.skill.skills[skill_name]["default_idx"]
        skill_selection_vec = np.zeros(env.num_skills)
        skill_selection_vec[skill_idx] = 1
        action = np.concatenate([skill_selection_vec, params])
        obs, _, _, _ = env.step(action)
        print("obj pos\n", obs["world_coords"])

        step += 1

    # save data to json
    with open(f"{save_dir}/data.json", "x") as outfile:
        json.dump(full_data, outfile)

def collect_data_auto(env, env_name, save_dir, perturb=0.0):

    """
    Collect (image, skill, parameter) data for a task
    Auto : uses predefined sequence of actions and per-object parameters to automatically collect data
    """
    # save_dir = f"il_data/{env_name}"
    os.makedirs(save_dir)
    
    obs = env.reset()
    obj2skill = env.obj2skills

    full_data = {}
    
    step = 0

    sequence = SEQUENCE[env_name]

    for action in sequence:
        
        # obj, skill, and params      
        obj_name, skill_name = action
        world_coords = obs["world_coords"]
        params = get_skill_param(env_name, obj_name, skill_name, world_coords, perturb=perturb) 

        # get camera images
        cam0_raw = get_camera_image(camera_interface=env.camera_interfaces[0])
        cam1_raw = get_camera_image(camera_interface=env.camera_interfaces[1])
        cam0_box = cv2.imread("camera0.png")
        cam1_box = cv2.imread("camera1.png")

        # save data
        cv2.imwrite(f"{save_dir}/cam0_raw_{step}.png", cam0_raw)
        cv2.imwrite(f"{save_dir}/cam1_raw_{step}.png", cam1_raw)
        cv2.imwrite(f"{save_dir}/cam0_box_{step}.png", cam0_box)
        cv2.imwrite(f"{save_dir}/cam1_box_{step}.png", cam1_box)
        
        for key, value in world_coords.items():
            if isinstance(value, np.ndarray):
                world_coords[key] = value.tolist()
        data = {
            "object" : obj_name,
            "skill" : skill_name,
            "params" : params,
            "world_coords" : obs["world_coords"],
            "pixel_coords0" : obs["pixel_coords0"],
            "pixel_coords1" : obs["pixel_coords1"],
        }
        full_data[step] = data

        # execute the skill
        print(f"Executing skill {skill_name}({params})")
        skill_idx = env.skill.skills[skill_name]["default_idx"]
        skill_selection_vec = np.zeros(env.num_skills)
        skill_selection_vec[skill_idx] = 1
        action = np.concatenate([skill_selection_vec, params])
        obs, _, _, _ = env.step(action)
        print("obj pos\n", obs["world_coords"])

        step += 1

    # save final image and observations
    cam0_raw = get_camera_image(camera_interface=env.camera_interfaces[0])
    cam1_raw = get_camera_image(camera_interface=env.camera_interfaces[1])
    cam0_box = cv2.imread("camera0.png")
    cam1_box = cv2.imread("camera1.png")
    cv2.imwrite(f"{save_dir}/cam0_raw_{step}.png", cam0_raw)
    cv2.imwrite(f"{save_dir}/cam1_raw_{step}.png", cam1_raw)
    cv2.imwrite(f"{save_dir}/cam0_box_{step}.png", cam0_box)
    cv2.imwrite(f"{save_dir}/cam1_box_{step}.png", cam1_box)

    world_coords = obs["world_coords"]
    for key, value in world_coords.items():
        if isinstance(value, np.ndarray):
            world_coords[key] = value.tolist()
    data = {
        "object" : "none",
        "skill" : "none",
        "params" : [],
        "world_coords" : world_coords,
        "pixel_coords0" : obs["pixel_coords0"],
        "pixel_coords1" : obs["pixel_coords1"],
    }
    full_data[step] = data

    # save data to json
    with open(f"{save_dir}/data.json", "x") as outfile:
        json.dump(full_data, outfile)


def main(args):

    ### test to make sure all data is readable from json ###    
    # with open("affordance_data/single_obj_fixed_pos_fixed_orn/data.json") as json_file:
    #     data = json.load(json_file)
    #     pdb.set_trace()

    # for i in range(args.start_idx, args.start_idx + args.n_iter):
    #     collect_data_manual(
    #         env_name=args.env,
    #         save_dir=f"il_data/{args.env}/traj{i}",
    #     )

    # initialize environment
    env = RealRobotEnvGeneral(
        env_name=args.env,
        controller_type="OSC_POSE"
    )
    for i in range(args.start_idx, args.start_idx + args.n_iter):
        collect_data_auto(
            env=env,
            env_name=args.env,
            save_dir=f"il_data/{args.env}/traj{i}",
            perturb=args.pert
        )
        env.robot_interface.reset()
        input("Reset the environment for next trajectory and press Enter")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--env", type=str)
    parser.add_argument("--start_idx", type=int, default=0) # save file index
    parser.add_argument("--pert", type=float, default=0.0) # magnitude of place position noise (x,y)
    args = parser.parse_args()
    main(args)