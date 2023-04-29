from utils.detection_utils import DetectionUtils
from deoxys.camera_redis_interface import CameraRedisSubInterface
from utils.camera_utils import get_camera_image, get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, pose_inv
import argparse
import os
import cv2
import json
import numpy as np

import time
import pdb
camera_interfaces = {
        0 : CameraRedisSubInterface(camera_id=0),
        1 : CameraRedisSubInterface(camera_id=1),
    }

detection_utils = DetectionUtils()


def collect_affordance_data(texts, save_dir_name="test", n_iter=1):

    """
    Collect (image, skill parameter) data for specified object

    Args:
        texts (list of str) : text for object of interest for OWL-ViT detector
        save_file_name (str) : name of file to save data in
        n_iter (int) : number of datapoints to detect

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

    save_dir = f"affordance_data/{save_dir_name}"
    os.makedirs(save_dir, exist_ok=True)

    full_data = {}

    for i in range(n_iter):
        # wait for start signal input
        # input("Set the physical environment, then press enter")

        # get image arrays
        image0 = get_camera_image(camera_interface=camera_interfaces[0])
        image1 = get_camera_image(camera_interface=camera_interfaces[1])

        # get 2d image coordinate in both images
        coords0 = detection_utils.get_obj_pixel_coord(
            camera_interface=camera_interfaces[0],
            camera_id=0,
            texts=texts,
            thresholds=[0.001]*4,
            save_img=True,
            n_instances=1,
        )
        coords1 = detection_utils.get_obj_pixel_coord(
            camera_interface=camera_interfaces[1],
            camera_id=1,
            texts=texts,
            thresholds=[0.001]*4,
            save_img=True,
            n_instances=1,
        )

        print("2d coords in camera 0", coords0)
        print("2d coords in camera 1", coords1)

        world_coords = detection_utils.get_object_world_coords(
            camera_interface0=camera_interfaces[0],
            camera_interface1=camera_interfaces[1],
            texts=texts,
            thresholds=[0.001],
            wait=True,
        )
        for text in world_coords.keys():
            print(f"Detected {text} at {world_coords[text]}")

        # ask for parameter input

        # record coordinates and parameter
        obj_name = texts[0]
        image0_box = cv2.imread("camera0.png")
        image1_box = cv2.imread("camera1.png")
        data = {
            i : {
                # "cam0_raw" : image0.tolist(),
                # "cam1_raw" : image1.tolist(),
                # "cam0_box" : image0_box.tolist(),
                # "cam1_box" : image1_box.tolist(),
                "cam0_2d_coord" : coords0[obj_name]["centers"][0],
                "cam1_2d_coord" : coords1[obj_name]["centers"][0],
                "3d_coord" : world_coords[obj_name].tolist(),
                "params" : world_coords[obj_name].tolist(),
            }
        } 
        full_data.update(data)

        # save images
        cv2.imwrite(f"{save_dir}/cam0_raw_{i}.png", image0)
        cv2.imwrite(f"{save_dir}/cam1_raw_{i}.png", image1)
        cv2.imwrite(f"{save_dir}/cam0_box_{i}.png", image0_box)
        cv2.imwrite(f"{save_dir}/cam1_box_{i}.png", image1_box)

    
    with open(f"affordance_data/{save_dir_name}/data.json", "x") as outfile:
        pdb.set_trace()
        json.dump(full_data, outfile)

def main(args):

    ### test to make sure all data is readable from json ###    
    # with open("affordance_data/fixed_pos_and_ori_no_distractions.json") as json_file:
    #     data = json.load(json_file)
    #     pdb.set_trace()

    collect_affordance_data(texts=["pink grip handle on a cooking knife"], save_dir_name=args.save_dir_name, n_iter=args.n_iter)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--text", type=str)
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--save_dir_name", type=str, default="affordance_data")
    args = parser.parse_args()
    main(args)