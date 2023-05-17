from utils.detection_utils import DetectionUtils
from deoxys.camera_redis_interface import CameraRedisSubInterface
from utils.camera_utils import get_camera_image, get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, pose_inv, project_points_from_base_to_camera
import argparse
import os
import cv2
import json
import numpy as np
from PIL import Image, ImageDraw

import time
import pdb
camera_interfaces = {
        0 : CameraRedisSubInterface(camera_id=0),
        1 : CameraRedisSubInterface(camera_id=1),
        2 : CameraRedisSubInterface(camera_id=2)
    }

detection_utils = DetectionUtils()


# def collect_affordance_data(texts, save_dir_name="test", n_iter=1):

#     """
#     Collect (image, skill parameter) data for specified object

#     Args:
#         texts (list of str) : text for object of interest for OWL-ViT detector
#         save_file_name (str) : name of file to save data in
#         n_iter (int) : number of datapoints to detect

#     Returns:
#         data (dict) : 
#             {
#                 datapoint # :
#                     2d coord in cam0 : [],
#                     2d coord in cam1 : [],
#                     img0 : np.array,
#                     img1 : np.array,
#                     img0_boxes : np.array,
#                     img1_boxes : np.array,
#                     3d coord : [],
#                     params : [],
#             }
#     """

#     save_dir = f"affordance_data/{save_dir_name}"
#     os.makedirs(save_dir, exist_ok=True)

#     full_data = {}

#     for i in range(n_iter):

#         print(f"--------- Collecting Sample {i} -----------")
        
#         # wait for start signal input
#         input("Set the physical environment, then press enter")

#         # get image arrays
#         image0 = get_camera_image(camera_interface=camera_interfaces[0])
#         image1 = get_camera_image(camera_interface=camera_interfaces[1])

#         # get 2d image coordinate in both images
#         coords0 = detection_utils.get_obj_pixel_coord(
#             camera_interface=camera_interfaces[0],
#             camera_id=0,
#             texts=texts,
#             thresholds=[0.001]*4,
#             save_img=True,
#             n_instances=1,
#         )
#         coords1 = detection_utils.get_obj_pixel_coord(
#             camera_interface=camera_interfaces[1],
#             camera_id=1,
#             texts=texts,
#             thresholds=[0.001]*4,
#             save_img=True,
#             n_instances=1,
#         )

#         print("2d coords in camera 0", coords0)
#         print("2d coords in camera 1", coords1)

#         world_coords = detection_utils.get_object_world_coords(
#             camera_interface0=camera_interfaces[0],
#             camera_interface1=camera_interfaces[1],
#             texts=texts,
#             thresholds=[0.001],
#             wait=True,
#         )
#         for text in world_coords.keys():
#             print(f"Detected {text} at {world_coords[text]}")

#         # ask for parameter input

#         # record coordinates and parameter
#         obj_name = texts[0]
#         image0_box = cv2.imread("camera0.png")
#         image1_box = cv2.imread("camera1.png")
#         data = {
#             i : {
#                 # "cam0_raw" : image0.tolist(),
#                 # "cam1_raw" : image1.tolist(),
#                 # "cam0_box" : image0_box.tolist(),
#                 # "cam1_box" : image1_box.tolist(),
#                 "cam0_2d_coord" : coords0[obj_name]["centers"][0],
#                 "cam1_2d_coord" : coords1[obj_name]["centers"][0],
#                 "3d_coord" : world_coords[obj_name].tolist(),
#                 "params" : world_coords[obj_name].tolist(),
#             }
#         } 
#         full_data.update(data)

#         # save images
#         cv2.imwrite(f"{save_dir}/cam0_raw_{i}.png", image0)
#         cv2.imwrite(f"{save_dir}/cam1_raw_{i}.png", image1)
#         cv2.imwrite(f"{save_dir}/cam0_box_{i}.png", image0_box)
#         cv2.imwrite(f"{save_dir}/cam1_box_{i}.png", image1_box)

    
#     with open(f"affordance_data/{save_dir_name}/data.json", "x") as outfile:
#         json.dump(full_data, outfile)

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
                    2d coord in cam2 : [],
                    3d coord : [],
                    params : [],
            }
    """
    trim_low = [90, 130]
    trim_high = [450, 370]
    
    save_dir = f"affordance_data/{save_dir_name}"
    os.makedirs(save_dir, exist_ok=True)

    full_data = {}

    # for i in range(n_iter):
    i = 0

    while i < n_iter:

        print(f"--------- Collecting Sample {i} -----------")
        
        # wait for start signal input
        input("Set the physical environment, then press enter")

        # get 2d coordinates in cropped camera2 image
        image2 = get_camera_image(camera_interface=camera_interfaces[2])
        image2 = image2[trim_low[1]:trim_high[1], trim_low[0]:trim_high[0]]

        coords2 = detection_utils.get_obj_pixel_coord_from_image(
            img_array=image2,
            texts=texts,
            camera_id=2,
            thresholds=[0.001]*len(texts),
            save_img=True,
            n_instances=1,
        )

        print("2d coords in camera2", coords2)

        # record coordinates and parameter
        obj_name = texts[0]
        xy_world_coord = detection_utils.get_world_xy_from_topdown_view(
            pix_coords=coords2[obj_name]["centers"][0],
            camera_interface=camera_interfaces[2],
            trim=True,
        )
        data = {
            i : {
                # "cam2_2d_coord" : im2_coords.tolist(),
                "cam2_2d_coord" : coords2[obj_name]["centers"][0],
                "xy_world_coord" : list(xy_world_coord), # xy world coordinates estimated from topdown view
                # "3d_coord" : world_coords[obj_name].tolist(), # 3d world coordinates estimated from 2 side views
                # "params" : world_coords[obj_name].tolist(),
            }
        }

        # choose if data should be saved
        while True:
            save = input("Save this sample? (y/n)")
            if save == "y":
                print(f"Saved sample {i}")
                full_data.update(data)

                # save images
                image2_box = cv2.imread("camera2.png")
                cv2.imwrite(f"{save_dir}/cam2_raw_{i}.png", image2)
                cv2.imwrite(f"{save_dir}/cam2_box_{i}.png", image2_box)
                i += 1
                break
            elif save == "n":
                break

        print("full data size : ", len(list(full_data.keys())))

    with open(f"affordance_data/{save_dir_name}/data.json", "x") as outfile:
        json.dump(full_data, outfile)



# import numpy as np
# import sys
# import time
# import argparse
# import json

# import tkinter as tk
# from PIL import Image, ImageTk
# import numpy as np
# import cv2
# from utils.detection_utils_eeg import DetectionUtils

# from PIL import Image, ImageDraw
# import cv2
# import json


# class FullPipelineDemo:
#     def __init__(
#         self,
#         env_name,
#     ):
#         self.pix_pos = None
#         self.chose_xy = False
#         self.chose_z = False
#         self.canvas = None # TODO - not needed if not using tkinter UI
#         self.window = None # TODO - not needed if not using tkinter UI
#         self.detection_utils = DetectionUtils()

#         with open('config/task_obj_skills.json') as json_file:
#             task_dict = json.load(json_file)
#             assert env_name in task_dict.keys(), f"Unrecognized environment name. Choose from {task_dict.keys()}"
#             self.obj2skills = task_dict[env_name]
#             self.obj_names = list(self.obj2skills.keys())

#         with open('config/skill_config.json') as json_file:
#             self.skill_dict = json.load(json_file)
#             self.num_skills = len(self.skill_dict.keys())

#         self.skill_names = ["pick_from_top", "push_x"]
#         self.skill_idx = 0

#     def collect_affordance_data(texts, save_dir_name="test", n_iter=1):

#         """
#         Collect (image, skill parameter) data for specified object

#         Args:
#             texts (list of str) : text for object of interest for OWL-ViT detector
#             save_file_name (str) : name of file to save data in
#             n_iter (int) : number of datapoints to detect

#         Returns:
#             data (dict) : 
#                 {
#                     datapoint # :
#                         2d coord in cam2 : [],
#                         3d coord : [],
#                         params : [],
#                 }
#         """
#         trim_low = [90, 130]
#         trim_high = [450, 370]
        
#         save_dir = f"affordance_data/{save_dir_name}"
#         os.makedirs(save_dir, exist_ok=True)

#         full_data = {}

#         # for i in range(n_iter):
#         i = 0

#         while i < n_iter:

#             print(f"--------- Collecting Sample {i} -----------")
            
#             # wait for start signal input
#             input("Set the physical environment, then press enter")

#             # get 2d coordinates in cropped camera2 image
#             image2 = get_camera_image(camera_interface=camera_interfaces[2])
#             image2 = image2[trim_low[1]:trim_high[1], trim_low[0]:trim_high[0]]

#             coords2 = detection_utils.get_obj_pixel_coord(
#                 img_array=image2,
#                 texts=texts,
#                 camera_id=2,
#                 thresholds=[0.001]*len(texts),
#                 save_img=True,
#                 n_instances=1,
#             )

#             print("2d coords in camera2", coords2)

#             # record coordinates and parameter
#             obj_name = texts[0]
#             xy_world_coord = detection_utils.get_world_xy_from_topdown_view(
#                 pix_coords=coords2[obj_name]["centers"][0],
#                 camera_interface=camera_interfaces[2],
#                 trim=True,
#             )
#             data = {
#                 i : {
#                     # "cam2_2d_coord" : im2_coords.tolist(),
#                     "cam2_2d_coord" : coords2[obj_name]["centers"][0],
#                     "xy_world_coord" : list(xy_world_coord), # xy world coordinates estimated from topdown view
#                     # "3d_coord" : world_coords[obj_name].tolist(), # 3d world coordinates estimated from 2 side views
#                     # "params" : world_coords[obj_name].tolist(),
#                 }
#             }

#             # choose if data should be saved
#             while True:
#                 save = input("Save this sample? (y/n)")
#                 if save == "y":
#                     print(f"Saved sample {i}")
#                     full_data.update(data)

#                     # save images
#                     image2_box = cv2.imread("camera2.png")
#                     cv2.imwrite(f"{save_dir}/cam2_raw_{i}.png", image2)
#                     cv2.imwrite(f"{save_dir}/cam2_box_{i}.png", image2_box)
#                     i += 1
#                     break
#                 elif save == "n":
#                     break

#             print("full data size : ", len(list(full_data.keys())))

#         with open(f"affordance_data/{save_dir_name}/data.json", "x") as outfile:
#             json.dump(full_data, outfile)


#     def get_action(self, topdown_img_path, sideview_img_path, side_camera_id):
#         """
#         Gets action to be sent to robot side
#         """
#         # obj_name = self.get_obj_selection()
#         # skill_name = self.get_skill_selection(obj_name)
#         skill_name = self.skill_names[self.skill_idx]
#         skill_idx = not skill_idx
#         params = self.get_param_selection(topdown_img_path, sideview_img_path, side_camera_id, skill_name)

#         # get one-hot skill selection vector
#         skill_idx = self.skill_dict[skill_name]["default_idx"]
#         skill_selection_vec = np.zeros(self.num_skills)
#         skill_selection_vec[skill_idx] = 1

#         # concatenate params
#         action = np.concatenate([skill_selection_vec, params])
#         return action

#     def get_obj_selection(self):
#         """
#         Let human choose object
#         Returns:
#             obj_name (str) : name of selected object
#         """
#         # TODO - run object detector, SSVEP stimulus generation, SSVEP decoding, etc.

#         obj_name = input(f"choose obj from {self.obj_names}") # TODO - replace with SSVEP results
#         assert (obj_name in self.obj_names), f"object must be one of {self.obj_names}, but got {obj_name}\n"
#         return obj_name

#     def get_skill_selection(self, obj_name):
#         """
#         Present human with skill options, and returns name of chosen skill

#         Args:
#             obj_name (str) : name of object selected by human

#         Returns:
#             skill_name (str) : name of selected skill
#         """
#         skill_options = self.obj2skills[obj_name] # list of skills to preesnt to the human, given 
        
#         skill_name = input(f"choose skill from {skill_options}") # TODO - replace this from motor imagery results
#         assert skill_name in skill_options, f"skill must be one of {skill_options}, but got {skill_name}\n"
#         return skill_name

#     def get_param_selection(self, topdown_img_path, skill_name):
#         """
#         Given images and name of selected skill, get parameters
#         TODO - for future: add cases for skills with more than 3 inputs

#         Args:
#             topdown_img_path (str) : path to topdown image used for xy param selection
#             sideview_mig_path (str) : path to sideview image used for z param selection
#             side_camera_id (int) : 0 or 1, corresponding to the sideview image used for z param selection
#             skill_name (str) : name of selected skill
#             obj_name (str) : name of selected object - TODO remove this after z decoding works

#         Returns:
#             params (list of floats) : selected parameters for the given skill
#         """
#         # brings up a UI to click positions on images to select parameters for "pick" skill

#         # reads images
#         topdown_img = cv2.imread(topdown_img_path)
#         topdown_pil_image = Image.fromarray(topdown_img[:,:,::-1])

#         ############### choose x and y ################
#         # Setup Tkinter window and canvas with topdown view (cam 2 for param selection) - TODO Replace this block with cursor control
#         self.window = tk.Tk()
#         self.canvas = tk.Canvas(self.window, width=topdown_pil_image.width, height=topdown_pil_image.height)
#         self.canvas.pack()
#         photo = ImageTk.PhotoImage(topdown_pil_image)
#         self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
#         self.canvas.bind("<Button 1>", self.get_pixel_position) # bind mouseclick event to canvas
#         self.window.mainloop() # start Tkinter event loop

#         # get xy world position from pixel position
#         world_xy = self.detection_utils.get_world_xy_from_topdown_view(pix_coords=self.pix_pos, img_array=topdown_img)
        
#         ################ hardcode z according to object ##################
#         if "pick" in skill_name:
#             world_z = 0.14
#         else:
#             world_z = 0.15

#         # get full params
#         params = [world_xy[0], world_xy[1], world_z]
#         print("selected world coordinates", params)
        
#         return params
#         # action = np.concatenate([skill_selection_vec, params])
#         # obs, reward, done, info = self.env.step(action)

#     def get_pixel_position(self, event):
#         # Get the position of the mouse click in the image
#         x, y = event.x, event.y
#         # Print the pixel position
#         print("Pixel position:", x, y)
#         self.pix_pos = x,y
#         self.window.destroy()
        
# def main(args):
#     demo = FullPipelineDemo(env_name=args.env_name)
    
#     detection_utils = DetectionUtils()
#     action = demo.get_action(
#         topdown_img_path="param_selection_img2.png",
#         sideview_img_path="param_selection_img0.png",
#         side_camera_id=0,
#     )
#     print("Action to send to robot is", action)

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--env_name", type=str, default="TableSetting")
#     args = parser.parse_args()
#     main(args)


def main(args):

    ### test to make sure all data is readable from json ###    
    # with open("affordance_data/single_obj_fixed_pos_fixed_orn/data.json") as json_file:
    #     data = json.load(json_file)
    #     pdb.set_trace()

    # collect_affordance_data(texts=["pink grip handle on a cooking knife"], save_dir_name=args.save_dir_name, n_iter=args.n_iter)
    collect_affordance_data(texts=["mug handle"], save_dir_name=args.save_dir_name, n_iter=args.n_iter)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--text", type=str)
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--save_dir_name", type=str, default="test_data")
    args = parser.parse_args()
    main(args)