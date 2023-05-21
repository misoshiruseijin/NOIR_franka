from utils.detection_utils_eeg import DetectionUtils, ObjectDetector
from deoxys.camera_redis_interface import CameraRedisSubInterface
from utils.camera_utils import get_camera_image, get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, pose_inv, project_points_from_base_to_camera
import argparse
import os
import cv2
import json
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk


import time
import pdb
camera_interfaces = {
        0 : CameraRedisSubInterface(camera_id=0),
        1 : CameraRedisSubInterface(camera_id=1),
        2 : CameraRedisSubInterface(camera_id=2)
    }

class AffordanceDataCollectorUI:
    def __init__(
        self,
    ):
        self.pix_pos = None
        self.chose_xy = False
        self.canvas = None # TODO - not needed if not using tkinter UI
        self.window = None # TODO - not needed if not using tkinter UI
        self.detection_utils = DetectionUtils()

    def get_param_selection(self, topdown_img):
        """
        Given images and name of selected skill, get parameters
        TODO - for future: add cases for skills with more than 3 inputs

        Args:
            topdown_img_path (str) : path to topdown image used for xy param selection

        Returns:
            params (list of floats) : selected parameters for the given skill
        """
        # brings up a UI to click positions on images to select parameters for "pick" skill

        # reads images
        topdown_pil_image = Image.fromarray(topdown_img[:,:,::-1])

        # which params are needed for this skill?
        params = []

        ############### choose x and y ################
        # Setup Tkinter window and canvas with topdown view (cam 2 for param selection) - TODO Replace this block with cursor control
        self.window = tk.Tk()
        self.canvas = tk.Canvas(self.window, width=topdown_pil_image.width, height=topdown_pil_image.height)
        self.canvas.pack()
        photo = ImageTk.PhotoImage(topdown_pil_image)
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.canvas.bind("<Button 1>", self.get_pixel_position) # bind mouseclick event to canvas
        self.window.mainloop() # start Tkinter event loop

        # get xy world position from pixel position
        world_xy = self.detection_utils.get_world_xy_from_topdown_view(pix_coords=self.pix_pos, img_array=topdown_img)
        params.append(world_xy[0])
        params.append(world_xy[1])

        return params

    def get_pixel_position(self, event):
        # Get the position of the mouse click in the image
        x, y = event.x, event.y
        # Print the pixel position
        print("Pixel position:", x, y)
        self.pix_pos = x,y
        self.window.destroy()

    def collect_affordance_data_w_clicks(self, save_dir_name="test", n_iter=1):

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
        os.makedirs(save_dir, exist_ok=False)

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
            cv2.imwrite("camera2.png", image2)

            xy_world_coord = self.get_param_selection(topdown_img=image2)
            coords2 = self.pix_pos
            print("2d coords in camera2", coords2)
            data = {
                i : {
                    # "cam2_2d_coord" : im2_coords.tolist(),
                    "cam2_2d_coord" : [coords2[0], coords2[1]],
                    "xy_world_coord" : list(xy_world_coord), # xy world coordinates estimated from topdown view
                    # "3d_coord" : world_coords[obj_name].tolist(), # 3d world coordinates estimated from 2 side views
                    # "params" : world_coords[obj_name].tolist(),
                }
            }
            print("New data", data)

            # choose if data should be saved
            while True:
                save = input("Save this sample? (y/n)")
                if save == "y":
                    print(f"Saved sample {i}")
                    full_data.update(data)

                    # save images
                    # image2_box = cv2.imread("camera2.png")
                    cv2.imwrite(f"{save_dir}/cam2_raw_{i}.png", image2)
                    # cv2.imwrite(f"{save_dir}/cam2_box_{i}.png", image2_box)
                    i += 1
                    break
                elif save == "n":
                    break

            print("full data size : ", len(list(full_data.keys())))

        with open(f"affordance_data/{save_dir_name}/data.json", "x") as outfile:
            json.dump(full_data, outfile)

    def collect_affordance_data_w_detector(self, texts, save_dir_name="test", n_iter=1):

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
        os.makedirs(save_dir, exist_ok=False)


        detector = ObjectDetector()
        full_data = {}
        i = 0

        while i < n_iter:

            print(f"--------- Collecting Sample {i} -----------")
            
            # wait for start signal input
            input("Set the physical environment, then press enter")

            # get 2d coordinates in cropped camera2 image
            image2 = get_camera_image(camera_interface=camera_interfaces[2])
            image2 = image2[trim_low[1]:trim_high[1], trim_low[0]:trim_high[0]]
            cv2.imwrite("camera2.png", image2)

            coords2 = detector.get_obj_pixel_coord(
                img_array=image2,
                texts=texts,
                # camera_id=2,
                thresholds=[0.001]*len(texts),
                save_img=True,
                n_instances=1,
            )

            # record coordinates and parameter
            obj_name = texts[0]
            xy_world_coord = self.detection_utils.get_world_xy_from_topdown_view(
                pix_coords=coords2[obj_name]["centers"][0],
                img_array=image2,
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
            print("New data ", data)

            # choose if data should be saved
            while True:
                save = input("Save this sample? (y/n)")
                if save == "y":
                    print(f"Saved sample {i}")
                    print("Added new data", data)
                    full_data.update(data)

                    # save images
                    # image2_box = cv2.imread("camera2.png")
                    cv2.imwrite(f"{save_dir}/cam2_raw_{i}.png", image2)
                    # cv2.imwrite(f"{save_dir}/cam2_box_{i}.png", image2_box)
                    i += 1
                    break
                elif save == "n":
                    break

            print("full data size : ", len(list(full_data.keys())))

        with open(f"affordance_data/{save_dir_name}/data.json", "x") as outfile:
            json.dump(full_data, outfile)


def main(args):

    data_collector = AffordanceDataCollectorUI()
    # data_collector.collect_affordance_data_w_detector(texts=["brown handle"], save_dir_name=args.save_dir_name, n_iter=20)
    data_collector.collect_affordance_data_w_clicks(save_dir_name=args.save_dir_name, n_iter=args.n_iter)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--text", type=str)
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--save_dir_name", type=str, default="test_data")
    args = parser.parse_args()
    main(args)