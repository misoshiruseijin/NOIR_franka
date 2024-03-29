
import numpy as np
import sys
import time
import argparse
import json

import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import cv2
from utils.detection_utils_eeg import DetectionUtils

from PIL import Image, ImageDraw
import cv2
import json


class FullPipelineDemo:
    def __init__(
        self,
        env_name,
    ):
        self.pix_pos = None
        self.chose_xy = False
        self.chose_z = False
        self.canvas = None # TODO - not needed if not using tkinter UI
        self.window = None # TODO - not needed if not using tkinter UI
        self.detection_utils = DetectionUtils()

        with open('config/task_obj_skills.json') as json_file:
            task_dict = json.load(json_file)
            assert env_name in task_dict.keys(), f"Unrecognized environment name. Choose from {task_dict.keys()}"
            self.obj2skills = task_dict[env_name]
            self.obj_names = list(self.obj2skills.keys())

        with open('config/skill_config.json') as json_file:
            self.skill_dict = json.load(json_file)
            self.num_skills = len(self.skill_dict.keys())

    def get_action(self, topdown_img_path, sideview_img_path, side_camera_id):
        """
        Gets action to be sent to robot side
        """
        obj_name = self.get_obj_selection()
        skill_name = self.get_skill_selection(obj_name)
        params = self.get_param_selection(topdown_img_path, sideview_img_path, side_camera_id, skill_name, obj_name)

        # get one-hot skill selection vector
        skill_idx = self.skill_dict[skill_name]["default_idx"]
        skill_selection_vec = np.zeros(self.num_skills)
        skill_selection_vec[skill_idx] = 1

        # concatenate params
        action = np.concatenate([skill_selection_vec, params])
        return action

    def get_obj_selection(self):
        """
        Let human choose object
        Returns:
            obj_name (str) : name of selected object
        """
        # TODO - run object detector, SSVEP stimulus generation, SSVEP decoding, etc.

        obj_name = input(f"choose obj from {self.obj_names}") # TODO - replace with SSVEP results
        assert (obj_name in self.obj_names), f"object must be one of {self.obj_names}, but got {obj_name}\n"
        return obj_name

    def get_skill_selection(self, obj_name):
        """
        Present human with skill options, and returns name of chosen skill

        Args:
            obj_name (str) : name of object selected by human

        Returns:
            skill_name (str) : name of selected skill
        """
        skill_options = self.obj2skills[obj_name] # list of skills to preesnt to the human, given 
        
        skill_name = input(f"choose skill from {skill_options}") # TODO - replace this from motor imagery results
        assert skill_name in skill_options, f"skill must be one of {skill_options}, but got {skill_name}\n"
        return skill_name

    def get_param_selection(self, topdown_img_path, sideview_img_path, side_camera_id, skill_name, obj_name):
        """
        Given images and name of selected skill, get parameters
        TODO - for future: add cases for skills with more than 3 inputs

        Args:
            topdown_img_path (str) : path to topdown image used for xy param selection
            sideview_mig_path (str) : path to sideview image used for z param selection
            side_camera_id (int) : 0 or 1, corresponding to the sideview image used for z param selection
            skill_name (str) : name of selected skill
            obj_name (str) : name of selected object - TODO remove this after z decoding works

        Returns:
            params (list of floats) : selected parameters for the given skill
        """
        # brings up a UI to click positions on images to select parameters for "pick" skill

        # which params are needed for this skill?
        param_names = self.skill_dict[skill_name]["params"]
        params = []

        # reads images
        topdown_img = cv2.imread(topdown_img_path)
        sideview_img = cv2.imread(sideview_img_path)
        topdown_pil_image = Image.fromarray(topdown_img[:,:,::-1])

        ############### choose x and y ################
        if "xy" in param_names:
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

            ############### choose z ############### - z can be a parameter iff xy is a parameter
            if "z" in param_names:
                # discretize z coordinate and save image visualizing the discrete points
                pix_pts, world_pts = self.detection_utils.get_points_on_z(world_xy=world_xy, img_array=sideview_img, camera_id=side_camera_id, max_height=0.3)
                
                projection_img = cv2.imread("projections.png")
                projection_pil_img = Image.fromarray(projection_img[:,:,::-1])

                # Setup Tkinter window and canvas with side view (cam 0 for param selection) - TODO Replace this block with cursor control
                self.window = tk.Tk()
                self.canvas = tk.Canvas(self.window, width=projection_pil_img.width, height=projection_pil_img.height)
                self.canvas.pack()
                photo = ImageTk.PhotoImage(projection_pil_img)
                self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                self.canvas.bind("<Button 1>", self.get_pixel_position)
                self.window.mainloop()

                # find the point in the discrete pixel points closest to clicked point
                selected_pix_pos = np.array([self.pix_pos[1], self.pix_pos[0]])
                idx = np.argmin(np.linalg.norm(pix_pts - selected_pix_pos, axis=1))
                world_z = world_pts[idx][2]
                params.append(world_z)

                ################ hardcode z according to object ##################
                # hardcoded_pick_z = {
                #     "light blue bowl" : 0.02,
                #     "shiny silver cup" : 0.03,
                #     "green handle" : 0.0,
                # }

                # if "pick" in skill_name:
                #     world_z = hardcoded_pick_z[obj_name]
                # elif "place" in skill_name:
                #     world_z = 0.04

            #############################################

        ################# choose yaw ###################
        if "yaw" in param_names:
            print("Yaw selection is not implemented yet. Appending default value of 0.0")
            params.append(0.0)

        # get full params
        params = [world_xy[0], world_xy[1], world_z]
        print("selected world coordinates", params)
        
        return params

    def get_pixel_position(self, event):
        # Get the position of the mouse click in the image
        x, y = event.x, event.y
        # Print the pixel position
        print("Pixel position:", x, y)
        self.pix_pos = x,y
        self.window.destroy()
        
def main(args):
    demo = FullPipelineDemo(env_name=args.env_name)
    action = demo.get_action(
        topdown_img_path="param_selection_img2.png",
        sideview_img_path="param_selection_img0.png",
        side_camera_id=0,
    )
    print("Action to send to robot is", action)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="TableSetting")
    args = parser.parse_args()

    main(args)