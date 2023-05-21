
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
from environments.realrobot_env_multi import RealRobotEnvMulti

from PIL import Image, ImageDraw
import cv2
import json
import time
import os


class FullPipelineDemo:
    def __init__(
        self,
        env_name,
    ):

        """
        Full pipeline demo without EEG or object detection.
        Commandline input for object and skill selection, mouseclick input for param selection.

        Args:
            env_name (str) : name of environment. used to get correct objects
        """
        self.pix_pos = None
        self.chose_xy = False
        self.chose_z = False
        self.canvas = None # TODO - not needed if not using tkinter UI
        self.window = None # TODO - not needed if not using tkinter UI
        self.detection_utils = DetectionUtils()
        self.env = RealRobotEnvMulti()
        self.img_obs = self.env.get_image_observations(save_images=True)

        with open('config/task_obj_skills.json') as json_file:
            task_dict = json.load(json_file)
            assert env_name in task_dict.keys(), f"Unrecognized environment name. Choose from {task_dict.keys()}"
            self.obj2skills = task_dict[env_name]
            self.obj_names = list(self.obj2skills.keys())

        with open('config/skill_config.json') as json_file:
            self.skill_dict = json.load(json_file)
            self.num_skills = len(self.skill_dict.keys())

    def take_action(self, side_camera_id):
        """
        Gets action to be sent to robot side
        """
        obj_name = self.get_obj_selection()
        skill_name = self.get_skill_selection(obj_name)

        topdown_img = self.img_obs["param_image2"]
        sideview_img = self.img_obs[f"param_image{side_camera_id}"]
        params = self.get_param_selection(topdown_img, sideview_img, side_camera_id, skill_name)

        # get one-hot skill selection vector
        skill_idx = self.skill_dict[skill_name]["default_idx"]
        skill_selection_vec = np.zeros(self.num_skills)
        skill_selection_vec[skill_idx] = 1

        # execute action
        action = np.concatenate([skill_selection_vec, params])
        obs, reward, done, info = self.env.step(action)

        # update img_obs
        self.img_obs = self.env.get_image_observations(action=action, save_images=True)

        return obj_name, skill_name

    def get_obj_selection(self):
        """
        Let human choose object
        Returns:
            obj_name (str) : name of selected object
        """
        # TODO - run object detector, SSVEP stimulus generation, SSVEP decoding, etc.

        obj_name = input(f"choose obj from {self.obj_names}\n") # TODO - replace with SSVEP results
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
        
        skill_name = input(f"choose skill from {skill_options}\n") # TODO - replace this from motor imagery results
        assert skill_name in skill_options, f"skill must be one of {skill_options}, but got {skill_name}\n"
        return skill_name

    def get_param_selection(self, topdown_img, sideview_img, side_camera_id, skill_name):
        """
        Given images and name of selected skill, get parameters
        TODO - for future: add cases for skills with more than 3 inputs

        Args:
            topdown_img_path (str) : path to topdown image used for xy param selection
            sideview_mig_path (str) : path to sideview image used for z param selection
            side_camera_id (int) : 0 or 1, corresponding to the sideview image used for z param selection

        Returns:
            params (list of floats) : selected parameters for the given skill
        """
        # brings up a UI to click positions on images to select parameters for "pick" skill

        # reads images
        topdown_pil_image = Image.fromarray(topdown_img[:,:,::-1])

        # which params are needed for this skill?
        param_names = self.skill_dict[skill_name]["params"]
        params = []

        if "xy" in param_names:
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

            if "z" in param_names:
                ############### choose z ###############
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

        return params

    def get_pixel_position(self, event):
        # Get the position of the mouse click in the image
        x, y = event.x, event.y
        # Print the pixel position
        print("Pixel position:", x, y)
        self.pix_pos = x,y
        self.window.destroy()

def ask_for_task_state():
    """
    Ask for human input to determine if task is done and/or failed
    """
    yes_no = ["y", "n"]
    done = False
    failed = False
    task_state = ""
    while task_state not in yes_no:
        task_state = input("Is task complete or in unrecoverable state? (y/n)\n")
    if task_state == "n":
        # task has not completed (or failed)
        return done, failed
    else:
        done = True
        task_state = ""
        while task_state not in yes_no:
            task_state = input("Was task successful? (y/n)\n")
        failed = True if task_state == "n" else False
        return done, failed

def ask_for_execution_failure():
    """
    Ask if execution failure happened
    """
    yes_no = ["y", "n"]
    status = ""
    while status not in yes_no:
        status = input("Skill execution failed? (y/n)")
    execution_failure = True if status == "y" else False
    return execution_failure

def main(args):

    demo = FullPipelineDemo(env_name=args.env_name)

    # don't record anything. just run the demo
    if not args.record:
        while True:
            demo.take_action(side_camera_id=0)

    # record clock time, selected objects, executed skills, execution failure, episode success/failure, episode length
    else:
        save_path = f"{args.save_dir}/{args.env_name}"
        os.makedirs(save_path, exist_ok=True)
        full_data = {}
        for i in range(args.n_iter):
            
            obj_names, skill_names, execution_failures = [], [], []
            start_time = time.time()
            done = False
            failed = False
            
            # run episode
            while not done:
                # take action and check task status, execution failure
                obj_name, skill_name = demo.take_action(side_camera_id=0)
                execution_failure = ask_for_execution_failure() # was skill execution successful?
                done, failed = ask_for_task_state() # has episode ended?

                # record
                obj_names.append(obj_name)
                skill_names.append(skill_name)
                execution_failures.append(execution_failure)

            end_time = time.time()
            full_data["episodes"][f"episode{i}"] = {
                "objects" : obj_names,
                "skills" : skill_names,
                "execution_failures" : execution_failures,
                "success" : not failed,
                "failed" : failed,
                "episode_len" : len(skill_names),
                "clock_time" : end_time - start_time,
            }

        # save data
        with open(f"{save_path}/data.json", "x") as outfile:
            json.dump(full_data, outfile, indent=4)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="TableSetting")
    parser.add_argument("--n_eps", type=int, default=1) # number of episodes
    parser.add_argument("--record", action="store_true") # whether to record data
    parser.add_argument("--save_dir", type=str, default="robot_standalone_experiments")
    args = parser.parse_args()

    main(args)