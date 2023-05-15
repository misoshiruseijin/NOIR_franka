from environments.realrobot_env_multi import RealRobotEnvMulti
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import cv2
from utils.detection_utils_eeg import DetectionUtils

class ParamSelectionDemo:
    def __init__(self):
        self.pix_pos = None
        self.chose_xy = False
        self.chose_z = False
        self.canvas = None
        self.window = None
        self.env = RealRobotEnvMulti()
        self.detection_utils = DetectionUtils()

    def demo_pick(self):
        # brings up a UI to click positions on images to select parameters for "pick" skill

        # take images and display cam 2 image
        dummy_action = np.zeros(self.env.skill.num_skills)
        imgs = self.env.get_image_observations(action=dummy_action, img_as_list=False, save_images=False)
        param_img2 = imgs["param_image2"]
        param_img0 = imgs["param_image0"]
        pil_image0 = Image.fromarray(param_img0[:,:,::-1])
        pil_image2 = Image.fromarray(param_img2[:,:,::-1])

        ############### choose x and y ################
        # Setup Tkinter window and canvas with topdown view (cam 2 for param selection) - NOTE this block can be replaced with 
        self.window = tk.Tk()
        self.canvas = tk.Canvas(self.window, width=pil_image2.width, height=pil_image2.height)
        self.canvas.pack()
        photo = ImageTk.PhotoImage(pil_image2)
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.canvas.bind("<Button 1>", self.get_pixel_position) # bind mouseclick event to canvas
        self.window.mainloop() # start Tkinter event loop

        # get xy world position from pixel position
        world_xy = self.detection_utils.get_world_xy_from_topdown_view(pix_coords=self.pix_pos, img_array=param_img2)
        
        ############### choose z ###############
        # discretize z coordinate and save image visualizing the discrete points
        pix_pts, world_pts = self.detection_utils.get_points_on_z(world_xy=world_xy, img_array=param_img0, camera_id=0, max_height=0.3)
        
        projection_img = cv2.imread("projections.png")
        projection_pil_img = Image.fromarray(projection_img[:,:,::-1])

        # Setup Tkinter window and canvas with side view (cam 0 for param selection)
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

        # execute pick
        skill_selection_vec = np.zeros(self.env.num_skills)
        skill_selection_vec[0] = 1
        params = [world_xy[0], world_xy[1], world_z]
        print("selected world coordinates", params)
        action = np.concatenate([skill_selection_vec, params])
        obs, reward, done, info = self.env.step(action)

    def get_pixel_position(self, event):
        # Get the position of the mouse click in the image
        x, y = event.x, event.y
        # Print the pixel position
        print("Pixel position:", x, y)
        self.pix_pos = x,y
        self.window.destroy()

def main():
    demo = ParamSelectionDemo()
    demo.demo_pick()

if __name__ == "__main__":
    main()