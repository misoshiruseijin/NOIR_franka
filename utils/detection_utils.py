import cv2
import numpy as np
import pdb
import torch
from PIL import Image, ImageDraw, ImageFont
from deoxys.camera_redis_interface import CameraRedisSubInterface
from utils.camera_utils import get_camera_image, get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, pose_inv, project_points_from_base_to_camera
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# TODO - This class should be used in EEG ws side. Currently uses OWL-ViT as detector
class DetectionUtils:

    def __init__(
        self,
    ):

        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

        self.camera_interfaces = {
            0 : CameraRedisSubInterface(camera_id=0),
            1 : CameraRedisSubInterface(camera_id=1),
            2 : CameraRedisSubInterface(camera_id=2),
        }
        for id in self.camera_interfaces.keys():
            self.camera_interfaces[id].start()
        
        self.top_down_corr = {
            # NOTE : Change "pix" coordinates to match cropping. No need to change "world"
            "pix" : [ 
                [11.0, 35.0], # pixel coordinate of top left cross in image (vertical image axis, horizontal image axis)
                [334.0, 33.0], # top right cross
                [337.0, 205.0], # bottom right cross
                [10.0, 213.0], # bottom left cross
            ],
            "world" : [
                [0.27961881, -0.31456524], 
                [0.28143991, 0.25359549], 
                [0.58866825, 0.24407219],
                [0.58267162, -0.31841734],
            ]
        }
        self.get_linear_mapping_coeff(top_down_corr=self.top_down_corr)


    def get_obj_pixel_coord(self, camera_interface, camera_id, texts, thresholds=None, save_img=True, n_instances=1):
        """
        Get center 2d coordinate of detected objects. Grabs current camera frame.

        Args:
            texts : ["text1", "text2", "text3",...] each str describe object to look for
            thresholds (list of floats) : confidence score threshold for each object to look for
            save_img (bool) : if True, save an image visualizing detected objects with bounding box
            n_instances (int) : how many of the same object to find (maximum)

        Returns: 
            coords (dict) : dictionary mapping text to pixel coordinates and score { text : [([x,y], score), ([x,y], score)] }
                { text : [] } for objects not found
        """
        
        # get camera image and conver to hsv
        raw_image = get_camera_image(camera_interface)
        rgb_image = raw_image[:,:,::-1] # convert from bgr to rgb
        image = Image.fromarray(np.uint8(rgb_image))
        
        if thresholds is None:
            thresholds = [0.001] * len(texts)
        obj2thresh = dict(zip(texts, thresholds))

        inputs = self.processor(text=[texts], images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        
        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self.processor.post_process(outputs=outputs, target_sizes=target_sizes)

        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"] # this includes everything detected
     
        # coords = {key : {"coords" : [], "scores" : []} for key in texts} # { text : [ [[list of coords], [list of scores] ] }
        coords = {key : {"boxes" : [], "centers" : [], "scores" : []} for key in texts}

        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            obj_name = texts[label]
            if score >= obj2thresh[obj_name]:
                # print(f"Detected {obj_name} with confidence {round(score.item(), 3)} at location {box}")
                coords[obj_name]["boxes"].append(box)
                coords[obj_name]["scores"].append(score.item())
                center = ( round((box[0] + box[2])/2), round((box[1] + box[3])/2) )
                coords[obj_name]["centers"].append(center)
            
        # extract the top n_instances objects - TODO if needed, allow different max number for each object
        for key in coords:
            # check if there are more than n_instances 
            if len(coords[key]["scores"]) > n_instances:
                scores = coords[key]["scores"]
                indices = np.argsort(scores)[::-1][:n_instances]
                # discard all except top scoring ones
                coords[key]["boxes"] = np.array(coords[key]["boxes"])[indices].tolist()
                coords[key]["scores"] = np.array(coords[key]["scores"])[indices].tolist()
                coords[key]["centers"] = np.array(coords[key]["centers"])[indices].tolist()

        if save_img:
            draw = ImageDraw.Draw(image)
            colors = ["red", "blue", "green", "purple", "orange", "black", "violet", "teal", "darkgreen"]
            idx = 0
            font = ImageFont.truetype('arial.ttf', 24)
            txt_start_pos = (15, 15)
            r = 5 # radius of center circle
            # draw bounding boxes and save image
            for key in coords:
                for box, center in zip(coords[key]["boxes"], coords[key]["centers"]):
                    color = colors[idx]
                    draw.rectangle(box, fill=None, outline=color) # draw bounding box
                    x0, y0 = center[0] - r, center[1] - r
                    x1, y1 = center[0] + r, center[1] + r
                    draw.ellipse([x0, y0, x1, y1], fill=color) # draw center coord
                    draw.text((txt_start_pos[0], txt_start_pos[1]+28*idx), key, font = font, align ="left", fill=color) 
                    idx += 1
            image.save(f"camera{camera_id}.png")
            print(f"Saving camera{camera_id}.png")
        return coords        

    # convert back to 3d world coordinates
    def convert_points_from_camera_to_base(self, X1, X2, I1, I2, E1, E2):
        """
        Args:
            X1, X2 : coordinate in camera 1 frame, coordinate in camera 2 frame
            I1, I2 : camera 1 and 2 intrinsic matrices
            E1, E2 : camera 1 and 2 extrensic matrices wrt robot base
        """
        m1M = np.matmul(I1[:3, :3], E1[:3])
        m2M = np.matmul(I2[:3, :3], E2[:3])

        # X1 = X1[..., ::-1]
        # X2 = X2[..., ::-1]
        X1 = X1.transpose()
        X2 = X2.transpose()

        X = cv2.triangulatePoints(m1M, m2M, X1, X2)
        X = X.transpose()
        X = X / X[:, -1:]

        return X[..., :3]

    # def get_object_world_coords(self, camera_interface0, camera_interface1, texts, thresholds, wait=True):
    #     """
    #     Finds 3d world coordinate of an object with certain color specified by HSV thresholds

    #     Args:
    #         camera_interface0, camera_interface1: camera interfaces for 2 cameras used for 3d coordinate reconstruction
    #         texts (list of str) : descriptors of objects to find
    #         thresholds (list of floats) : detection confidence score thresholds for each object of interest
    #     """

    #     # get 2d pixel coordinates of all objects in both cameras
    #     found_in_0, found_in_1 = False, False # exactly one instance for each object is found
        
    #     # keep trying until all objects are found in both cameras
    #     while not found_in_0 or not found_in_1:
    #         coords0 = self.get_obj_pixel_coord(camera_interface0, camera_id=0, texts=texts, thresholds=thresholds)
    #         pdb.set_trace()
    #         # n_instances = np.array([len(value["scores"]) for value in coords0.values()])
    #         # found_in_0 = np.all(n_instances == 1)
    #         found_in_0 = len() 
    #         coords1 = self.get_obj_pixel_coord(camera_interface1, camera_id=1, texts=texts, thresholds=thresholds)
    #         # n_instances = np.array([len(value["scores"]) for value in coords1.values()])
    #         # found_in_1 = np.all(n_instances == 1)

    #     # get camera intrinsic and extrinsic matrices
    #     camera_to_base_matrix_0 = get_camera_extrinsic_matrix(camera_id=0)
    #     camera_intrinsic_matrix_0 = get_camera_intrinsic_matrix(camera_id=0)
    #     base_to_camera_matrix_0 = pose_inv(camera_to_base_matrix_0)

    #     camera_to_base_matrix_1 = get_camera_extrinsic_matrix(camera_id=1)
    #     camera_intrinsic_matrix_1 = get_camera_intrinsic_matrix(camera_id=1)
    #     base_to_camera_matrix_1 = pose_inv(camera_to_base_matrix_1)

    #     # get 3d position estimate for each object
    #     pos_in_world = {}
    #     for key in coords0:
    #         pos_3d = self.convert_points_from_camera_to_base(
    #             np.array(coords0[key]["centers"][0], dtype=float),
    #             np.array(coords1[key]["centers"][0], dtype=float),
    #             camera_intrinsic_matrix_0,
    #             camera_intrinsic_matrix_1,
    #             base_to_camera_matrix_0,
    #             base_to_camera_matrix_1,
    #         )
    #         pos_in_world[key] = pos_3d.flatten()

    #     return pos_in_world

    def get_object_world_coords(self, camera_interface0, camera_interface1, texts, thresholds, return_2d_coords=False):
        """
        Finds 3d world coordinate of an object with certain color specified by HSV thresholds

        Args:
            camera_interface0, camera_interface1: camera interfaces for 2 cameras used for 3d coordinate reconstruction
            texts (list of str) : descriptors of objects to find
            thresholds (list of floats) : detection confidence score thresholds for each object of interest
        """

        # get 2d pixel coordinates of all objects in both cameras
        found_in_0, found_in_1 = False, False # exactly one instance for each object is found
        
        # keep trying until all objects are found in both cameras
        while not found_in_0 or not found_in_1:
            coords0 = self.get_obj_pixel_coord(camera_interface0, camera_id=0, texts=texts, thresholds=thresholds)
            n_instances = np.array([len(value["scores"]) for value in coords0.values()])
            found_in_0 = np.all(n_instances == 1) 
            coords1 = self.get_obj_pixel_coord(camera_interface1, camera_id=1, texts=texts, thresholds=thresholds)
            n_instances = np.array([len(value["scores"]) for value in coords1.values()])
            found_in_1 = np.all(n_instances == 1)

        # get camera intrinsic and extrinsic matrices
        camera_to_base_matrix_0 = get_camera_extrinsic_matrix(camera_id=0)
        camera_intrinsic_matrix_0 = get_camera_intrinsic_matrix(camera_id=0)
        base_to_camera_matrix_0 = pose_inv(camera_to_base_matrix_0)

        camera_to_base_matrix_1 = get_camera_extrinsic_matrix(camera_id=1)
        camera_intrinsic_matrix_1 = get_camera_intrinsic_matrix(camera_id=1)
        base_to_camera_matrix_1 = pose_inv(camera_to_base_matrix_1)

        # get 3d position estimate for each object
        pos_in_world = {}
        for key in coords0:
            pos_3d = self.convert_points_from_camera_to_base(
                np.array(coords0[key]["centers"][0], dtype=float),
                np.array(coords1[key]["centers"][0], dtype=float),
                camera_intrinsic_matrix_0,
                camera_intrinsic_matrix_1,
                base_to_camera_matrix_0,
                base_to_camera_matrix_1,
            )
            pos_in_world[key] = pos_3d.flatten()

        if return_2d_coords:
            return pos_in_world, coords0, coords1

        return pos_in_world

    # def get_world_xy_from_topdown_view(self, pix_coords, camera_interface=None, visualize=False):
    #     """
    #     Gets world x and y coordinates from pixel coordinate of top down view image using simple linear mapping.
    #     Assumes camera and world frames are parallel and aligned. 
        
    #     Args:
    #         pix_coords (2d array of floats) : pixel coordinates
        
    #     Returns:
    #         world_xy (2-tuple of floats) : x and y world coordinates corresponging to input pixel coordinates
    #     """

    #     if visualize:
    #         assert camera_interface is not None, "camera interface must be provided for visualization"
    #         raw_image = get_camera_image(camera_interface)
    #         rgb_image = raw_image[:,:,::-1] # convert from bgr to rgb
    #         image = Image.fromarray(np.uint8(rgb_image))
    #         draw = ImageDraw.Draw(image)
    #         r = 5 # radius of center circle
    #         # draw bounding boxes and save image
    #         x0, y0 = pix_coords[0] - r, pix_coords[1] - r
    #         x1, y1 = pix_coords[0] + r, pix_coords[1] + r
    #         draw.ellipse([x0, y0, x1, y1], fill="red") # draw center coord
    #         image.save("top_down_params.png")
            
    #     pix_y, pix_x = pix_coords
    #     world_x = self.top_down_coeff["x"][0] * pix_x + self.top_down_coeff["x"][1] 
    #     world_y = self.top_down_coeff["y"][0] * pix_y + self.top_down_coeff["y"][1]
    #     return (world_x, world_y)

    def get_world_xy_from_topdown_view(self, pix_coords, camera_interface, trim=False):
        """
        Gets world x and y coordinates from pixel coordinate of top down view image using simple linear mapping.
        Assumes camera and world frames are parallel and aligned. 
        
        Args:
            pix_coords (2d array of floats) : pixel coordinates
        
        Returns:
            world_xy (2-tuple of floats) : x and y world coordinates corresponging to input pixel coordinates
        """
        trim_low = [90, 130]
        trim_high = [450, 370]
        assert camera_interface is not None, "camera interface must be provided for visualization"
        assert trim == True, "trim should be true!"
        raw_image = get_camera_image(camera_interface)
        rgb_image = raw_image[:,:,::-1] # convert from bgr to rgb
        rgb_image = rgb_image[trim_low[1]:trim_high[1], trim_low[0]:trim_high[0]]
        pix_y, pix_x = pix_coords

        image = Image.fromarray(np.uint8(rgb_image))
        draw = ImageDraw.Draw(image)
        r = 5 # radius of center circle
        x0, y0 = pix_coords[0] - r, pix_coords[1] - r
        x1, y1 = pix_coords[0] + r, pix_coords[1] + r
        draw.ellipse([x0, y0, x1, y1], fill="red") # draw center coord
        image.save("top_down_params.png")
        world_x = self.top_down_coeff["x"][0] * pix_x + self.top_down_coeff["x"][1] 
        world_y = self.top_down_coeff["y"][0] * pix_y + self.top_down_coeff["y"][1]
        return (world_x, world_y)

    def get_linear_mapping_coeff(
        self,
        top_down_corr,
    ):
        """
        Finds linear map between image xy and world xy

        Args:
            top_down_corr (dict) : 
                "pix" : pixel xy coordinates of 4 points
                "world" : world xy coordinates of 4 points corresponding to 4 points in "pix"
        Returns:
            coefficients:
                world_x = pixel_x * solx[0] + solx[1]
                world_y = pixel_y * soly[0] + solx[1]
        """
        # fill in the correspondences for top down view camera
        world = top_down_corr["world"]
        pix = top_down_corr["pix"]
        wx1 = (world[0][0] + world[1][0]) / 2
        wx2 = (world[2][0] + world[3][0]) / 2
        wy1 = (world[0][1] + world[3][1]) / 2
        wy2 = (world[1][1] + world[2][1]) / 2

        px1 = (pix[0][1] + pix[1][1]) / 2
        px2 = (pix[2][1] + pix[3][1]) / 2
        py1 = (pix[0][0] + pix[3][0]) / 2
        py2 = (pix[1][0] + pix[2][0]) / 2
        
        ax = np.array([[px1, 1], [px2, 1]])
        bx = np.array([wx1, wx2])
        solx = np.linalg.solve(ax, bx)
        ay = np.array([[py1, 1], [py2, 1]])
        by = np.array([wy1, wy2])
        soly = np.linalg.solve(ay, by)
        self.top_down_corr = {"x" : solx, "y" : soly}

    def get_points_on_z(self, world_xy, camera_interface, camera_id, max_height=0.3, save_image=False): 
        """
        Given coordinates in 3d world, updates visualization of 3d points in camera image.
        First point is displayed in red and the second point is displayed in blue.
        Yaw angle is shown as a green line, where the direction of the line is the direction the front of the end effector will point to
        Args:
            action : one-hot skill selection vector concatentated with params
        """
        delta = 0.005
        n_points = int(max_height / delta)
        heights = np.linspace(start=0, stop=max_height, num=n_points)

        world_points = np.zeros((n_points, 3))
        world_points[:,0] = world_xy[0]
        world_points[:,1] = world_xy[1]
        world_points[:,2] = heights

        pix_points = np.zeros((n_points, 2))

        for i in range(world_points.shape[0]):
            pix_points[i] = project_points_from_base_to_camera(
                points=world_points[i][np.newaxis,:],
                camera_id=camera_id,
                camera_height=480,
                camera_width=640,
            )

        if save_image:
            raw_image = get_camera_image(camera_interface)
            rgb_image = raw_image[:,:,::-1] # convert from bgr to rgb
            image = Image.fromarray(np.uint8(rgb_image))
            draw = ImageDraw.Draw(image)
            for pix in pix_points:
                # print("drawing point at ", pix)
                draw.point((pix[1], pix[0]), "blue")
            image.save("projections.png")

        return pix_points