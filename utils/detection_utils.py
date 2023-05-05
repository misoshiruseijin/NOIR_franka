import cv2
import numpy as np
import pdb
import torch
from PIL import Image, ImageDraw, ImageFont
from deoxys.camera_redis_interface import CameraRedisSubInterface
from utils.camera_utils import get_camera_image, get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, pose_inv
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# TODO - This class should be used in EEG ws side. Currently uses OWL-ViT as detector
class DetectionUtils:

    def __init__(
        self,
    ):

        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    
    # def get_obj_pixel_coord(self, camera_interface, camera_id, texts, thresholds=None, save_img=True, n_instances=1):
    #     """
    #     Get center 2d coordinate of detected blob in image frame

    #     Args:
    #         texts : ["text1", "text2", "text3",...] each str describe object to look for
    #         thresholds (list of floats) : confidence score threshold for each object to look for
    #         save_img (bool) : if True, save an image visualizing detected objects with bounding box
    #         n_instances (int) : how many of the same object to find (maximum)

    #     Returns: 
    #         coords (dict) : dictionary mapping text to pixel coordinates and score { text : [([x,y], score), ([x,y], score)] }
    #             { text : [] } for objects not found
    #     """
        
    #     assert n_instances == 1, "n_instances > 1 is currently not supported"

    #     # get camera image
    #     image = Image.fromarray(np.uint8(get_camera_image(camera_interface)[:,:,::-1]))
        
    #     if thresholds is None:
    #         thresholds = [0.001] * len(texts)
    #     obj2thresh = dict(zip(texts, thresholds))
    #     obj2highscore = {text : 0.0 for text in texts}

    #     inputs = self.processor(text=[texts], images=image, return_tensors="pt")
    #     outputs = self.model(**inputs)
        
    #     # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    #     target_sizes = torch.Tensor([image.size[::-1]])
    #     # Convert outputs (bounding boxes and class logits) to COCO API
    #     results = self.processor.post_process(outputs=outputs, target_sizes=target_sizes)

    #     i = 0  # Retrieve predictions for the first image for the corresponding text queries
    #     boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"] # this includes everything detected
     
    #     # coords = {key : {"coords" : [], "scores" : []} for key in texts} # { text : [ [[list of coords], [list of scores] ] }
    #     coords = {key : {"boxes" : [], "centers" : [], "scores" : []} for key in texts}

    #     for box, score, label in zip(boxes, scores, labels):
    #         box = [round(i, 2) for i in box.tolist()]
    #         obj_name = texts[label]
    #         if score >= obj2thresh[obj_name] and score > obj2highscore[obj_name]:
    #             # print(f"Detected {obj_name} with confidence {round(score.item(), 3)} at location {box}")
    #             coords[obj_name]["boxes"] = box
    #             coords[obj_name]["scores"] = score.item()
    #             center = ( round((box[0] + box[2])/2), round((box[1] + box[3])/2) )
    #             coords[obj_name]["centers"] = center
    
    #     if save_img:
    #         draw = ImageDraw.Draw(image)
    #         colors = ["red", "blue", "green", "purple", "orange", "black", "violet", "teal", "darkgreen"]
    #         idx = 0
    #         font = ImageFont.truetype('arial.ttf', 24)
    #         txt_start_pos = (15, 15)
    #         r = 5 # radius of center circle
    #         # draw bounding boxes and save image
    #         for key in coords:
    #             box = coords[key]["boxes"]
    #             center = coords[key]["centers"]
    #             color = colors[idx]
    #             x0, y0 = center[0] - r, center[1] - r
    #             x1, y1 = center[0] + r, center[1] + r
    #             draw.ellipse([x0, y0, x1, y1], fill=color) # draw center coord
    #             draw.text((txt_start_pos[0], txt_start_pos[1]+28*idx), key, font = font, align ="left", fill=color) 
    #             idx += 1
    #         image.save(f"camera{camera_id}.png")
    #         print(f"Saving camera{camera_id}.png")
    #     return coords        

    def get_obj_pixel_coord(self, camera_interface, camera_id, texts, thresholds=None, save_img=True, n_instances=1):
        """
        Get center 2d coordinate of detected blob in image frame

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



