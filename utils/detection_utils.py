import cv2
import numpy as np
import pdb
import torch
from PIL import Image, ImageDraw
from deoxys.camera_redis_interface import CameraRedisSubInterface
from utils.camera_utils import get_camera_image, get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, pose_inv
from transformers import OwlViTProcessor, OwlViTForObjectDetection


class DetectionUtils:

    def __init__(
        self,
    ):

        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")


    # TODO - replace this with OWL-VIT
    def get_obj_pixel_coord(self, camera_interface, camera_id, obj_name="", save_img=True):
        """
        Get center 2d coordinate of detected blob in image frame
        """
        # camera_interface = camera_interfaces[camera_id]
        # get camera image and conver to hsv
        raw_image = get_camera_image(camera_interface)
        rgb_image = raw_image[:,:,::-1] # convert from bgr to rgb
        image = Image.fromarray(np.uint8(rgb_image))
        
        texts = [["blue bowl", "red spoon", "silver cup"]] # TODO - this should be input to the function
        
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        
        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self.processor.post_process(outputs=outputs, target_sizes=target_sizes)

        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        score_threshold = 0.07 # TODO - define difference thresh for different objects (should be input arg)
        bounding_boxes = []
        center_coords = []

        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            if score >= score_threshold:
                bounding_boxes.append(box)
                center = ( (box[0] + box[2])/2, (box[1] + box[3])/2 )
                center_coords.append(center)
                print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

        # TODO - clean up code
        for box, center in zip(bounding_boxes, center_coords):

            # create rectangle image
            img1 = ImageDraw.Draw(image)  
            img1.rectangle(box, fill=None, outline="red")
            r = 5
            x0, y0 = center[0] - r, center[1] - r
            x1, y1 = center[0] + r, center[1] + r
            img1.ellipse([x0, y0, x1, y1], fill="red")
        image.show()
        pdb.set_trace()
        # TODO - return pixel coordinate

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

    def get_object_world_coords(self, camera_interface0, camera_interface1, obj_name="", wait=True):
        """
        Finds 3d world coordinate of an object with certain color specified by HSV thresholds

        Args:
            camera_interface0, camera_interface1: camera interfaces for 2 cameras used for 3d coordinate reconstruction
            hsv_low, hsv_high : HSV thresholds for object to detect
        """
        # get object pixel positions in both cameras
        pos_in_cam0 = None
        pos_in_cam1 = None

        if wait:
            while pos_in_cam0 is None:
                print(f"looking for object {obj_name} in camera 0")
                pos_in_cam0 = self.get_obj_pixel_coord(camera_interface0, camera_id=0, obj_name=obj_name)
            while pos_in_cam1 is None:
                print(f"looking for object {obj_name} in camera 1")
                pos_in_cam1 = self.get_obj_pixel_coord(camera_interface1, camera_id=1, obj_name=obj_name)
        else:
            pos_in_cam0 = self.get_obj_pixel_coord(camera_interface0, camera_id=0, obj_name=obj_name)
            pos_in_cam1 = self.get_obj_pixel_coord(camera_interface1, camera_id=1, obj_name=obj_name)

        # get camera intrinsic and extrinsic matrices
        camera_to_base_matrix_0 = get_camera_extrinsic_matrix(camera_id=0)
        camera_intrinsic_matrix_0 = get_camera_intrinsic_matrix(camera_id=0)
        base_to_camera_matrix_0 = pose_inv(camera_to_base_matrix_0)

        camera_to_base_matrix_1 = get_camera_extrinsic_matrix(camera_id=1)
        camera_intrinsic_matrix_1 = get_camera_intrinsic_matrix(camera_id=1)
        base_to_camera_matrix_1 = pose_inv(camera_to_base_matrix_1)

        # estimate position in world coordinate (robot base frame)
        if pos_in_cam0 is None or pos_in_cam1 is None:
            return None

        pos_in_world = self.convert_points_from_camera_to_base(
            pos_in_cam0,
            pos_in_cam1,
            camera_intrinsic_matrix_0,
            camera_intrinsic_matrix_1,
            base_to_camera_matrix_0,
            base_to_camera_matrix_1
        )

        return pos_in_world.flatten()




