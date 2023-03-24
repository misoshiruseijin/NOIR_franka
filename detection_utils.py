import cv2
import numpy as np
import pdb

from deoxys.camera_redis_interface import CameraRedisSubInterface
from camera_utils import get_camera_image, get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, pose_inv


# define hsv thresholds
# SAUSAGE_LOW = np.array([0, 120, 20])
# SAUSAGE_HIGH = np.array([5, 255, 255])

# # start camera interfaces
# camera_interfaces = {
#     0 : CameraRedisSubInterface(camera_id=0),
#     1 : CameraRedisSubInterface(camera_id=1),
# }

# for id in camera_interfaces.keys():
#     camera_interfaces[id].start()

class DetectionUtils:

    def __init__(
        self,
        detector_params=None
    ):

        self.params = detector_params
        if detector_params is None:
            self.params = cv2.SimpleBlobDetector_Params()
            self.params.filterByArea = True
            self.params.filterByColor = False
            self.params.filterByCircularity = False
            self.params.filterByConvexity = False
            self.params.filterByInertia = False
            self.params.minArea = 150.0
            self.params.maxArea = 15000.0
        
        self.detector = cv2.SimpleBlobDetector_create(self.params)

    def get_obj_pixel_coord(self, camera_interface, camera_id, hsv_low, hsv_high, obj_name="", save_img=True, return_size=False, dilation=False):
        """
        Get center 2d coordinate of detected blob in image frame
        """
        # camera_interface = camera_interfaces[camera_id]
        # get camera image and conver to hsv
        raw_image = get_camera_image(camera_interface)
        hsv_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv_image, hsv_low, hsv_high)

        if dilation:
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        masked = raw_image.copy()
        masked = cv2.bitwise_and(masked, masked, mask=mask)

        # input mask to detector and get center coordinates of blobs
        keypoints = list(self.detector.detect(mask))

        if len(keypoints) == 0:
            # print("No blobs found")        
            return None
        
        else:
            keypoint = keypoints[0]
            if len(keypoints) > 1:
                # if more than one keypoints are detected, choose the blob with largest area
                for k in keypoints:
                    if k.size > keypoint.size:
                        keypoint = k
        
        center_pix = np.array(keypoint.pt)
        blob_size = keypoint.size

        blank = np.zeros((1, 1))
        blobs = cv2.drawKeypoints(mask, [keypoint], blank, (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        if save_img:
            # cv2.imwrite(f"images/raw{camera_id}_{obj_name}.png", raw_image)
            # cv2.imwrite(f"images/mask{camera_id}_{obj_name}.png", mask)
            # cv2.imwrite(f"images/result{camera_id}_{obj_name}.png", masked)
            cv2.imwrite(f"images/blobs{camera_id}_{obj_name}.png", blobs)

        if not return_size:
            return np.array([center_pix])

        return np.array([center_pix]), blob_size

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

    def get_object_world_coords(self, camera_interface0, camera_interface1, hsv_low, hsv_high, obj_name="", wait=True, dilation=False):
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
                pos_in_cam0 = self.get_obj_pixel_coord(camera_interface0, camera_id=0, hsv_low=hsv_low, hsv_high=hsv_high, obj_name=obj_name, dilation=dilation)
            while pos_in_cam1 is None:
                print(f"looking for object {obj_name} in camera 1")
                pos_in_cam1 = self.get_obj_pixel_coord(camera_interface1, camera_id=1, hsv_low=hsv_low, hsv_high=hsv_high, obj_name=obj_name, dilation=dilation)
        else:
            pos_in_cam0 = self.get_obj_pixel_coord(camera_interface0, camera_id=0, hsv_low=hsv_low, hsv_high=hsv_high, obj_name=obj_name, dilation=dilation)
            pos_in_cam1 = self.get_obj_pixel_coord(camera_interface1, camera_id=1, hsv_low=hsv_low, hsv_high=hsv_high, obj_name=obj_name, dilation=dilation)

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







# # setup detector
# params = cv2.SimpleBlobDetector_Params()
# params.filterByArea = True
# params.filterByColor = False
# params.filterByCircularity = False
# params.filterByConvexity = False
# params.filterByInertia = False
# params.minArea = 150.0
# params.maxArea = 15000.0
# detector = cv2.SimpleBlobDetector_create(params)

# def get_obj_pixel_coord(camera_interface, camera_id, hsv_low, hsv_high, obj_name="", save_img=True, return_size=False):
#     """
#     Get center 2d coordinate of detected blob in image frame
#     """
#     # camera_interface = camera_interfaces[camera_id]
#     # get camera image and conver to hsv
#     raw_image = get_camera_image(camera_interface)
#     hsv_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)
    
#     mask = cv2.inRange(hsv_image, hsv_low, hsv_high)
#     masked = raw_image.copy()
#     masked = cv2.bitwise_and(masked, masked, mask=mask)

#     # input mask to detector and get center coordinates of blobs
#     keypoints = list(detector.detect(mask))

#     if len(keypoints) == 0:
#         # print("No blobs found")        
#         return None
    
#     else:
#         keypoint = keypoints[0]
#         if len(keypoints) > 1:
#             # if more than one keypoints are detected, choose the blob with largest area
#             for k in keypoints:
#                 if k.size > keypoint.size:
#                     keypoint = k
    
#     center_pix = np.array(keypoint.pt)
#     blob_size = keypoint.size

#     blank = np.zeros((1, 1))
#     blobs = cv2.drawKeypoints(mask, [keypoint], blank, (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
#     if save_img:
#         # cv2.imwrite(f"images/raw{camera_id}_{obj_name}.png", raw_image)
#         # cv2.imwrite(f"images/mask{camera_id}_{obj_name}.png", mask)
#         # cv2.imwrite(f"images/result{camera_id}_{obj_name}.png", masked)
#         cv2.imwrite(f"images/blobs{camera_id}_{obj_name}.png", blobs)

#     if not return_size:
#         return np.array([center_pix])

#     return np.array([center_pix]), blob_size

# # convert back to 3d world coordinates
# def convert_points_from_camera_to_base(X1, X2, I1, I2, E1, E2):
#     """
#     Args:
#         X1, X2 : coordinate in camera 1 frame, coordinate in camera 2 frame
#         I1, I2 : camera 1 and 2 intrinsic matrices
#         E1, E2 : camera 1 and 2 extrensic matrices wrt robot base
#     """
#     m1M = np.matmul(I1[:3, :3], E1[:3])
#     m2M = np.matmul(I2[:3, :3], E2[:3])

#     # X1 = X1[..., ::-1]
#     # X2 = X2[..., ::-1]
#     X1 = X1.transpose()
#     X2 = X2.transpose()

#     X = cv2.triangulatePoints(m1M, m2M, X1, X2)
#     X = X.transpose()
#     X = X / X[:, -1:]

#     return X[..., :3]

# def get_object_world_coords(camera_interface0, camera_interface1, hsv_low, hsv_high, obj_name=""):
#     """
#     Finds 3d world coordinate of an object with certain color specified by HSV thresholds

#     Args:
#         camera_interface0, camera_interface1: camera interfaces for 2 cameras used for 3d coordinate reconstruction
#         hsv_low, hsv_high : HSV thresholds for object to detect
#     """
#     # get object pixel positions in both cameras
#     pos_in_cam0 = None
#     pos_in_cam1 = None
#     while pos_in_cam0 is None:
#         print(f"looking for object {obj_name} in camera 0")
#         pos_in_cam0 = get_obj_pixel_coord(camera_interface0, camera_id=0, hsv_low=hsv_low, hsv_high=hsv_high, obj_name=obj_name)
#     while pos_in_cam1 is None:
#         print(f"looking for object {obj_name} in camera 1")
#         pos_in_cam1 = get_obj_pixel_coord(camera_interface1, camera_id=1, hsv_low=hsv_low, hsv_high=hsv_high, obj_name=obj_name)
    
#     # get camera intrinsic and extrinsic matrices
#     camera_to_base_matrix_0 = get_camera_extrinsic_matrix(camera_id=0)
#     camera_intrinsic_matrix_0 = get_camera_intrinsic_matrix(camera_id=0)
#     base_to_camera_matrix_0 = pose_inv(camera_to_base_matrix_0)

#     camera_to_base_matrix_1 = get_camera_extrinsic_matrix(camera_id=1)
#     camera_intrinsic_matrix_1 = get_camera_intrinsic_matrix(camera_id=1)
#     base_to_camera_matrix_1 = pose_inv(camera_to_base_matrix_1)

#     # estimate position in world coordinate (robot base frame)
#     pos_in_world = convert_points_from_camera_to_base(
#         pos_in_cam0,
#         pos_in_cam1,
#         camera_intrinsic_matrix_0,
#         camera_intrinsic_matrix_1,
#         base_to_camera_matrix_0,
#         base_to_camera_matrix_1
#     )

#     return pos_in_world.flatten()

