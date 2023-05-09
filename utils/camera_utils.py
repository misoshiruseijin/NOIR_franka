"""
Largely inherited from UT Austin RPL rpl_vision_utilities
"""

"""
GPRS version of testing front camera calibration

Script to test camera calibration by reading robot end effector pose and projecting it onto camera image.
Many functions are based on https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/utils/camera_utils.py
"""

import time
import cv2
import imageio
import numpy as np
from PIL import Image, ImageDraw

from deoxys.franka_interface import FrankaInterface
from deoxys.camera_redis_interface import CameraRedisSubInterface
from deoxys import config_root


def get_camera_intrinsic_matrix(camera_id=0):
    """
    Fill out this function to put the intrinsic matrix of your camera.
    Returns:
        K (np.array): 3x3 camera matrix
    """
    if camera_id == 0:
        K = np.array([
            [605.32720947,   0.       ,  325.39172363],
            [ 0.         ,605.03729248, 239.00517273],
            [0., 0., 1.],
        ])
    elif camera_id == 1:
        K = np.array([
            [614.56201172,   0.      ,   296.10009766],
            [0.         ,614.26635742 ,240.11764526],
            [0., 0., 1.],
        ])
    elif camera_id == 2:
        K= np.array([
            [608.31689453,   0.,         331.44717407],
            [  0.,         607.91522217, 245.0897522 ],
            [  0.,           0.,           1.        ],
        ])
        
    return K


def get_camera_extrinsic_matrix(camera_id=0):
    """
    Fill out this function to put the extrinsic matrix of your camera.
    This should correspond to the camera pose in the robot base frame. 
    Returns:
        R (np.array): 4x4 camera extrinsic matrix
    """
    # New values taking distortion into consideration
    R = np.eye(4)

    if camera_id == 0:
        R[:3, :3] = np.array([
            [ 0.89342926,  0.29425294, -0.33941033],
            [ 0.44116528, -0.71707462,  0.53960836],
            [-0.08460119, -0.63183795, -0.77046962],
        ])
        R[:3, 3] = np.array([ 0.61526343, -0.44320185,  0.71633382])
    if camera_id == 1:
        R[:3, :3] = np.array([
            [-0.8516527,  0.11228411, -0.51193746],
            [ 0.48210559,  0.55097077, -0.68117943],
            [ 0.20557695, -0.82693621, -0.52336854],
        ])
        R[:3, 3] = np.array([0.70538269, 0.42807427, 0.45873377])
    if camera_id == 2:
        R = np.eye(4)
        R[:3, :3] = np.array([
            [-0.88914262,  0.27376638, -0.36671158],
            [ 0.45761172,  0.52468279, -0.71784364],
            [-0.0041142,  -0.80607688, -0.59179653],
        ])
        R[:3, 3] = np.array([0.67537891, 0.43226307, 0.49220737])
        return R

    return R


def pose_inv(pose):
    """
    Computes the inverse of a homogeneous matrix corresponding to the pose of some
    frame B in frame A. The inverse is the pose of frame A in frame B.
    Args:
        pose (np.array): 4x4 matrix for the pose to inverse
    Returns:
        np.array: 4x4 matrix for the inverse pose
    """

    # Note, the inverse of a pose matrix is the following
    # [R t; 0 1]^-1 = [R.T -R.T*t; 0 1]

    # Intuitively, this makes sense.
    # The original pose matrix translates by t, then rotates by R.
    # We just invert the rotation by applying R-1 = R.T, and also translate back.
    # Since we apply translation first before rotation, we need to translate by
    # -t in the original frame, which is -R-1*t in the new frame, and then rotate back by
    # R-1 to align the axis again.

    pose_inv = np.zeros((4, 4))
    pose_inv[:3, :3] = pose[:3, :3].T
    pose_inv[:3, 3] = -pose_inv[:3, :3].dot(pose[:3, 3])
    pose_inv[3, 3] = 1.0
    return pose_inv

def get_robot_eef_position(robot_interface):
    """
    Fill out this function to get the robot end effector position in the robot base frame.
    Returns:
        e (np.array): end effector position of shape (3,)
    """
    # e = np.zeros(3)
    # return e
    last_robot_state = robot_interface._state_buffer[-1]
    ee_pose = np.array(last_robot_state.O_T_EE).reshape((4, 4)).T
    return ee_pose[:3, 3]


def get_camera_image(camera_interface):
    """
    Fill out this function to get an RGB image from the camera.
    Returns:
        I (np.array): array of shape (H, W, 3) and type np.uint8
    """
    # I = np.zeros((480, 640, 3), dtype=np.uint8)
    # return I

    # get image
    imgs = camera_interface.get_img()
    im = imgs["color"]
    
    # resize image
    im_size = (480, 640)
    im = Image.fromarray(im).resize((im_size[1], im_size[0]), Image.BILINEAR)
    return np.array(im).astype(np.uint8)


def get_camera_transform_matrix(camera_id=0):
    """
    Camera transform matrix to project from world coordinates to pixel coordinates.
    Returns:
        K (np.array): 4x4 camera matrix to project from world coordinates to pixel coordinates
    """
    R = get_camera_extrinsic_matrix(camera_id=camera_id)
    K = get_camera_intrinsic_matrix(camera_id=camera_id)
    K_exp = np.eye(4)
    K_exp[:3, :3] = K

    # Takes a point in world, transforms to camera frame, and then projects onto image plane.
    return K_exp @ pose_inv(R)


def pose_inv(pose):
    """
    Computes the inverse of a homogeneous matrix corresponding to the pose of some
    frame B in frame A. The inverse is the pose of frame A in frame B.
    Args:
        pose (np.array): 4x4 matrix for the pose to inverse
    Returns:
        np.array: 4x4 matrix for the inverse pose
    """

    # Note, the inverse of a pose matrix is the following
    # [R t; 0 1]^-1 = [R.T -R.T*t; 0 1]

    # Intuitively, this makes sense.
    # The original pose matrix translates by t, then rotates by R.
    # We just invert the rotation by applying R-1 = R.T, and also translate back.
    # Since we apply translation first before rotation, we need to translate by
    # -t in the original frame, which is -R-1*t in the new frame, and then rotate back by
    # R-1 to align the axis again.

    pose_inv = np.zeros((4, 4))
    pose_inv[:3, :3] = pose[:3, :3].T
    pose_inv[:3, 3] = -pose_inv[:3, :3].dot(pose[:3, 3])
    pose_inv[3, 3] = 1.0
    return pose_inv


def project_points_from_base_to_camera(points, camera_id, camera_height, camera_width):
    """
    Helper function to project a batch of points in the base frame
    into camera pixels using the base to camera transformation.
    Args:
        points (np.array): 3D points in base frame to project onto camera pixel locations. Should
            be shape [..., 3].
        base_to_camera_transform (np.array): 4x4 Tensor to go from robot coordinates to pixel
            coordinates.
        camera_height (int): height of the camera image
        camera_width (int): width of the camera image
    Return:
        pixels (np.array): projected pixel indices of shape [..., 2]
    """
    base_to_camera_transform = get_camera_transform_matrix(camera_id=camera_id)

    assert points.shape[-1] == 3  # last dimension must be 3D
    assert len(base_to_camera_transform.shape) == 2
    assert base_to_camera_transform.shape[0] == 4 and base_to_camera_transform.shape[1] == 4

    # convert points to homogenous coordinates -> (px, py, pz, 1)
    ones_pad = np.ones(points.shape[:-1] + (1,))
    points = np.concatenate((points, ones_pad), axis=-1)  # shape [..., 4]

    # batch matrix multiplication of 4 x 4 matrix and 4 x 1 vectors to do robot frame to pixels transform
    mat_reshape = [1] * len(points.shape[:-1]) + [4, 4]
    cam_trans = base_to_camera_transform.reshape(mat_reshape)  # shape [..., 4, 4]
    pixels = np.matmul(cam_trans, points[..., None])[..., 0]  # shape [..., 4]

    # re-scaling from homogenous coordinates to recover pixel values
    # (x, y, z) -> (x / z, y / z)
    pixels = pixels / pixels[..., 2:3]
    pixels = pixels[..., :2].round().astype(int)  # shape [..., 2]

    # swap first and second coordinates to get pixel indices that correspond to (height, width)
    # and also clip pixels that are out of range of the camera image
    pixels = np.concatenate(
        (
            pixels[..., 1:2].clip(0, camera_height - 1),
            pixels[..., 0:1].clip(0, camera_width - 1),
        ),
        axis=-1,
    )

    return pixels