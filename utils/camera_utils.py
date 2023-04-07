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


def get_camera_intrinsic_matrix(camera_id):
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
        
    return K


def get_camera_extrinsic_matrix(camera_id):
    """
    Fill out this function to put the extrinsic matrix of your camera.
    This should correspond to the camera pose in the robot base frame. 
    Returns:
        R (np.array): 4x4 camera extrinsic matrix
    """
    # R = np.eye(4)

    # if camera_id == 0:
    #     R[:3, :3] = np.array([
    #         [ 0.87339449,  0.36883096, -0.31803426],
    #         [ 0.48664941, -0.63571221,  0.59920142],
    #         [ 0.01882577, -0.6781104,  -0.73471891]
    #     ])
    #     R[:3, 3] = np.array([0.61565302, -0.46807638,  0.59688543])

    # if camera_id == 1:
    #     R[:3, :3] = np.array([
    #         [-0.79786389,  0.46530727, -0.38327843],
    #         [ 0.60163534,  0.57447786, -0.55498659],
    #         [-0.03805433, -0.6733976,  -0.73830044]
    #     ])
    #     R[:3, 3] = np.array([0.66133318, 0.3837803,  0.54324011])
        
    # return R

    # New values taking distortion into consideration
    R = np.eye(4)

    if camera_id == 0:
        R[:3, :3] = np.array([
            [ 0.85744557,  0.29368317, -0.42253672],
            [ 0.50697709 ,-0.62272824 , 0.59597295],
            [-0.08809832 ,-0.72523081, -0.68284622],
        ])
        R[:3, 3] = np.array([0.62791933, -0.44117322,  0.60937688])

    if camera_id == 1:
        R[:3, :3] = np.array([
            [-0.80475523, 0.48178413, -0.34677526],
            [0.58780064, 0.56526125, -0.57876603],
            [-0.08282167, -0.66959971, -0.73808997],
        ])
        R[:3, 3] = np.array([0.62760096, -0.47136479, 0.5347833])
        
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