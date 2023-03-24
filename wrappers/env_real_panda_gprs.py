"""
Real robot env wrapper for Yifeng's GPRS control stack.
"""
import os
import time
import json
import sys
import numpy as np
from copy import deepcopy
from easydict import EasyDict as edict

import math

import cv2
from PIL import Image

import RobotTeleop
import RobotTeleop.utils as U
from RobotTeleop.utils import Rate, RateMeasure, Timers

# GPRS imports
from gprs.franka_interface import FrankaInterface
from gprs.camera_redis_interface import CameraRedisSubInterface
from gprs.utils import YamlConfig
from gprs import config_root

from rpl_vision_utils.utils import img_utils as ImgUtils
from PIL import Image, ImageDraw

import robomimic.envs.env_base as EB
import robomimic.utils.obs_utils as ObsUtils

def get_camera_intrinsic_matrix(CAMERA_ID):
    """
    Fill out this function to put the intrinsic matrix of your camera.

    Returns:
        K (np.array): 4x4 camera matrix
    """

    if CAMERA_ID == 0:
        K = np.array([
            [305.20733643,   0.        , 164.74075317],
            [  0.        , 304.64050293, 125.61497498],
            [  0.        ,   0.        ,   1.        ],
        ])
    else:
        K = np.array([
            [302.989999023,  0.        , 161.22071838],
            [  0.        , 302.84692383, 124.71057129],
            [  0.        ,   0.        ,   1.        ],
        ])

    K_exp = np.eye(4)
    K_exp[:3, :3] = K
    return K_exp

def get_camera_extrinsic_matrix(CAMERA_ID):
    """
    Fill out this function to put the extrinsic matrix of your camera.
    This should correspond to the camera pose in the base frame
    (camera frame to base frame transform).

    Returns:
        R (np.array): 4x4 camera extrinsic matrix
    """
    R = np.eye(4)

    if CAMERA_ID == 0:
        R[:3, :3] = np.array([
            [0.78389277,  0.35924061, -0.50641714],
            [0.60945925, -0.60101991,  0.51704399],
            [-0.11862359, -0.71394765, -0.69007767],
        ])
        R[:3, 3] = np.array([0.81514207, -0.38492553,  0.76167183])
    else:
        R[:3, :3] = np.array([
            [-0.6439447644359739, 0.6338725177213157, -0.42841658655231635],
            [0.7619834984756672, 0.4810986854862178, -0.43350340583923497],
            [-0.06867523866926933, -0.605598617981476, -0.7928013783367467]
        ])
        R[:3, 3] = np.array([0.8325469362284665, 0.2712139953995133, 0.7870214751921175])

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

def convert_points_from_camera_to_base(X1, X2, I1, I2, E1, E2):
    m1M = np.matmul(I1[:3, :3], E1[:3])
    m2M = np.matmul(I2[:3, :3], E2[:3])

    X1 = X1[::-1]
    X2 = X2[::-1]
    X1 = np.expand_dims(X1, axis=0).transpose()
    X2 = np.expand_dims(X2, axis=0).transpose()

    X = cv2.triangulatePoints(m1M, m2M, X1, X2)
    X /= X[3]
    X = X.transpose()[0][:3]

    return X

def project_points_from_base_to_camera(
    points,
    base_to_camera_matrix,
    camera_intrinsic_matrix,
    camera_height,
    camera_width,
    round_indices=True,
):
    """
    Helper function to project a batch of points in the base frame
    into camera pixels using the base to camera transformation.

    Args:
        points (np.array): 3D points in base frame to project onto camera pixel locations. Should
            be shape [..., 3].
        base_to_camera_matrix (np.array): 4x4 Tensor to go from robot base coordinates to camera
            frame coordinates (robot base pose in camera frame).
        camera_intrinsic_matrix (np.array): 4x4 Tensor for camera intrinsics (projects camera coordinates
            into pixel coordinates)
        camera_height (int): height of the camera image
        camera_width (int): width of the camera image
        round_indices (bool): if True, round pixel values to nearest integer, else return floats

    Returns:
        pixels (np.array): projected pixel indices of shape [..., 2]
        camera_points (np.array): 3D points in camera frame of shape [..., 3]
    """
    assert points.shape[-1] == 3  # last dimension must be 3D
    assert len(base_to_camera_matrix.shape) == 2
    assert base_to_camera_matrix.shape[0] == 4 and base_to_camera_matrix.shape[1] == 4
    assert len(camera_intrinsic_matrix.shape) == 2
    assert camera_intrinsic_matrix.shape[0] == 4 and camera_intrinsic_matrix.shape[1] == 4

    # convert points to homogenous coordinates -> (px, py, pz, 1)
    ones_pad = np.ones(points.shape[:-1] + (1,))
    points = np.concatenate((points, ones_pad), axis=-1)  # shape [..., 4]

    # prepare to do batch matrix multiplication of 4 x 4 matrix and 4 x 1 vectors
    mat_reshape = [1] * len(points.shape[:-1]) + [4, 4]
    base_to_camera_matrix_batch = base_to_camera_matrix.reshape(mat_reshape) # shape [..., 4, 4]
    camera_intrinsic_matrix_batch = camera_intrinsic_matrix.reshape(mat_reshape) # shape [..., 4, 4]

    # transform points from base frame to camera frame
    camera_points = np.matmul(base_to_camera_matrix_batch, points[..., None]) # shape [..., 4, 1]
    camera_points_ret = camera_points[..., :3, 0] # shape [..., 3]

    # project points onto camera plane
    pixels = np.matmul(camera_intrinsic_matrix_batch, camera_points)[..., 0]  # shape [..., 4]

    # re-scaling from homogenous coordinates to recover pixel values
    # (x, y, z) -> (x / z, y / z)
    pixels = pixels / pixels[..., 2:3]
    pixels = pixels[..., :2]
    if round_indices:
        pixels = pixels.round().astype(int)  # shape [..., 2]

    # swap first and second coordinates to get pixel indices that correspond to (height, width)
    # and also clip pixels that are out of range of the camera image
    pixels = np.concatenate(
        (
            pixels[..., 1:2].clip(0, camera_height - 1),
            pixels[..., 0:1].clip(0, camera_width - 1),
        ),
        axis=-1,
    )

    return pixels, camera_points_ret

def draw_empty_circle_on_image_at_pixel(im, pixel, circle_size=15, fill_color=(255, 0, 0), outline_width=8):

    image = Image.fromarray(im)
    draw = ImageDraw.Draw(image)
    r = circle_size
    x, y = pixel[0], pixel[1]
    y, x = x, y
    draw.ellipse((x - r, y - r, x + r, y + r), outline=fill_color, width=outline_width)
    im = np.array(image).astype(np.uint8)

    return im

def draw_empty_square_on_image_at_pixel(im, pixel, square_size=40, fill_color=(255, 0, 0), outline_width=4):
    r = square_size // 2
    x, y = pixel[0], pixel[1]
    y, x = x, y
    h, w = im.shape[0], im.shape[1]
    # print(max((y-r), 0), min((y+r), h), max((x-r), 0), min((x+r), w))
    delta_x = 0
    delta_y = 0
    if y - r < 0:
        delta_x = r-y
    if y + r > w:
        delta_x = w-(r+y)
    if x - r < 0:
        delta_y = r-x
    if x + r > h:
        delta_y = h-(r+x)

    crop_img = im[(y-r+delta_x):(y+r+delta_x), (x-r+delta_y):(x+r+delta_y)]
    # print((y-r+delta_x), (y+r+delta_x))

    image = Image.fromarray(im)
    draw = ImageDraw.Draw(image)
    # draw.ellipse((x - r, y - r, x + r, y + r), outline=fill_color, width=outline_width)
    draw.rectangle((x - r, y - r, x + r, y + r), outline=fill_color, width=outline_width, fill=fill_color)
    im = np.array(image).astype(np.uint8)
    return im, crop_img.copy()

def draw_circle_on_image_at_pixel(im, pixel, circle_size=10):
    image = Image.fromarray(im)
    draw = ImageDraw.Draw(image)
    r = circle_size
    x, y = pixel[0], pixel[1]
    y, x = x, y
    draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))
    im = np.array(image).astype(np.uint8)
    return im

def center_crop(im, t_h, t_w):
    assert(im.shape[-3] >= t_h and im.shape[-2] >= t_w)
    assert(im.shape[-1] in [1, 3])
    crop_h = int((im.shape[-3] - t_h) / 2)
    crop_w = int((im.shape[-2] - t_w) / 2)
    return im[..., crop_h:crop_h + t_h, crop_w:crop_w + t_w, :]

def interpolate_poses(pos1, rot1, pos2, rot2, num_steps=None, step_size=None, perturb=False):
    """
    Linear interpolation between two poses.

    Args:
        pos1 (np.array): np array of shape (3,) for first position
        rot1 (np.array): np array of shape (3, 3) for first rotation
        pos2 (np.array): np array of shape (3,) for second position
        rot2 (np.array): np array of shape (3, 3) for second rotation
        num_steps (int): if provided, specifies the number of desired interpolated points (not excluding
            the start and end points). Passing 0 corresponds to no interpolation, and passing None
            means that @step_size must be provided to determine the number of interpolated points.
        step_size (float): if provided, will be used to infer the number of steps, by taking the norm
            of the delta position vector, and dividing it by the step size
        perturb (bool): if True, randomly move all the interpolated position points in a uniform, non-overlapping grid.

    Returns:
        pos_steps (np.array): array of shape (N + 2, 3) corresponding to the interpolated position path, where N is @num_steps
        rot_steps (np.array): array of shape (N + 2, 3, 3) corresponding to the interpolated rotation path, where N is @num_steps
        num_steps (int): the number of interpolated points (N) in the path
    """
    assert step_size is None or num_steps is None

    if num_steps == 0:
        # skip interpolation
        return np.concatenate([pos1[None], pos2[None]], axis=0), np.concatenate([rot1[None], rot2[None]], axis=0), num_steps

    delta_pos = pos2 - pos1
    if num_steps is None:
        assert np.linalg.norm(delta_pos) > 0
        num_steps = math.ceil(np.linalg.norm(delta_pos) / step_size)

    num_steps += 1  # include starting pose
    assert num_steps >= 2

    # linear interpolation of positions
    pos_step_size = delta_pos / num_steps
    grid = np.arange(num_steps).astype(np.float64)
    if perturb:
        # move the interpolation grid points by up to a half-size forward or backward
        perturbations = np.random.uniform(
            low=-0.5,
            high=0.5,
            size=(num_steps - 2,),
        )
        grid[1:-1] += perturbations
    pos_steps = np.array([pos1 + grid[i] * pos_step_size for i in range(num_steps)])

    # add in endpoint
    pos_steps = np.concatenate([pos_steps, pos2[None]], axis=0)

    # interpolate the rotations too
    # rot_steps = interpolate_rotations(R1=rot1, R2=rot2, num_steps=num_steps, axis_angle=True)
    rot_steps = interpolate_rotations(R1=rot1, R2=rot2, num_steps=num_steps)

    return pos_steps, rot_steps, num_steps - 1

def interpolate_tranlations(T1, T2, num_steps, perturb=False):
    delta_pos = T2 - T1
    pos_step_size = delta_pos / num_steps
    grid = np.arange(num_steps).astype(np.float64)
    if perturb:
        # move the interpolation grid points by up to a half-size forward or backward
        perturbations = np.random.uniform(
            low=-0.5,
            high=0.5,
            size=(num_steps - 2,),
        )
        grid[1:-1] += perturbations
    pos_steps = np.array([T1 + grid[i] * pos_step_size for i in range(num_steps)])

    # add in endpoint
    pos_steps = np.concatenate([pos_steps, T2[None]], axis=0)

    return pos_steps[1:]

def interpolate_rotations(R1, R2, num_steps):
    """
    Interpolate between 2 rotation matrices.
    """
    q1 = U.mat2quat(R1)
    q2 = U.mat2quat(R2)
    rot_steps = np.array([U.quat2mat(quat_slerp(q1, q2, tau=(float(i) / num_steps))) for i in range(num_steps)])

    # add in endpoint
    rot_steps = np.concatenate([rot_steps, R2[None]], axis=0)

    return rot_steps[1:]

def quat_slerp(q1, q2, tau):
    """
    Adapted from robosuite.
    """
    if tau == 0.0:
        return q1
    elif tau == 1.0:
        return q2
    d = np.dot(q1, q2)
    if abs(abs(d) - 1.0) < np.finfo(float).eps * 4.:
        return q1
    if d < 0.0:
        # invert rotation
        d = -d
        q2 *= -1.0
    angle = math.acos(np.clip(d, -1, 1))
    if abs(angle) < np.finfo(float).eps * 4.:
        return q1
    isin = 1.0 / math.sin(angle)
    q1 = q1 * math.sin((1.0 - tau) * angle) * isin
    q2 = q2 * math.sin(tau * angle) * isin
    q1 = q1 + q2
    return q1

class EnvRealPandaGPRS(EB.EnvBase):
    """Wrapper class for real panda environment"""
    def __init__(
        self,
        env_name,
        render=False,
        render_offscreen=False,
        use_image_obs=True,
        postprocess_visual_obs=True,
        control_freq=20.,
        camera_names_to_sizes=None,
        general_cfg_file=None,
        controller_type=None,
        controller_cfg_file=None,
        controller_cfg_dict=None,
        use_depth_obs=True,
        # additional GPRS-specific args
        state_freq=100.,
        control_timeout=1.0,
        has_gripper=True,
        use_visualizer=False,
        debug=False,
    ):
        """
        Args:
            env_name (str): name of environment.

            render (bool): ignored - on-screen rendering is not supported

            render_offscreen (bool): ignored - image observations are supplied by default

            use_image_obs (bool): ignored - image observations are used by default.

            postprocess_visual_obs (bool): if True, postprocess image observations
                to prepare for learning. This should only be False when extracting observations
                for saving to a dataset (to save space on RGB images for example).

            control_freq (int): real-world control frequency to try and enforce through rate-limiting

            camera_names_to_sizes (dict):  dictionary that maps camera names to tuple of image height and width
                to return
        """
        self._env_name = env_name
        self.postprocess_visual_obs = postprocess_visual_obs
        self.control_freq = control_freq
        self.general_cfg_file = general_cfg_file
        self.controller_type = controller_type
        self.controller_cfg_file = controller_cfg_file
        self.controller_cfg_dict = deepcopy(controller_cfg_dict) if controller_cfg_dict is not None else None
        if self.controller_cfg_dict is not None:
            # control code expects easydict
            self.controller_cfg = edict(self.controller_cfg_dict)
        else:
            assert controller_cfg_file is not None
            self.controller_cfg = YamlConfig(os.path.join(config_root, controller_cfg_file)).as_easydict()
        self.use_depth_obs = use_depth_obs

        # to enforce control rate
        self.rate = Rate(control_freq)
        self.rate_measure = RateMeasure(name="robot", freq_threshold=round(0.95 * control_freq))
        self.timers = Timers(history=100, disable_on_creation=False)

        camera_names_to_sizes = deepcopy(camera_names_to_sizes)
        if camera_names_to_sizes is None:
            self.camera_names_to_sizes = {}
        else:
            self.camera_names_to_sizes = camera_names_to_sizes

        self.camera_to_base_matrix_1 = get_camera_extrinsic_matrix(0)
        self.camera_intrinsic_matrix_1 = get_camera_intrinsic_matrix(0)
        self.base_to_camera_matrix_1 = pose_inv(self.camera_to_base_matrix_1)

        self.camera_to_base_matrix_2 = get_camera_extrinsic_matrix(1)
        self.camera_intrinsic_matrix_2 = get_camera_intrinsic_matrix(1)
        self.base_to_camera_matrix_2 = pose_inv(self.camera_to_base_matrix_2)

        # save kwargs for serialization
        kwargs = dict(
            camera_names_to_sizes=camera_names_to_sizes,
            general_cfg_file=general_cfg_file,
            control_freq=control_freq,
            controller_type=controller_type,
            controller_cfg_file=controller_cfg_file,
            controller_cfg_dict=controller_cfg_dict,
            use_depth_obs=use_depth_obs,
            state_freq=state_freq,
            control_timeout=control_timeout,
            has_gripper=has_gripper,
            use_visualizer=use_visualizer,
            debug=debug,
        )
        self._init_kwargs = deepcopy(kwargs)

        # connect to robot
        self.robot_interface = FrankaInterface(
            general_cfg_file=os.path.join(config_root, general_cfg_file),
            control_freq=control_freq,
            state_freq=state_freq,
            control_timeout=control_timeout,
            has_gripper=has_gripper,
            use_visualizer=use_visualizer,
            debug=debug,
        )

        # TODO: clean up camera ID definition later

        # start camera interfaces
        camera_ids = list(range(len(self.camera_names_to_sizes)))
        self.cr_interfaces = {}
        for c_id, c_name in enumerate(self.camera_names_to_sizes):
            cr_interface = CameraRedisSubInterface(camera_id=c_id, use_depth=self.use_depth_obs)
            cr_interface.start()
            self.cr_interfaces[c_name] = cr_interface

        # IMPORTANT: initialize JIT functions that may need to compile
        self._compile_jit_functions()

    def _compile_jit_functions(self):
        """
        Helper function to incur the cost of compiling jit functions used by this class upfront.

        NOTE: this function looks strange because we apparently need to make it look like the env.step function
              for it to compile properly, otherwise we will have a heavy delay on the first env.step call...

        TODO: figure out why this needs to look like the step function code below...
        """

        # current robot state to use as reference
        # ee_pos, ee_quat = self.robot_interface.ee_pose
        ee_mat = U.quat2mat(np.array([0., 0., 0., 1.]))
        ee_quat_hat = U.mat2quat(ee_mat)

        # convert delta axis-angle to delta rotation matrix, and from there, to absolute target rotation
        drot = np.array([0., 0., 0.05])
        angle = np.linalg.norm(drot)
        if U.isclose(angle, 0.):
            drot_quat = np.array([0., 0., 0., 1.])
        else:
            axis = drot / angle
            drot_quat = U.axisangle2quat(axis, angle)

        # get target rotation
        drot_mat = U.quat2mat(drot_quat)
        target_rot_mat = (drot_mat.T).dot(ee_mat)
        target_rot_quat = U.mat2quat(target_rot_mat)


    def interpolate_poses(self, target_pos, target_rot=None, num_steps=None, step_size=None):
        assert num_steps is None or step_size is None
        last_robot_state = self.robot_interface._state_buffer[-1]
        last_gripper_state = self.robot_interface._gripper_state_buffer[-1]
        ee_pose = np.array(last_robot_state.O_T_EE).reshape((4, 4)).T
        if np.count_nonzero(ee_pose.reshape(-1)) == 0:
            raise Exception("GOT ZERO EE POSE")
        start_pos = ee_pose[:3, 3]
        start_rot = ee_pose[:3, :3]
        target_rot = U.quat2mat(target_rot)

        if num_steps is None:
            # calculate number of steps in terms of translation
            delta_pos = target_pos - start_pos
            if np.linalg.norm(delta_pos) > 0:
                num_steps_pos = math.ceil(np.linalg.norm(delta_pos) / step_size)
            else:
                num_steps_pos = 1
            # calculate number of steps in terms of rotation
            rot_angle = np.arccos((np.trace(np.dot(start_rot, np.transpose(target_rot))) - 1) / 2)
            if rot_angle >= np.radians(2):
                num_steps_rot = int(np.ceil(rot_angle / np.radians(2)))  # 2 degree for one step
            else:
                num_steps_rot = 1
            # print(f'num_steps_pos: {num_steps_pos}')
            # print(f'num_steps_rot: {num_steps_rot}')
            num_steps = max(num_steps_rot, num_steps_pos)

        tran_inter = interpolate_tranlations(start_pos, target_pos, num_steps)
        ori_inter = interpolate_rotations(start_rot, target_rot, num_steps)

        return tran_inter, ori_inter

    def poses_to_action(self, target_pos, target_rot=None, max_dpos=0.05, max_drot=0.2):
        """
        Takes a starting eef pose and target controller pose and returns a normalized action that
        corresponds to the desired controller target.
        """
        last_robot_state = self.robot_interface._state_buffer[-1]
        last_gripper_state = self.robot_interface._gripper_state_buffer[-1]
        ee_pose = np.array(last_robot_state.O_T_EE).reshape((4, 4)).T
        if np.count_nonzero(ee_pose.reshape(-1)) == 0:
            raise Exception("GOT ZERO EE POSE")
        start_pos = ee_pose[:3, 3]
        start_rot = ee_pose[:3, :3]

        delta_position = target_pos - start_pos
        delta_position = np.clip(delta_position / max_dpos, -1., 1.)
        if target_rot is None:
            return delta_position

        target_rot = U.quat2mat(target_rot)

        # use the OSC controller's convention for delta rotation
        delta_rot_mat = target_rot.dot(start_rot.T)
        delta_quat = U.mat2quat(delta_rot_mat)
        delta_rotation = U.quat2euler(delta_quat)
        delta_rotation = np.clip(delta_rotation / max_drot, -1., 1.)
        return np.concatenate([delta_position, delta_rotation])

    def step(self, action, waypoints=None, need_obs=True):
        """
        Step in the environment with an action.

        Args:
            action (np.array): action to take, should be in [-1, 1]
            need_obs (bool): if False, don't return the observation, because this
                can involve copying image data around. This allows for more
                flexibility on when observations are retrieved.

        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """

        action = np.array(action)

        # assert len(action.shape) == 1 and action.shape[0] == 7, "action has incorrect dimensions"
        # assert np.min(action) >= -1. and np.max(action) <= 1., "incorrect action bounds"

        # meaure rate-limiting
        # self.rate.sleep()
        self.rate_measure.measure()

        self.timers.tic("real_panda_step")

        if action.shape[0] == 9: #use_pose_to_action
            step_size = action[-1]
            if step_size >= 0.03:
                print(f'*** [env_real_panda_gprs.py] requested step size: {step_size}, clipping to 0.03 ***')
                step_size = 0.03
            action = action[:-1]
            action = np.clip(action, -1, 1)

            tran_inter, ori_inter = self.interpolate_poses(action[:3], action[3:7], step_size=step_size)

            for i in range(len(tran_inter)):
                trans = tran_inter[i]
                ori = U.mat2quat(ori_inter[i])

                for _ in range(5):
                    new_action = self.poses_to_action(trans, ori)
                    new_action = np.concatenate((new_action, action[-1:]))

                    self.robot_interface.control(
                        control_type=self.controller_type,
                        action=new_action,
                        controller_cfg=self.controller_cfg,
                    )

                    # print(trans, new_action, i, num_steps)
                    # time.sleep(0.01)

        else:
            action = np.clip(action, -1, 1)
            self.robot_interface.control(
                control_type=self.controller_type,
                action=action,
                controller_cfg=self.controller_cfg,
            )
        # time.sleep(0.01)

        # get observation
        obs = None
        if need_obs:
            obs = self.get_observation()
        r = self.get_reward()
        done = self.is_done()

        self.timers.toc("real_panda_step")

        return obs, r, done, {}

    def reset(self, skip_clear_buffer=False):
        """
        Reset environment.

        Returns:
            observation (dict): initial observation dictionary.
        """

        # self.robot_interface.close()
        # del self.robot_interface
        # self.robot_interface = FrankaInterface(
        #     general_cfg_file=os.path.join(config_root, self._init_kwargs['general_cfg_file']),
        #     control_freq=self._init_kwargs['control_freq'],
        #     state_freq=self._init_kwargs['state_freq'],
        #     control_timeout=self._init_kwargs['control_timeout'],
        #     has_gripper=self._init_kwargs['has_gripper'],
        #     use_visualizer=self._init_kwargs['use_visualizer'],
        #     debug=self._init_kwargs['debug'],
        # )
        if not skip_clear_buffer:
            self.robot_interface.clear_buffer()

        print("restarting the robot interface")

        # Code below based on https://github.com/UT-Austin-RPL/robot_infra/blob/master/gprs/examples/reset_robot_joints.py

        # Golden resetting joints
        # reset_joint_positions = [0.0, -0.49826458111314524, -0.01990020486871322, -2.4732269941140346, -0.01307073642274261, 2.00396583422025, 0.8480939705504309]
        # reset_joint_positions = [1.5, -1.19826458111314524, -0.01990020486871322, -1.7732269941140346, -0.01307073642274261, 2.40396583422025, 0.8480939705504309]

        # (added by wenlong) higher up resetting joints such that robots don't cause occlusions
        # reset_joint_positions = [0.0, -0.4343765842287163, -0.01928744912408946, -1.581905981767789, -0.01094451389755545, 1.1993245112398392, 0.8474906211462537]

        # (added by wenlong) higher up resetting joints such that robots don't cause occlusions (slightly lower than last one)
        reset_joint_positions = [-0.0, -0.4570295298960899, -0.009145100616049347, -2.1223430879492504, -0.013212399939695994, 1.730008026409128, 0.8501183513913816]

        # This is for varying initialization of joints a little bit to
        # increase data variation.
        # reset_joint_positions = [e + np.clip(np.random.randn() * 0.005, -0.005, 0.005) for e in reset_joint_positions]
        action = reset_joint_positions + [-1.]

        # temp robot interface to use for joint position control
        # tmp_robot_interface = FrankaInterface(os.path.join(config_root, self.general_cfg_file), use_visualizer=False)
        # tmp_controller_cfg = YamlConfig(os.path.join(config_root, self.controller_cfg_file)).as_easydict()
        tmp_controller_cfg = deepcopy(self.controller_cfg)

        while True:
            if len(self.robot_interface._state_buffer) > 0:
                # print(self.robot_interface._state_buffer[-1].q)
                # print(reset_joint_positions)
                # print(np.max(np.abs(np.array(self.robot_interface._state_buffer[-1].q) - np.array(reset_joint_positions))))
                # print("-----------------------")

                # if np.max(np.abs(np.array(self.robot_interface._state_buffer[-1].q) - np.array(reset_joint_positions))) < 1e-3:
                if np.max(np.abs(np.array(self.robot_interface._state_buffer[-1].q) - np.array(reset_joint_positions))) < 1e-2:
                    break

            self.robot_interface.control(
                control_type="JOINT_POSITION",
                action=action,
                controller_cfg=tmp_controller_cfg,
            )

        # tmp_robot_interface.close()

        # We added this sleep here to give the C++ controller time to reset from joint control mode to no control mode
        # to prevent some issues.
        time.sleep(1.0)
        print("RESET DONE")

        return self.get_observation()

    def reset_to_default_pose(self, gripper_action=None):
        """
        Reset to default pose
        """
        # Golden resetting joints
        # reset_joint_positions = [0.0, -0.49826458111314524, -0.01990020486871322, -2.4732269941140346, -0.01307073642274261, 2.00396583422025, 0.8480939705504309]
        # (added by wenlong) higher up resetting joints such that robots don't cause occlusions
        # reset_joint_positions = [0.0, -0.4343765842287163, -0.01928744912408946, -1.581905981767789, -0.01094451389755545, 1.1993245112398392, 0.8474906211462537]
        # (added by wenlong) higher up resetting joints such that robots don't cause occlusions (slightly lower than last one)
        reset_joint_positions = [-0.0, -0.4570295298960899, -0.009145100616049347, -2.1223430879492504, -0.013212399939695994, 1.730008026409128, 0.8501183513913816]

        if gripper_action is None:
            action = reset_joint_positions + [-1.]
        else:
            action = reset_joint_positions + [gripper_action]

        action = np.array(action)
        print("[env_real_panda_gprs.py] resetting to default pose: {}".format(action))

        tmp_controller_cfg = deepcopy(self.controller_cfg)

        while True:
            if len(self.robot_interface._state_buffer) > 0:
                if np.max(np.abs(np.array(self.robot_interface._state_buffer[-1].q) - np.array(reset_joint_positions))) < 1e-2:
                    break
            self.robot_interface.control(
                control_type="JOINT_POSITION",
                action=action,
                controller_cfg=tmp_controller_cfg,
            )

        time.sleep(1.0)

        return self.get_observation()

    def reset_to(self, state):
        """
        Reset to a specific state. On real robot, we visualize the start image,
        and a human should manually reset the scene.

        Reset to a specific simulator state.

        Args:
            state (dict): initial state that contains:
                - image (np.ndarray): initial workspace image

        Returns:
            None
        """
        assert "front_image_1" in state

        ref_img = cv2.cvtColor(state["front_image_1"], cv2.COLOR_RGB2BGR)

        print("\n" + "*" * 50)
        print("Reset environment to image shown in left pane")
        print("Press 'c' when ready to continue.")
        print("*" * 50 + "\n")
        while(True):
            # read current image
            cur_img = self._get_image(camera_name="front_image_1")
            if self.use_depth_obs:
                cur_img = cur_img[0]

            # concatenate frames to display
            img = np.concatenate([ref_img, cur_img], axis=1)

            # display frame
            cv2.imshow('initial state alignment window', img)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                cv2.destroyAllWindows()
                break

    def render(self, mode="human", height=None, width=None, camera_name=None, **kwargs):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
        """
        if mode =="human":
            raise Exception("on-screen rendering not supported currently")
        if mode == "rgb_array":
            assert (height is None) and (width is None), "cannot resize images"
            assert camera_name in self.camera_names_to_sizes, "invalid camera name"
            return self._get_image(camera_name=camera_name)[..., ::-1]
        else:
            raise NotImplementedError("mode={} is not implemented".format(mode))

    def get_observation(self, obs=None):
        """
        Get current environment observation dictionary.

        Args:
            ob (np.array): current observation dictionary.
        """
        self.timers.tic("get_observation")
        observation = {}
        last_robot_state = self.robot_interface._state_buffer[-1]
        last_gripper_state = self.robot_interface._gripper_state_buffer[-1]
        ee_pose = np.array(last_robot_state.O_T_EE).reshape((4, 4)).T
        if np.count_nonzero(ee_pose.reshape(-1)) == 0:
            raise Exception("GOT ZERO EE POSE")
        ee_pos = ee_pose[:3, 3]
        ee_quat = U.mat2quat(ee_pose[:3, :3])
        observation["ee_pose"] = np.concatenate([ee_pos, ee_quat])
        observation["joint_positions"] = np.array(last_robot_state.q)
        observation["joint_velocities"] = np.array(last_robot_state.dq)
        observation["gripper_position"] = np.array(last_gripper_state.width)

        # hand_loc_1, _ = project_points_from_base_to_camera(
        #     points=ee_pos,
        #     base_to_camera_matrix=self.base_to_camera_matrix_1,
        #     camera_intrinsic_matrix=self.camera_intrinsic_matrix_1,
        #     camera_height=240,
        #     camera_width=320,
        #     round_indices=False,
        # )
        #
        # hand_loc_2, _ = project_points_from_base_to_camera(
        #     points=ee_pos,
        #     base_to_camera_matrix=self.base_to_camera_matrix_2,
        #     camera_intrinsic_matrix=self.camera_intrinsic_matrix_2,
        #     camera_height=240,
        #     camera_width=320,
        #     round_indices=False,
        # )

        # estimated_ee_pos = convert_points_from_camera_to_base(hand_loc_1, hand_loc_2,
        #                                self.camera_intrinsic_matrix_1, self.camera_intrinsic_matrix_2,
        #                                self.base_to_camera_matrix_1, self.base_to_camera_matrix_2)
        #
        # print("EE pose", ee_pos, estimated_ee_pos)

        # normalize hand location to 0~1

        # hand_loc_1[0] = hand_loc_1[0] / 240.0 * 120.0
        # hand_loc_1[1] = hand_loc_1[1] / 320.0 * 160.0
        # hand_loc_1[1] = np.clip(hand_loc_1[1], 20.0, 140.0) - 20.0
        # hand_loc_1 = hand_loc_1 / 120.0
        #
        # hand_loc_2[0] = hand_loc_2[0] / 240.0 * 120.0
        # hand_loc_2[1] = hand_loc_2[1] / 320.0 * 160.0
        # hand_loc_2[1] = np.clip(hand_loc_2[1], 20.0, 140.0) - 20.0
        # hand_loc_2 = hand_loc_2 / 120.0
        #
        # observation["hand_loc"] = np.concatenate([hand_loc_1, hand_loc_2], axis=-1)
        # observation["hand_loc_3d"] = observation["ee_pose"][..., :3]

        # observation["gripper_velocity"] = self.robot_interface.gripper_velocity
        for cam_name in self.camera_names_to_sizes:
            im = self._get_image(camera_name=cam_name)
            if self.use_depth_obs:
                im, points = im
                observation[cam_name + "_points"] = points / 1000.
                # observation[cam_name + "_depth"] = depth_im
                # print(f'depth_im.shape: {depth_im.shape}')
                # im, depth_im, depth_im_unaligned = im
                # observation[cam_name + "_depth"] = depth_im
                # observation[cam_name + "_unaligned_depth"] = depth_im_unaligned
            im = im[..., ::-1]
            # if cam_name == 'wrist_image':
            #     observation['wrist_image_40'] = cv2.resize(im, (40, 40))
            #     observation['wrist_image_40'] = ObsUtils.process_image(observation['wrist_image_40'])
            # if self.postprocess_visual_obs:
            #     # if cam_name == 'front_image_1':
            #     #     im, crop_im_1 = draw_empty_square_on_image_at_pixel(im, (hand_loc_1 * 120).round().astype(int))
            #     #     observation['front_image_crop_1'] = ObsUtils.process_image(crop_im_1)
            #     # if cam_name == 'front_image_2':
            #     #     im, crop_im_2 = draw_empty_square_on_image_at_pixel(im, (hand_loc_2 * 120).round().astype(int))
            #     #     observation['front_image_crop_2'] = ObsUtils.process_image(crop_im_2)
            #
            #     im = ObsUtils.process_image(im)
            #     assert not self.use_depth_obs, "TODO: support for process depth"
            observation[cam_name] = im

        self.timers.toc("get_observation")
        return observation

    def _get_image(self, camera_name):
        """
        Get image from camera interface
        """

        # get image
        imgs = self.cr_interfaces[camera_name].get_img()
        im = imgs["color"]

        # resize image
        # im_size = self.camera_names_to_sizes[camera_name]
        # if im_size is not None:
        #     im = Image.fromarray(im).resize((im_size[1], im_size[0]), Image.BILINEAR)
        im = np.array(im).astype(np.uint8)

        # center crop image
        # crop_size = min(im.shape[:2])
        # im = center_crop(im, crop_size, crop_size)

        if self.use_depth_obs:
            # depth_im = imgs["depth"]
            # if im_size is not None:
                # depth_im = Image.fromarray(depth_im).resize((im_size[1], im_size[0]), Image.BILINEAR)
                # depth_im = Image.fromarray(depth_im).resize((im_size[1], im_size[0]))
            # note: depth images are uint16, with default scale 0.001m
            # depth_im = np.array(depth_im).astype(np.uint16)
            # if len(depth_im.shape) < 3:
            #     depth_im = depth_im[..., None] # add channel dimension
            # depth_im = center_crop(depth_im, crop_size, crop_size)
            # check if point cloud is provided
            # if "points" in imgs:
            points = imgs["points"]
            return im, points
            # else:
            #     return im, depth_im
            # depth_images = []
            # for k in ["depth", "unaligned_depth"]:
            #     depth_im = imgs[k]
            #     if im_size is not None:
            #         # depth_im = Image.fromarray(depth_im).resize((im_size[1], im_size[0]), Image.BILINEAR)
            #         depth_im = Image.fromarray(depth_im).resize((im_size[1], im_size[0]))
            #     # note: depth images are uint16, with default scale 0.001m
            #     depth_im = np.array(depth_im).astype(np.uint16)
            #     if len(depth_im.shape) < 3:
            #         depth_im = depth_im[..., None]  # add channel dimension
            #     depth_im = center_crop(depth_im, crop_size, crop_size)
            #     depth_images.append(depth_im)
            # return im, depth_images[0], depth_images[1]
        return im

    def get_state(self):
        """
        Get current environment simulator state as a dictionary. Should be compatible with @reset_to.
        """
        return dict(states=np.zeros(1))
        # raise Exception("Real robot has no simulation state.")

    def get_reward(self):
        """
        Get current reward.
        """
        return 0.

    def get_goal(self):
        """
        Get goal observation. Not all environments support this.
        """
        raise NotImplementedError

    def set_goal(self, **kwargs):
        """
        Set goal observation with external specification. Not all environments support this.
        """
        raise NotImplementedError

    def is_done(self):
        """
        Check if the task is done (not necessarily successful).
        """
        return False

    def is_success(self):
        """
        Check if the task condition(s) is reached. Should return a dictionary
        { str: bool } with at least a "task" key for the overall task success,
        and additional optional keys corresponding to other task criteria.
        """

        # real robot environments don't usually have a success check - this must be done manually
        return { "task" : False }

    @property
    def action_dimension(self):
        """
        Returns dimension of actions (int).
        """
        return 7

    @property
    def name(self):
        """
        Returns name of environment name (str).
        """
        # return self._env_name

        # for real robot. ensure class name is stored in env meta (as env name) for use with any external
        # class registries
        return self.__class__.__name__

    @property
    def type(self):
        """
        Returns environment type (int) for this kind of environment.
        This helps identify this env class.
        """
        return EB.EnvType.GPRS_REAL_TYPE

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(env_name=self.name, type=self.type, env_kwargs=deepcopy(self._init_kwargs))

    @classmethod
    def create_for_data_processing(cls, env_name, camera_names, camera_height, camera_width, reward_shaping, **kwargs):
        """
        Create environment for processing datasets, which includes extracting
        observations, labeling dense / sparse rewards, and annotating dones in
        transitions. For gym environments, input arguments (other than @env_name)
        are ignored, since environments are mostly pre-configured.

        Args:
            env_name (str): name of gym environment to create

        Returns:
            env (EnvRealPanda instance)
        """

        # initialize obs utils so it knows which modalities are image modalities
        assert "camera_names_to_sizes" in kwargs
        image_modalities = list(kwargs["camera_names_to_sizes"].keys())
        obs_modality_specs = {
            "obs": {
                "low_dim": [], # technically unused, so we don't have to specify all of them
                "image": image_modalities,
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

        # note that @postprocess_visual_obs is False since this env's images will be written to a dataset
        return cls(
            env_name=env_name,
            render=False,
            render_offscreen=True,
            use_image_obs=True,
            postprocess_visual_obs=False,
            **kwargs,
        )

    @property
    def rollout_exceptions(self):
        """
        Return tuple of exceptions to except when doing rollouts. This is useful to ensure
        that the entire training run doesn't crash because of a bad policy that causes unstable
        simulation computations.
        """
        return ()

    @property
    def base_env(self):
        """
        Grabs base simulation environment.
        """
        # we don't wrap any env
        return self

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.name + "\n" + json.dumps(self._init_kwargs, sort_keys=True, indent=4)

    def close(self):
        """
        Clean up env
        """
        for c_name in self.cr_interfaces:
            self.cr_interfaces[c_name].stop()
        self.robot_interface.close()
