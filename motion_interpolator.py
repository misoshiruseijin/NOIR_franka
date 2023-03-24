"""
motion interpolation script for real franka using deoxys
based on functions written by Chen
"""

import numpy as np
import math
import sys
import pdb
sys.path.insert(1, "/home/eeg/deoxys_control/deoxys")
from deoxys.utils import transform_utils as U
from deoxys.franka_interface.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig

class RealFrankaEnv:

    def __init__(
        self,
        interface_cfg="config/charmander.yml",
        controller_cfg="config/osc-yaw-controller.yml",
        controller_type="OSC_POSE",
    ):
        self.robot_interface = FrankaInterface(interface_cfg)
        controller_cfg = YamlConfig(controller_cfg).as_easydict()
        self.controller_type = controller_type

    def interpolate_poses(self, target_pos, target_rot=None, num_steps=10):
        last_robot_state = self.robot_interface._state_buffer[-1]
        last_gripper_state = self.robot_interface._gripper_state_buffer[-1]
        ee_pose = np.array(last_robot_state.O_T_EE).reshape((4, 4)).T
        if np.count_nonzero(ee_pose.reshape(-1)) == 0:
            raise Exception("GOT ZERO EE POSE")
        start_pos = ee_pose[:3, 3]
        start_rot = ee_pose[:3, :3]

        tran_inter = interpolate_translations(start_pos, target_pos, num_steps)
        target_rot = U.quat2mat(target_rot)
        ori_inter = interpolate_rotations(start_rot, target_rot, num_steps)

        return tran_inter, ori_inter

    def poses_to_action(self, target_pos, target_rot=None, max_dpos=0.08, max_drot=0.5):
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

def interpolate_translations(T1, T2, num_steps, perturb=False):
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


"""
Calling the controller on real robot
"""
import time
env = RealFrankaEnv()
num_steps = 5

pdb.set_trace()

# tran_inter, ori_inter = env.interpolate_poses(action[:3], action[3:7], num_steps=num_steps)
# for i in range(num_steps):
#     trans = tran_inter[i]
#     ori = U.mat2quat(ori_inter[i])
#     new_action = env.poses_to_action(trans, ori)
#     new_action = np.concatenate((new_action, action[-1:]))

    # env.robot_interface.control(
    #     control_type=env.controller_type,
    #     action=new_action,
    #     controller_cfg=env.controller_cfg,
    # )
#     time.sleep(0.05)