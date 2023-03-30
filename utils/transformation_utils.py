"""
Some utility functions. Adopted from RobotTeleop library

NOTE: convention for quaternions is (x, y, z, w).
"""
import sys
import math
# import time
# import json
# import h5py
# import hashlib
import numpy as np
# import threading
from collections import deque, OrderedDict
from contextlib import contextmanager
import pybullet as p

pi = np.pi
EPS = np.finfo(float).eps * 4.
np.set_printoptions(precision=5)

_quaternion_resolution = 10 * np.finfo(float).resolution

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


# numba-related variables and decorators

# use numba on python 3 by default and not on python 2, since modern versions
# of numba don't support python 2
ENABLE_NUMBA = (sys.version_info.major == 3)


# whether to cache numba compilation (set to True if these function implementations will not change)
CACHE_NUMBA = False


# conditional numba import
if ENABLE_NUMBA:
    import numba


# numba decorator
def jit_decorator(func):
    if ENABLE_NUMBA:
        return numba.jit(nopython=True, cache=CACHE_NUMBA)(func)
    return func


def convert_quat(q, to='xyzw'):
    """
    Converts quaternion from one convention to another. 
    The convention to convert TO is specified as an optional argument. 
    If to == 'xyzw', then the input is in 'wxyz' format, and vice-versa.

    :param q: a 4-dim numpy array corresponding to a quaternion
    :param to: a string, either 'xyzw' or 'wxyz', determining 
               which convention to convert to.
    """
    if to == 'xyzw':
        return q[[1, 2, 3, 0]]
    elif to == 'wxyz':
        return q[[3, 0, 1, 2]]
    else:
        raise Exception("convert_quat: choose a valid `to` argument (xyzw or wxyz)")


def vec(values):
    """
    Convert value tuple into a vector
    :param values: a tuple of numbers
    :return: vector of given values
    """
    return np.array(values, dtype=np.float32)


def mat4(array):
    """
    Convert an array to 4x4 matrix
    :param array: the array in form of vec, list, or tuple
    :return: 4x4 numpy matrix
    """
    return np.array(array, dtype=np.float32).reshape((4, 4))


def mat2pose(hmat):
    """
    Convert a homogeneous 4x4 matrix into pose
    :param hmat: a 4x4 homogeneous matrix
    :return: (pos, orn) tuple where pos is 
    vec3 float in cartesian, orn is vec4 float quaternion
    """
    pos = hmat[:3, 3]
    orn = mat2quat(hmat[:3, :3])
    return pos, orn


@jit_decorator
def mat2quat(rmat, precise=False):
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat: 3x3 rotation matrix
        precise: If isprecise is True, the input matrix is assumed to be a precise
             rotation matrix and a faster algorithm is used.

    Returns:
        vec4 float quaternion angles
    """
    M = np.asarray(rmat).astype(np.float32)[:3, :3]
    if precise:
        # This code uses a modification of the algorithm described in:
        # https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
        # which is itself based on the method described here:
        # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        # Altered to work with the column vector convention instead of row vectors
        m = M.conj().transpose() # This method assumes row-vector and postmultiplication of that vector
        if m[2, 2] < 0:
            if m[0, 0] > m[1, 1]:
                t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
                q = [m[1, 2]-m[2, 1],  t,  m[0, 1]+m[1, 0],  m[2, 0]+m[0, 2]]
            else:
                t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
                q = [m[2, 0]-m[0, 2],  m[0, 1]+m[1, 0],  t,  m[1, 2]+m[2, 1]]
        else:
            if m[0, 0] < -m[1, 1]:
                t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
                q = [m[0, 1]-m[1, 0],  m[2, 0]+m[0, 2],  m[1, 2]+m[2, 1],  t]
            else:
                t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
                q = [t,  m[1, 2]-m[2, 1],  m[2, 0]-m[0, 2],  m[0, 1]-m[1, 0]]
        q = np.array(q)
        q *= 0.5 / np.sqrt(t)
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array(
            [
                [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
                [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
                [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        K /= 3.0
        # quaternion is Eigen vector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        inds = np.array([3, 0, 1, 2])
        q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]


def mat2euler(rmat, axes='sxyz'):
    """
    Convert given rotation matrix to euler angles in radian.
    :param rmat: 3x3 rotation matrix
    :param axes: One of 24 axis sequences as string or encoded tuple
    :return: converted euler angles in radian vec3 float
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.array(rmat, dtype=np.float32, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return vec((ax, ay, az)) 


def euler2mat(euler, axes='sxyz'):
    """Return homogeneous rotation matrix from Euler angles and axis sequence.
    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple
    >>> R = euler_matrix(1, 2, 3, 'syxz')
    >>> numpy.allclose(numpy.sum(R[0]), -1.34786452)
    True
    >>> R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
    >>> numpy.allclose(numpy.sum(R[0]), -0.383436184)
    True
    >>> ai, aj, ak = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)
    >>> for axes in _TUPLE2AXES.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)
    """
    ai, aj, ak = euler[0], euler[1], euler[2]

    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = np.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj*si
        M[i, k] = sj*ci
        M[j, i] = sj*sk
        M[j, j] = -cj*ss+cc
        M[j, k] = -cj*cs-sc
        M[k, i] = -sj*ck
        M[k, j] = cj*sc+cs
        M[k, k] = cj*cc-ss
    else:
        M[i, i] = cj*ck
        M[i, j] = sj*sc-cs
        M[i, k] = sj*cc+ss
        M[j, i] = cj*sk
        M[j, j] = sj*ss+cc
        M[j, k] = sj*cs-sc
        M[k, i] = -sj
        M[k, j] = cj*si
        M[k, k] = cj*ci
    return M[:3, :3]


def pose2mat(pose):
    """
    Convert pose to homogeneous matrix
    :param pose: a (pos, orn) tuple where
    pos is vec3 float cartesian, and
    orn is vec4 float quaternion.
    :return:
    """
    homo_pose_mat = np.zeros((4, 4), dtype=np.float32)
    homo_pose_mat[:3, :3] = quat2mat(pose[1])
    homo_pose_mat[:3, 3] = np.array(pose[0], dtype=np.float32)
    homo_pose_mat[3, 3] = 1.
    return homo_pose_mat


@jit_decorator
def quat2mat(quaternion):
    """
    Converts given quaternion (x, y, z, w) to matrix.

    Args:
        quaternion: vec4 float angles

    Returns:
        3x3 rotation matrix
    """

    # awkward semantics for use with numba
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]

    n = np.dot(q, q)
    if n < EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ]
    )


def quat2euler(quaternion):
    """
    Convert quaternion to euler angle in radian
    :param quaternion: vec4 float angles in quaternion (x, y, z, w)
    :return: vec3 float angles in radian (roll, pitch, yaw)
    """
    return np.array(p.getEulerFromQuaternion(quaternion))


def euler2quat(euler):
    """
    Convert euler angle in radian to quaternion
    :param euler: vec3 float angles in radian
    :return: vec4 float angles in quaternion (x, y, z, w)
    """
    return np.array(p.getQuaternionFromEuler(euler))


def euler_matrix(ai, aj, ak, axes='sxyz'):
    """Return homogeneous rotation matrix from Euler angles and axis sequence.
    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple
    >>> R = euler_matrix(1, 2, 3, 'syxz')
    >>> numpy.allclose(numpy.sum(R[0]), -1.34786452)
    True
    >>> R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
    >>> numpy.allclose(numpy.sum(R[0]), -0.383436184)
    True
    >>> ai, aj, ak = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)
    >>> for axes in _TUPLE2AXES.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = np.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj*si
        M[i, k] = sj*ci
        M[j, i] = sj*sk
        M[j, j] = -cj*ss+cc
        M[j, k] = -cj*cs-sc
        M[k, i] = -sj*ck
        M[k, j] = cj*sc+cs
        M[k, k] = cj*cc-ss
    else:
        M[i, i] = cj*ck
        M[i, j] = sj*sc-cs
        M[i, k] = sj*cc+ss
        M[j, i] = cj*sk
        M[j, j] = sj*ss+cc
        M[j, k] = sj*cs-sc
        M[k, i] = -sj
        M[k, j] = cj*si
        M[k, k] = cj*ci
    return M


def pose_in_A_to_pose_in_B(pose_A, pose_A_in_B):
    """
    Converts a homogenous matrix corresponding to a point C in frame A 
    to a homogenous matrix corresponding to the same point C in frame B.

    :param pose_A: numpy array of shape (4,4) corresponding to the pose of C in frame A
    :param pose_A_in_B: numpy array of shape (4,4) corresponding to the pose of A in frame B
    
    :return: numpy array of shape (4,4) corresponding to the pose of C in frame B
    """

    # pose of A in B takes a point in A and transforms it to a point in C.

    # pose of C in B = pose of A in B * pose of C in A
    # take a point in C, transform it to A, then to B
    # T_B^C = T_A^C * T_B^A
    return pose_A_in_B.dot(pose_A)


def pose_inv(pose):
    """
    Computes the inverse of a homogenous matrix corresponding to the pose of some
    frame B in frame A. The inverse is the pose of frame A in frame B.

    :param pose: numpy array of shape (4,4) for the pose to inverse

    :return: numpy array of shape (4,4) for the inverse pose
    """

    # Note, the inverse of a pose matrix is the following
    # [R t; 0 1]^-1 = [R.T -R.T*t; 0 1]

    # Intuitively, this makes sense. 
    # The original pose matrix translates by t, then rotates by R.
    # We just invert the rotation by applying R-1 = R.T, and also translate back.
    # Since we apply translation first before rotation, we need to translate by 
    # -t in the original frame, which is -R-1*t in the new frame, and then rotate back by 
    # R-1 to align the axis again. 
    
    pose_inv = np.zeros((4,4))
    pose_inv[:3, :3] = pose[:3, :3].T
    pose_inv[:3, 3] = -pose_inv[:3, :3].dot(pose[:3, 3])
    pose_inv[3, 3] = 1.0
    return pose_inv


def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = numpy.identity(4, numpy.float64)
    >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> numpy.allclose(2., numpy.trace(rotation_matrix(math.pi/2,
    ...                                                direc, point)))
    True
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(((cosa, 0.0,  0.0),
                     (0.0,  cosa, 0.0),
                     (0.0,  0.0,  cosa)), dtype=np.float64)
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array((( 0.0,         -direction[2],  direction[1]),
                      ( direction[2], 0.0,          -direction[0]),
                      (-direction[1], direction[0],  0.0)),
                     dtype=np.float64)
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def reflection_matrix(point, normal):
    """Return matrix to mirror at plane defined by point and normal vector.
    >>> v0 = numpy.random.random(4) - 0.5
    >>> v0[3] = 1.0
    >>> v1 = numpy.random.random(3) - 0.5
    >>> R = reflection_matrix(v0, v1)
    >>> numpy.allclose(2., numpy.trace(R))
    True
    >>> numpy.allclose(v0, numpy.dot(R, v0))
    True
    >>> v2 = v0.copy()
    >>> v2[:3] += v1
    >>> v3 = v0.copy()
    >>> v2[:3] -= v1
    >>> numpy.allclose(v2, numpy.dot(R, v3))
    True
    """
    normal = unit_vector(normal[:3])
    M = np.identity(4)
    M[:3, :3] -= 2.0 * np.outer(normal, normal)
    M[:3, 3] = (2.0 * np.dot(point[:3], normal)) * normal
    return M


def make_pose(translation, rotation):
    """
    Make a homogenous pose matrix from a translation vector and a rotation matrix.

    :param translation: a 3-dim iterable
    :param rotation: a 3x3 matrix

    :return pose: a 4x4 homogenous matrix
    """
    pose = np.zeros((4,4))
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    pose[3, 3] = 1.0
    return pose


def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. eucledian norm, along axis.
    >>> v0 = numpy.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
    True
    >>> v0 = numpy.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float64)
    >>> unit_vector(v0, axis=1, out=v1)
    >>> numpy.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1.0]))
    [1.0]
    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    """
    Reimplement math.isclose()
    """
    if hasattr(math, "isclose"):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def quat2axisangle(quat):
    """
    Converts (x, y, z, w) quaternion to axis-angle format.
    Returns a unit vector direction and an angle.
    """

    # conversion from axis-angle to quaternion:
    #   qw = cos(theta / 2); qx, qy, qz = u * sin(theta / 2)

    # normalize qx, qy, qz by sqrt(qx^2 + qy^2 + qz^2) = sqrt(1 - qw^2)
    # to extract the unit vector

    # clipping for scalar with if-else is orders of magnitude faster than numpy
    if quat[3] > 1.:
        quat[3] = 1.
    elif quat[3] < -1.:
        quat[3] = -1.

    den = np.sqrt(1. - quat[3] * quat[3])
    if isclose(den, 0.):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3), 0.

    # convert qw to theta
    theta = 2. * math.acos(quat[3])

    return quat[:3] / den, 2. * math.acos(quat[3])


def axisangle2quat(axis, angle):
    """
    Converts axis-angle to (x, y, z, w) quat.
    """

    # handle zero-rotation case
    if isclose(angle, 0.):
        return np.array([0., 0., 0., 1.])

    # make sure that axis is a unit vector
    assert isclose(np.linalg.norm(axis), 1., rel_tol=1e-3)

    q = np.zeros(4)
    q[3] = np.cos(angle / 2.)
    q[:3] = axis * np.sin(angle / 2.)
    return q


def vec2axisangle(vec):
    """
    Converts Euler vector (exponential coordinates) to axis-angle.
    """
    angle = np.linalg.norm(vec)
    if isclose(angle, 0.):
        # treat as a zero rotation
        return np.array([1., 0., 0.]), 0.
    axis = vec / angle
    return axis, angle


def axisangle2vec(axis, angle):
    """
    Converts axis-angle to Euler vector (exponential coordinates).
    """
    return axis * angle


def quat_slerp(q1, q2, tau):
    """
    Taken from robosuite.
    """
    if tau == 0.0:
        return q1
    elif tau == 1.0:
        return q2
    d = np.dot(q1, q2)
    if abs(abs(d) - 1.0) < EPS:
        return q1
    if d < 0.0:
        # invert rotation
        d = -d
        q2 = -1.0 * q2
    angle = math.acos(np.clip(d, -1, 1))
    if abs(angle) < EPS:
        return q1
    isin = 1.0 / math.sin(angle)
    q1 = q1 * math.sin((1.0 - tau) * angle) * isin
    q2 = q2 * math.sin(tau * angle) * isin
    q1 = q1 + q2
    return q1


# def sha256_checksum(filename, block_size=65536):
#     """
#     Computes the checksum of a file. 
#     """
#     sha256 = hashlib.sha256()
#     with open(filename, 'rb') as f:
#         for block in iter(lambda: f.read(block_size), b''):
#             sha256.update(block)
#     return sha256.hexdigest()


# def json_dump(dic, filename=None):
#     """
#     Dumps a python dictionary to a json file.
#     If filename is not None, dump to file.
#     Returns a string.
#     """
#     json_string = json.dumps(dic, indent=4)
#     if filename is not None:
#         f = open(filename, "w")
#         f.write(json_string)
#         f.close()
#     return json_string


# def flatten_nested_dict(d, parent_key='', sep='_', item_key=''):
#     """
#     Flatten a nested dict to a list of key-value pairs. This function also works for hdf5 groups.
#     Converting the output to a dictionary is easy as well - just call `dict` on it.
#     """
#     items = []
#     if isinstance(d, dict) or isinstance(d, h5py.Group):
#         new_key = parent_key + sep + item_key if len(parent_key) > 0 else item_key
#         for k, v in d.items():
#             k = str(k)
#             assert isinstance(k, str)
#             items.extend(flatten_nested_dict(v, new_key, sep=sep, item_key=k))
#         return items
#     else:
#         new_key = parent_key + sep + item_key if len(parent_key) > 0 else item_key
#         return [(new_key, d)]


# class Rate(object):
#     """
#     Convenience class for enforcing rates in loops. Modeled after rospy.Rate.

#     See http://docs.ros.org/en/jade/api/rospy/html/rospy.timer-pysrc.html#Rate.sleep
#     """
#     def __init__(self, hz):
#         """
#         Args:
#             hz (int): frequency to enforce
#         """
#         self.hz = hz
#         self.last_time = time.time()
#         self.sleep_duration = (1. / hz)

#     def _remaining(self, curr_time):
#         """
#         Calculate time remaining for rate to sleep.
#         """
#         assert curr_time >= self.last_time, "time moved backwards!"
#         elapsed = curr_time - self.last_time
#         return self.sleep_duration - elapsed

#     def sleep(self):
#         """
#         Attempt to sleep at the specified rate in hz, by taking the time
#         elapsed since the last call to this function into account.
#         """
#         curr_time = time.time()
#         remaining = self._remaining(curr_time)
#         if remaining > 0:
#             time.sleep(remaining)

#         # assume successful rate sleeping
#         self.last_time = self.last_time + self.sleep_duration

#         # NOTE: this commented line is what we used to do, but this enforces a slower rate
#         # self.last_time = time.time()

#         # detect time jumping forwards (e.g. loop is too slow)
#         if curr_time - self.last_time > self.sleep_duration * 2:
#             # we didn't sleep at all
#             self.last_time = curr_time


# class Timer(object):
#     """
#     A simple timer.
#     """
#     def __init__(self, history=100, ignore_first=False, make_thread_safe=False):
#         """
#         Args:
#             history (int): number of recent timesteps to record for reporting statistics
#         """
#         self.total_time = 0.
#         self.calls = 0
#         self.start_time = 0.
#         self.last_diff = 0.
#         self.average_time = 0.
#         self.min_diff = float("inf")
#         self.max_diff = 0.
#         self._measurements = deque(maxlen=history)
#         self._enabled = True
#         self.ignore_first = ignore_first
#         self._had_first = False
#         self._thread_safe = make_thread_safe
#         if self._thread_safe:
#             self._lock = threading.Lock()

#     def maybe_acquire_lock(self):
#         if self._thread_safe:
#             self._lock.acquire()

#     def maybe_release_lock(self):
#         if self._thread_safe:
#             self._lock.release()

#     def enable(self):
#         """
#         Enable measurements with this timer.
#         """
#         self._enabled = True

#     def disable(self):
#         """
#         Disable measurements with this timer.
#         """
#         self._enabled = False

#     def tic(self):
#         # using time.time instead of time.clock because time time.clock
#         # does not normalize for multithreading
#         self.maybe_acquire_lock()
#         self.start_time = time.time()
#         self.maybe_release_lock()

#     def toc(self):
#         self.maybe_acquire_lock()
#         if self._enabled:

#             if self.ignore_first and (self.start_time > 0. and not self._had_first):
#                 self._had_first = True
#                 return time.time() - self.start_time

#             self.last_diff = time.time() - self.start_time
#             self.total_time += self.last_diff
#             self.calls += 1
#             self.average_time = self.total_time / self.calls
#             self.min_diff = min(self.min_diff, self.last_diff)
#             self.max_diff = max(self.max_diff, self.last_diff)
#             self._measurements.append(self.last_diff)
#         last_diff = self.last_diff
#         self.maybe_release_lock()
#         return last_diff

#     @contextmanager
#     def timed(self):
#         self.tic()
#         yield
#         self.toc()

#     def report_stats(self, verbose=False):
#         self.maybe_acquire_lock()
#         stats = OrderedDict()
#         stats["global"] = OrderedDict(
#             mean=self.average_time,
#             min=self.min_diff,
#             max=self.max_diff,
#             num=self.calls,
#         )
#         num = len(self._measurements)
#         stats["local"] = OrderedDict()
#         if num > 0:
#             stats["local"] = OrderedDict(
#                 mean=np.mean(self._measurements),
#                 std=np.std(self._measurements),
#                 min=np.min(self._measurements),
#                 max=np.max(self._measurements),
#                 num=num,
#             )
#         if verbose:
#             stats["local"]["values"] = list(self._measurements)
#         self.maybe_release_lock()
#         return stats


# class Timers(object):
#     """
#     Collection of timers.
#     """
#     def __init__(self, history=100, disable_on_creation=False, make_thread_safe=False):
#         """
#         Args:
#             history (int): number of recent timesteps to record for reporting statistics
#         """
#         self._timers = OrderedDict()
#         self.history = history
#         self.disable_on_creation = disable_on_creation
#         self.make_thread_safe = make_thread_safe

#     def enable(self, key=None):
#         """
#         Enable measurements with the timers.
#         """
#         if key is None:
#             [self._timers[k].enable() for k in self._timers]
#         else:
#             self._timers[key].enable()

#     def disable(self, key=None):
#         """
#         Disable measurements with the timers.
#         """
#         if key is None:
#             [self._timers[k].disable() for k in self._timers]
#         else:
#             self._timers[key].disable()

#     def tic(self, key):
#         if key not in self._timers:
#             self._timers[key] = Timer(history=self.history, make_thread_safe=self.make_thread_safe)
#             if self.disable_on_creation:
#                 self._timers[key].disable()
#         self._timers[key].tic()

#     def toc(self, key):
#         return self._timers[key].toc()

#     @contextmanager
#     def timed(self, key):
#         self.tic(key)
#         yield
#         self.toc(key)

#     def report_stats(self, verbose=False):
#         return { k : self._timers[k].report_stats(verbose=verbose) for k in self._timers }

#     def __str__(self):
#         stats = self.report_stats(verbose=False)
#         return json.dumps(stats, indent=4)


# class RateMeasure(object):
#     """
#     Measure approximate time intervals of code execution by calling @measure
#     """
#     def __init__(self, name=None, history=100, freq_threshold=None):
#         self._timer = Timer(history=history, ignore_first=True)
#         self._timer.tic()
#         self.name = name
#         self.freq_threshold = freq_threshold
#         self._enabled = True
#         self._first = False
#         self.sum = 0.
#         self.calls = 0

#     def enable(self):
#         """
#         Enable measurements.
#         """
#         self._timer.enable()
#         self._enabled = True

#     def disable(self):
#         """
#         Disable measurements.
#         """
#         self._timer.disable()
#         self._enabled = False

#     def measure(self):
#         """
#         Take a measurement of the time elapsed since the last @measure call
#         and also return the time elapsed.
#         """
#         interval = self._timer.toc()
#         self._timer.tic()
#         self.sum += (1. / interval)
#         self.calls += 1
#         if self._enabled and (self.freq_threshold is not None) and ((1. / interval) < self.freq_threshold):
#             print("WARNING: RateMeasure {} violated threshold {} hz with measurement {} hz".format(self.name, self.freq_threshold, (1. / interval)))
#             return (interval, True)
#         return (interval, False)

#     def report_stats(self, verbose=False):
#         """
#         Report statistics over measurements, converting timer measurements into frequencies.
#         """
#         stats = self._timer.report_stats(verbose=verbose)
#         stats["name"] = self.name
#         if stats["global"]["num"] > 0:
#             stats["global"] = OrderedDict(
#                 mean=(self.sum / float(self.calls)),
#                 min=(1. / stats["global"]["max"]),
#                 max=(1. / stats["global"]["min"]),
#                 num=stats["global"]["num"],
#             )
#         if len(stats["local"]) > 0:
#             measurements = [1. / x for x in self._timer._measurements]
#             stats["local"] = OrderedDict(
#                 mean=np.mean(measurements),
#                 std=np.std(measurements),
#                 min=np.min(measurements),
#                 max=np.max(measurements),
#                 num=stats["local"]["num"],
#             )
#         return stats

#     def __str__(self):
#         stats = self.report_stats(verbose=False)
#         return json.dumps(stats, indent=4)
