
from deoxys.franka_interface import FrankaInterface
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.utils.config_utils import (get_default_controller_config, verify_controller_config)

import argparse

import pdb
import time
import numpy as np

def main():
    reset_joint_positions = [
        0.09162008114028396,
        -0.19826458111314524,
        -0.01990020486871322,
        -2.4732269941140346,
        -0.01307073642274261,
        2.30396583422025,
        0.8480939705504309,
    ]

    robot_interface = FrankaInterface(
        general_cfg_file="config/charmander.yml",
        control_freq=20,
    )
    # reset_joints_to(robot_interface, reset_joint_positions)
    time.sleep(1)

    while True:
        robot_state = robot_interface.last_eef_quat_and_pos
        eef_pos = robot_state[1].flatten()
        eef_quat = robot_state[0].flatten()
        robot_joints = robot_interface.last_q
        
        print("eef_pos", eef_pos)
        print("joint q", robot_joints)
        print("eef_quat", eef_quat)

main()