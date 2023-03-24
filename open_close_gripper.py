
from deoxys.franka_interface import FrankaInterface
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys.utils.config_utils import (get_default_controller_config, verify_controller_config)

import argparse

import pdb
import time
import numpy as np

def main(args):
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
    # time.sleep(1)

    controller_type = "OSC_POSE"
    controller_cfg = get_default_controller_config(controller_type)

    for _ in range(20):
        robot_interface.control(
            controller_type=controller_type,
            action=np.array([0., 0., 0., 0., 0., 0., args.gripper]),
            controller_cfg=controller_cfg,
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gripper", type=int, default=-1)
    args = parser.parse_args()
    print(args.gripper)
    main(args)