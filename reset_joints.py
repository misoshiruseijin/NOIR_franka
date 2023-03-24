import numpy as np
import sys
sys.path.append("..")
sys.path.insert(1, "/home/eeg/deoxys_control/deoxys")

from deoxys.utils.config_utils import (get_default_controller_config, verify_controller_config)
from deoxys.franka_interface import FrankaInterface
from deoxys.experimental.motion_utils import reset_joints_to


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
    general_cfg_file="config/charmander.yml"
)
controller_type = "OSC_YAW"
controller_config = get_default_controller_config(controller_type)
reset_joints_to(robot_interface, reset_joint_positions)