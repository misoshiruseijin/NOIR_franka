import numpy as np
import sys
import argparse
sys.path.append("..")
sys.path.insert(1, "/home/eeg/deoxys_control/deoxys")

from deoxys.utils.config_utils import (get_default_controller_config, verify_controller_config)
from deoxys.franka_interface import FrankaInterface
from deoxys.experimental.motion_utils import reset_joints_to


#### default reset position ######
# reset_joint_positions = [
#             0.09162008114028396,
#             -0.19826458111314524,
#             -0.01990020486871322,
#             -2.4732269941140346,
#             -0.01307073642274261,
#             2.30396583422025,
#             0.8480939705504309,
#         ]

#### NOIR Reset Position ####
parser = argparse.ArgumentParser()
parser.add_argument("--out", action="store_true")
args = parser.parse_args()

if args.out:
    reset_joint_positions = [-np.pi/2, -0.34306933, -0.01955571, -2.45878116, -0.01170808, 2.18055725, 0.84792026]
else:
    reset_joint_positions = [0.07263956, -0.34306933, -0.01955571, -2.45878116, -0.01170808, 2.18055725, 0.84792026]

robot_interface = FrankaInterface(
    general_cfg_file="config/charmander.yml"
)
controller_type = "OSC_YAW"
controller_config = get_default_controller_config(controller_type)
reset_joints_to(robot_interface, reset_joint_positions)
