from environments.realrobot_env_multi import RealRobotEnvMulti
from robot_client import RobotClient

import numpy as np
import argparse
import time

def main(args):

    # # initialize environment
    # env = RealRobotEnvMulti()

    # # get initial observations
    # dummy_action = np.zeros(env.skill.num_skills)
    # dummy_action[0] = 1

    # initialize client
    robot_client = RobotClient(
        host=args.host,
        port=args.port,
    )

    # image_dict = env.get_image_observations(action=dummy_action, save_images=True)
    image_dict = {"im0" : np.zeros((32, 32, 3))}
    robot_client.send_images(image_dict=image_dict)
    # time.sleep(3)

    # while True:
    #     # get action from server
    #     action = robot_client.ask_action()
    #     print("received action", action)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=65432)
    args = parser.parse_args()
    main(args)