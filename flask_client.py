"""
Robot client script using flask. For communication between EEG lab.
"""

from environments.realrobot_env_multi import RealRobotEnvMulti

import requests
import cv2
import numpy as np
import json
import time
import argparse
from datetime import datetime
import os
import logging

# example image data to send
img0 = cv2.imread("sample_images/camera0.png")
img1 = cv2.imread("sample_images/camera1.png")
img2 = cv2.imread("sample_images/camera2.png")
dummy_data_to_send = {
    "name" : "TableSetting",
    "img0" : img0.tolist(),
    "img1" : img1.tolist(),
    "img2" : img2.tolist(),
}

# hardcoded params
pick_vec, place_vec = [0]*12, [0]*12
pick_vec[0] = 1
place_vec[2] = 1
actions = [
    pick_vec + [0.47116807, -0.09464132, 0.02], # pick bowl
    place_vec + [0.48592121, 0.06346929, 0.04183363], # place bowl
    pick_vec + [0.32698827, -0.09672487, 0.04459089], # pick cup
    place_vec + [0.35086609, 0.17256692, 0.05520492], # place cup
    pick_vec + [0.37522164, -0.24257553, 0.01533997], # pick spoon
    place_vec + [0.42237265, -0.00106914, 0.07387539], # place spoon
]



SERVER_IP = '10.124.52.226' # EEG
PORT = 5000
# SERVER_IP = '172.24.68.104' # Gates

# Send images JSON and receive processed data
def send_images(images):
    url = f"http://{SERVER_IP}:{PORT}/api/images"
    try:
        response = requests.post(url, json=images, timeout=120)
    except requests.exceptions.Timeout:
        print("Requests Timed Out!")
    except requests.exceptions.RequestException as e:
        print("Request error: ", e)
    if response.status_code == 200:
        # processed_data = response.json()
        # print("Processed data:", processed_data)
        # return processed_data
        print("images sent!")
    else:
        print("Error:", response.status_code)
        # return None


# Check interrupt value
def check_interrupt():
    url = f"http://{SERVER_IP}:{PORT}/api/interrupt"
    response = requests.get(url)
    if response.status_code == 200:
        interrupt_value = response.json()['interrupt']
        return interrupt_value
    else:
        print("Error:", response.status_code)
        return None

# Checks if skill and params are ready
def check_skill_params_ready():
    url = f"http://{SERVER_IP}:{PORT}/api/action"
    response = requests.get(url)
    if response.status_code == 200:
        code = response.json()["action"]
        # assert code in [0, 1, 2], f"inproper code {code} received"
        url = f"http://{SERVER_IP}:{PORT}/api/resetparams"
        reset_response = requests.get(url)
        if reset_response.status_code == 200:
            pass
            # print("params reset successfully")
        else:
            print("params not reset!!")
        # print("Received Code :", code)
        return code
    else:
        return None

def main(args):

    env_name = args.env
    subject = args.subject
    with open('config/task_obj_skills.json') as json_file:
        task_dict = json.load(json_file)
        assert env_name in task_dict.keys(), f"Unrecognized environment name. Choose from {task_dict.keys()}"

    log_dir = "experiment_logs"
    exp_datetime = datetime.now().strftime("%m_%d_%y_%H_%M")
    log_file = f"{log_dir}/{env_name}_{args.subject}_{exp_datetime}.txt"

    logging.basicConfig(level=logging.DEBUG)
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(file_handler)

    # initialize environments and gety initial observation
    resset_joints = False if args.resume else True
    env = RealRobotEnvMulti(
        reset_joints_on_init=resset_joints,
    )
    if resset_joints:
        env.reset()
    action = np.zeros(env.num_skills)
    # img_data = env.get_image_observations(dummy_action, img_as_list=True, save_images=True)

    ###### No Resend Image Case #####
    eps_start_time = time.time() # episode start time
    logging.critical(f"Start Experiment {env_name} with Subject {subject}")

    while True:
        step_start_time = time.time()

        # get new images and send 
        img_data = env.get_image_observations(action, img_as_list=True, save_images=True)
        img_data["env_name"] = env_name

        send_images(img_data)
        decode_start_time = time.time()

        # get params
        action = None
        while action is None:
            action = check_skill_params_ready()
            # action = np.zeros(env.num_skills).tolist()
            # action[5] = 1
            # action += [0.5, 0.1, 0.3]
            # print("No action received")
            time.sleep(1)

        decode_end_time = time.time()
        
        # take action
        print("Action received. Executing ", action)
        obs, reward, done, info = env.step(action)
        step_end_time = time.time()
        
        skill_name = info["skill_name"]
        params = action[env.num_skills:]
        logging.critical(f"Skill : {skill_name}")
        logging.critical(f"Params : {params}")
        logging.critical(f"Time this step : {step_end_time - step_start_time}")
        logging.critical(f"EEG response time : {decode_end_time - decode_start_time}")
        logging.critical(f"Time elapsed : {step_end_time - eps_start_time}")
        
        print("Time this step: ", step_end_time - step_start_time)
        print("Decoding time: ", decode_end_time - decode_start_time)
        print("Total time so far: ", step_end_time - eps_start_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    # parser.add_argument("--exp_name", type=str, default="exp")
    parser.add_argument("--subject", type=str, default="A")
    parser.add_argument("--logdir", type=str, default="experiment_logs")
    parser.add_argument("--env", type=str)
    args = parser.parse_args()
    main(args)


# """
# Robot client script using flask. For communication between EEG lab.
# """

# from environments.realrobot_env_multi import RealRobotEnvMulti

# import requests
# import cv2
# import numpy as np

# img0 = cv2.imread("sample_images/camera0.png")
# img1 = cv2.imread("sample_images/camera1.png")
# img2 = cv2.imread("sample_images/camera2.png")
# dummy_data_to_send = {
#     "img0" : img0.tolist(),
#     "img1" : img1.tolist(),
#     "img2" : img2.tolist(),
# }

# SERVER_IP = '10.124.52.226' # EEG
# PORT = 5000
# # SERVER_IP = '172.24.68.104' # Gates

# # Send images JSON and receive processed data
# def send_images(images):
#     url = f"http://{SERVER_IP}:{PORT}/api/action"
#     try:
#         response = requests.post(url, json=images, timeout=120)
#     except requests.exceptions.Timeout:
#         print("Requests Timed Out!")
#     except requests.exceptions.RequestException as e:
#         print("Request error: ", e)
#     if response.status_code == 200:
#         processed_data = response.json()
#         print("Processed data:", processed_data)
#         return processed_data
#     else:
#         print("Error:", response.status_code)
#         return None

# # # Send images 
# # def send_images(images):
# #     url = 'http://172.24.68.104:8000/api/images'
# #     try:
# #         response = requests.post(url, json=images, timeout=120)
# #     except requests.exceptions.Timeout:
# #         print("Requests Timed Out!")
# #     except requests.exceptions.RequestException as e:
# #         print("Request error: ", e)
# #     if response.status_code == 200:
# #         print("server received images")
# #     else:
# #         print("Error:", response.status_code)
# #         return None

# # Check interrupt value
# def check_interrupt():
#     url = f"http://{SERVER_IP}:{PORT}/api/interrupt"
#     response = requests.get(url)
#     if response.status_code == 200:
#         interrupt_value = response.json()['interrupt']
#         return interrupt_value
#     else:
#         print("Error:", response.status_code)
#         return None

# # Checks if skill and params are ready
# def check_skill_params_ready():
#     url = f"http://{SERVER_IP}:{PORT}/api/ready"
#     response = requests.get(url)
#     if response.status_code == 200:
#         code = response.json()["ready"]
#         assert code in [0, 1, 2], f"inproper code {code} received"
#         return code
#     else:
#         return None

# # # Request params
# # def request_action():
# #     url = 'http://172.24.68.104:8000/api/action'
# #     response = requests.get(url)
# #     if response.status_code == 200:
# #         action = response.json()["action"]
# #         return action
# #     else:
# #         print("error processing received params")
# #         return None


# def main():

#     # initialize environments and gety initial observation
#     env = RealRobotEnvMulti()
#     dummy_action = np.zeros(env.skill.num_skills)
#     # img_data = env.get_image_observations(dummy_action, img_as_list=True)

#     img_data = env.get_image_observations(dummy_action, img_as_list=False, save_images=True)
#     ######## Resend Image Case ######
#     # # define flags
#     # sent_imgs = False
#     # rcvd_params = False

#     # while True:
        
#     #     # should send new images
#     #     if not sent_imgs:
#     #         print("sending images...")
#     #         send_images(img_data)
#     #         sent_imgs = True
        
#     #     # should wait for action
#     #     elif not rcvd_params:
#     #         code = check_skill_params_ready()
#     #         if code == 0:
#     #             pass
#     #         elif code == 1:
#     #             action = request_action()
#     #             rcvd_params = True
#     #         elif code == 2:
#     #             sent_imgs = False
#     #         else:
#     #             print(f"received unexpected code {code}")

#     #     # ready to execute action
#     #     else:
#     #         obs, reward, done, info = env.step(action)
#     #         sent_imgs = False
#     #         rcvd_params = False


#     ###### No Resend Image Case #####
#     while True:
#         # get new skill + parmeters
#         action = send_images(img_data)["action"]
#         print("skill and params", action)
        
#         # take action
#         obs, reward, done, info = env.step(action)
#         img_data = env.get_image_observations(action, img_as_list=True)


# if __name__ == "__main__":
#     main()

# # # Example usage
# # images_data = data_to_send
# # send_images(images_data)
# # check_interrupt()