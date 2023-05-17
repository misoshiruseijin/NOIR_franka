"""
Robot client script using flask. For communication between EEG lab.
"""

from environments.realrobot_env_multi import RealRobotEnvMulti

import requests
import cv2
import numpy as np

img0 = cv2.imread("sample_images/camera0.png")
img1 = cv2.imread("sample_images/camera1.png")
img2 = cv2.imread("sample_images/camera2.png")
dummy_data_to_send = {
    "name" : "TableSetting",
    "img0" : img0.tolist(),
    "img1" : img1.tolist(),
    "img2" : img2.tolist(),
}

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
            print("params reset successfully")
        else:
            print("params not reset!!")
        print("Received Code :", code)
        return code
    else:
        return None


def main():

    breakpoint()
    # # initialize environments and gety initial observation
    # env = RealRobotEnvMulti()
    # dummy_action = np.zeros(env.skill.num_skills)
    # # img_data = env.get_image_observations(dummy_action, img_as_list=True)

    # img_data = env.get_image_observations(dummy_action, img_as_list=True, save_images=True)

    ###### No Resend Image Case #####
    # send_images(dummy_data_to_send)

    # while True:
    #     # get new skill + parmeters
    #     send_images(img_data)

    #     # get params
    #     action = None
    #     while action is None:
    #         action = check_skill_params_ready()
    #         print("skill and params", action)
        
    #     # take action
    #     obs, reward, done, info = env.step(action)
    #     img_data = env.get_image_observations(action, img_as_list=True)


if __name__ == "__main__":
    main()


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