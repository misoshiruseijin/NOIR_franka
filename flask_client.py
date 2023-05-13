# import requests
# data_to_send = {'name': 'John', 'age': 30}
# response = requests.post('http://172.24.68.104:8000/api/data', json=data_to_send)
# print(response)
# if response.status_code == 200:
#     print('Data sent successfully')
# else:
#     print('Failed to send data')


# import requests
# import cv2
# img0 = cv2.imread("sample_images/camera0.png")
# img1 = cv2.imread("sample_images/camera1.png")
# img2 = cv2.imread("sample_images/camera2.png")
# data_to_send = {
#     "img0" : img0.tolist(),
#     "img1" : img1.tolist(),
#     "img2" : img2.tolist(),
# }
# response = requests.post('http://172.24.68.104:8000/api/data', json=data_to_send)
# print(response)
# if response.status_code == 200:
#     print('skills and parameters received successfully!')
#     response = 
#     while(response.text):
#         response = requests.post('http://172.24.68.104:8000/api/data', json=data_to_send)
# else:
#     print('Failed to send data')
# # breakpoint()



# from flask import Flask, jsonify, request
# import cv2
# import requests
# app = Flask(__name__)
# @app.route('/api/data', methods=['POST'])

# class RobotServerClient:
#     def __init__(
#         self
#     ):
#         self.rcvd_data = {}
#         self.my_data = {}
#         self.sent_data = False
#         self.eeg_addr = 'http://172.24.68.104:8000/api/data'
#         self.robot_addr = 'http://172.16.0.1:8000/api/data'

#         img0 = cv2.imread("sample_images/camera0.png")
#         img1 = cv2.imread("sample_images/camera1.png")
#         img2 = cv2.imread("sample_images/camera2.png")
#         self.dummy_img_data = {
#             "img0" : img0.tolist(),
#             "img1" : img1.tolist(),
#             "img2" : img2.tolist(),
#         }

#     def receive_data(self):
#         self.data = request.get_json()
#         return jsonify({'message': 'Robot received data successfully'})
    
#     def send_data(self):
#         response = requests.post(self.eeg_addr, self.dummy_img_data)
#         self.sent_data = True
#         if response.status_code == 200:
#             print('Data sent successfully')
#         else:
#             print('Failed to send data')

# def main():
#     sc = RobotServerClient()
#     sc.send_data()
#     sc.receive_data()
#     # while True:
#     #     if not sc.sent_data:
#     #         response = sc.send_data(sc.eeg_addr, json=sc.dummy_img_data)
#     #         print("received response",response)
#     #     else:
#     breakpoint()

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8000)
#     main()

from environments.realrobot_env_multi import RealRobotEnvMulti

import requests
import cv2
import numpy as np

img0 = cv2.imread("sample_images/camera0.png")
img1 = cv2.imread("sample_images/camera1.png")
img2 = cv2.imread("sample_images/camera2.png")
dummy_data_to_send = {
    "img0" : img0.tolist(),
    "img1" : img1.tolist(),
    "img2" : img2.tolist(),
}

SERVER_IP = '10.124.52.226' # EEG
PORT = 5000
# SERVER_IP = '172.24.68.104' # Gates

# Send images JSON and receive processed data
def send_images(images):
    url = f"http://{SERVER_IP}:{PORT}/api/action"
    try:
        response = requests.post(url, json=images, timeout=120)
    except requests.exceptions.Timeout:
        print("Requests Timed Out!")
    except requests.exceptions.RequestException as e:
        print("Request error: ", e)
    if response.status_code == 200:
        processed_data = response.json()
        print("Processed data:", processed_data)
        return processed_data
    else:
        print("Error:", response.status_code)
        return None

# # Send images 
# def send_images(images):
#     url = 'http://172.24.68.104:8000/api/images'
#     try:
#         response = requests.post(url, json=images, timeout=120)
#     except requests.exceptions.Timeout:
#         print("Requests Timed Out!")
#     except requests.exceptions.RequestException as e:
#         print("Request error: ", e)
#     if response.status_code == 200:
#         print("server received images")
#     else:
#         print("Error:", response.status_code)
#         return None

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
    url = f"http://{SERVER_IP}:{PORT}/api/ready"
    response = requests.get(url)
    if response.status_code == 200:
        code = response.json()["ready"]
        assert code in [0, 1, 2], f"inproper code {code} received"
        return code
    else:
        return None

# # Request params
# def request_action():
#     url = 'http://172.24.68.104:8000/api/action'
#     response = requests.get(url)
#     if response.status_code == 200:
#         action = response.json()["action"]
#         return action
#     else:
#         print("error processing received params")
#         return None


def main():

    # initialize environments and gety initial observation
    env = RealRobotEnvMulti()
    dummy_action = np.zeros(env.skill.num_skills)
    img_data = env.get_image_observations(dummy_action, img_as_list=True)

    ######## Resend Image Case ######
    # # define flags
    # sent_imgs = False
    # rcvd_params = False

    # while True:
        
    #     # should send new images
    #     if not sent_imgs:
    #         print("sending images...")
    #         send_images(img_data)
    #         sent_imgs = True
        
    #     # should wait for action
    #     elif not rcvd_params:
    #         code = check_skill_params_ready()
    #         if code == 0:
    #             pass
    #         elif code == 1:
    #             action = request_action()
    #             rcvd_params = True
    #         elif code == 2:
    #             sent_imgs = False
    #         else:
    #             print(f"received unexpected code {code}")

    #     # ready to execute action
    #     else:
    #         obs, reward, done, info = env.step(action)
    #         sent_imgs = False
    #         rcvd_params = False


    ###### No Resend Image Case #####
    while True:
        # get new skill + parmeters
        action = send_images(img_data)["action"]
        print("skill and params", action)
        
        # take action
        obs, reward, done, info = env.step(action)
        img_data = env.get_image_observations(action, img_as_list=True)


if __name__ == "__main__":
    main()

# # Example usage
# images_data = data_to_send
# send_images(images_data)
# check_interrupt()