import socket
import pickle
import numpy as np
import time

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

class DummyServer:
    def __init__(
        self,
        host="127.0.0.1", # server IP address
        port=65432, # port used by server
    ):

        self.server_socket = socket.socket()
        self.server_socket.bind((host, port))
        self.server_socket.listen()
        self.conn, addr = self.server_socket.accept()
        print(f"Connected by {addr}")

    def ask_images(self):
        """
        Receive images
        """
        images = None
        data = b''

        while True:
            data_piece = self.conn.recv(4096)
            if not data_piece:
                break
            data += data_piece
        try:
            print("try")
            images = pickle.loads(data)
            return images
        except:
            print("Except")
            return {}

    def send_action(self, action:np.array):
        msg = action.tobytes()
        self.server_socket.sendall(msg)
        print("sent action", action)

dummy_server = DummyServer()
images = dummy_server.ask_images()
print("received images")

# skill_selection_vec = np.zeros(12)
# skill_selection_vec[0] = 1
# params = [0.5, 0.0, 0.15]
# action = np.concatenate([skill_selection_vec, params])
# dummy_server.send_action(action)


# import cv2
# for key in images:
#     cv2.imshow(f"{key}", images[key])
#     cv2.waitKey(0)
# cv2.destroyAllWindows()