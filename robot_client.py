import socket
import pickle
import numpy as np
import time

class RobotClient:
    def __init__(
        self,
        host="127.0.0.1", # server IP address
        port=65432, # port used by server
    ):
        self.client_socket = socket.socket()
        self.client_socket.connect((host, port))
        # self.client_socket.settimeout(1.0)

    def ask_action(self):
        """
        Receive action (skill selection vec, params)?
        """
        data = b''
        while True:
            data_piece = self.client_socket.recv(1024)
            if not data_piece:
                action = np.frombuffer(data)
                return action
            data += data_piece

    def send_images(self, image_dict):
        """
        Send observation images
        
        Args:
            image_dict (dict) : {key : image ndarray, key : image ndarray, ....}
        """
        print("sending image...")
        data = pickle.dumps(image_dict)
        self.client_socket.sendall(data)
        print("sent")
        response = self.client_socket.recv(1024)
        print("server's response", response)
        if response.decode() == "RECEIVED":
            return
        elif response.decode() == "FAILED":
            print("server did not receive the data")

    def close(self):
        print("Closing Connection")
        self.client_socket.close()

