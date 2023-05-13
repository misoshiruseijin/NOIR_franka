#!/usr/bin/env python3

import sys
import socket
import selectors
import traceback
import numpy as np
import cv2

from client_message import ClientMessage
sel = selectors.DefaultSelector()

def create_request(action, value): # TODO replace this to grab images
    if action == "search":
        return dict(
            type="text/json",
            encoding="utf-8",
            content=dict(action=action, value=value),
        )
    else:
        return dict(
            type="binary/custom-client-binary-type",
            encoding="binary",
            content=bytes(action + value, encoding="utf-8"),
        )

def start_connection(host, port, request):
    addr = (host, port)
    print(f"Starting connection to {addr}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)
    sock.connect_ex(addr)
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    message = ClientMessage(sel, sock, addr, request)
    sel.register(sock, events, data=message)


host = "127.0.0.1"
port = 5000
img0 = cv2.imread("sample_images/camera0.png")
img1 = cv2.imread("sample_images/camera1.png")
img2 = cv2.imread("sample_images/camera2.png")

while True:
    print("\n")
    input("hit enter")
    request = {
        "type" : "pickle",
        "encoding" : "utf-8",
        "content" : {
            "img0" : img0,
            "img1" : img1,
            "img2" : img2,
            # "img3" : np.ones((480,640,3))*30,
            # "img4" : np.ones((480,640,3))*40,
            # "img5" : np.ones((480,640,3))*50,
            # "img6" : np.ones((480,640,3))*60,
            # "img1" : np.zeros((2,2,3)),
            # "img2" : np.zeros((2,2,3)),
            # "img3" : np.zeros((2,2,3)),
        },
    }
    # request = create_request(action="search", value=values[idx])

    start_connection(host, port, request)

    try:
        while True:
            events = sel.select(timeout=1)
            msg_from_server = None
            for key, mask in events:
                message = key.data
                try:
                    msg_from_server = message.process_events(mask)
                    if msg_from_server is not None:
                        print("Got", msg_from_server)
                except Exception:
                    print(
                        f"Main: Error: Exception for {message.addr}:\n"
                        f"{traceback.format_exc()}"
                    )
                    message.close()
            # Check for a socket being monitored to continue.
            if not sel.get_map():
                break
    except KeyboardInterrupt:
        print("Caught keyboard interrupt, exiting")
