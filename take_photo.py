
from deoxys.camera_redis_interface import CameraRedisSubInterface
from utils.camera_utils import get_camera_image, get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, pose_inv
from PIL import Image
import numpy as np
import argparse
import time
import pdb

camera_interfaces = {
        0 : CameraRedisSubInterface(camera_id=0),
        1 : CameraRedisSubInterface(camera_id=1),
        2 : CameraRedisSubInterface(camera_id=2),
    }

time.sleep(0.5)


# range to trim camera image 2
trim_low = [90, 130]
trim_high = [450, 370]

def take_photos(prefix):
    for i in range(len(camera_interfaces)):
        camera_interface = camera_interfaces[i]
        raw_image = get_camera_image(camera_interface)
        rgb_image = raw_image[:,:,::-1] # convert from bgr to rgb
        # trim camera 2 images
        if i == 2:
            rgb_image = rgb_image[trim_low[1]:trim_high[1], trim_low[0]:trim_high[0]]
        image = Image.fromarray(np.uint8(rgb_image))
        image.save(f"photos/{prefix}_camera{i}.png")

def stream(camera_id, trim=True):
    trim_low = [90, 130]
    trim_high = [450, 370]
    camera_interface = camera_interfaces[camera_id]
    while True:
        raw_image = get_camera_image(camera_interface)
        rgb_image = raw_image[:,:,::-1] # convert from bgr to rgb
        if trim:
            rgb_image = rgb_image[trim_low[1]:trim_high[1], trim_low[0]:trim_high[0]]
        image = Image.fromarray(np.uint8(rgb_image))
        image.save(f"photos/_camera{camera_id}.png")
        time.sleep(0.2)

def main(args):
    if args.stream:
        stream(camera_id=args.id)
    else:
        take_photos(prefix=args.prefix)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--id", type=int, default=0)
    args = parser.parse_args()
    main(args)