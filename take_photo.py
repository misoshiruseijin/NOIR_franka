
from deoxys.camera_redis_interface import CameraRedisSubInterface
from utils.camera_utils import get_camera_image, get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, pose_inv
from PIL import Image
import numpy as np
import argparse
import time
import pdb

def main(args):
    camera_interfaces = {
        0 : CameraRedisSubInterface(camera_id=0),
        1 : CameraRedisSubInterface(camera_id=1),
        2 : CameraRedisSubInterface(camera_id=2),
    }

    time.sleep(0.5)
    
    for i in range(len(camera_interfaces)):
        camera_interface = camera_interfaces[i]
        raw_image = get_camera_image(camera_interface)
        rgb_image = raw_image[:,:,::-1] # convert from bgr to rgb
        image = Image.fromarray(np.uint8(rgb_image))
        image.save(f"photos/{args.prefix}_camera{i}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="")
    args = parser.parse_args()
    main(args)