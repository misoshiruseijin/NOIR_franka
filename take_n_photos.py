import argparse
import os
from utils.camera_utils import get_camera_image
from deoxys.camera_redis_interface import CameraRedisSubInterface
import numpy as np
from PIL import Image

def main(args):
    camera_interface = CameraRedisSubInterface(camera_id=2)
    trim_low = [90, 130]
    trim_high = [450, 370]
    os.makedirs(args.save_dir, exist_ok=True)
    n_imgs = args.n
    for i in range(n_imgs):
        input("press enter to take next photo")
        raw_image = get_camera_image(camera_interface)
        rgb_image = raw_image[:,:,::-1] # convert from bgr to rgb
        rgb_image = rgb_image[trim_low[1]:trim_high[1], trim_low[0]:trim_high[0]]
        image = Image.fromarray(np.uint8(rgb_image))
        image.save(f"{args.save_dir}/img{i}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="n_photos")
    args = parser.parse_args()
    main(args)