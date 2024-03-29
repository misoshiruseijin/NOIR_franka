from utils.detection_utils import DetectionUtils
from deoxys.camera_redis_interface import CameraRedisSubInterface
from utils.camera_utils import get_camera_image, get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, pose_inv
# from primitive_skills_noir import PrimitiveSkill

import time
import pdb
# camera_interfaces = {
#         0 : CameraRedisSubInterface(camera_id=0),
#         1 : CameraRedisSubInterface(camera_id=1),
#     }

# detection_utils = DetectionUtils()
# texts = ["light blue bowl", "shiny silver cup", "brown handle"]
# for i in range(2):
#     coords = detection_utils.get_obj_pixel_coord(
#         camera_interface=camera_interfaces[i],
#         camera_id=i,
#         texts=texts,
#         thresholds=[0.001] * len(texts),
#         save_img=True,
#         n_instances=1
#     )
#     print(f"Coords in camera {i}")
#     print(coords)

import cv2
from utils.detection_utils_eeg import ObjectDetector
import json
import argparse


def main(args):
    with open('config/task_obj_skills.json') as json_file:
        task_dict = json.load(json_file)
        assert args.env in task_dict.keys(), f"Unrecognized environment name. Choose from {task_dict.keys()}"
        obj2skills = task_dict[args.env]
        obj_names = list(obj2skills.keys())

    detector = ObjectDetector()
    texts = obj_names.copy()
    texts.remove("none")
    print("looking for\n", texts)
    coords = detector.get_obj_pixel_coord(
        img_array=cv2.imread("photos/_camera2.png"),
        # img_array=cv2.imread("param_selection_img2.png"),
        texts=texts,
        thresholds=[0.002]*len(texts),
        debug=True,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, help="name of task")
    args = parser.parse_args()
    main(args)


# while True:
#     start_time = time.time()

#     coords = detection_utils.get_obj_pixel_coord(
#         camera_interface=camera_interfaces[0],
#         camera_id=0,
#         texts=["red bowl", "light blue bowl", "shiny silver cup", "green handle"],
#         thresholds=[0.001]*4,
#         save_img=False,
#         n_instances=1,
#     )
#     end_time = time.time()
#     print("time this iter", end_time-start_time)
#     # print(coords)

# from PIL import Image, ImageDraw, ImageFont
# import torch
# import pdb
# import numpy as np 

# from transformers import OwlViTProcessor, OwlViTForObjectDetection

# processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
# model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# image = Image.open("photos/whiteboard1_camera1.png")
# texts = ["dark blue eraser"]
# inputs = processor(text=texts, images=image, return_tensors="pt")
# outputs = model(**inputs)

# # image.save("testimg.png")

# # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
# target_sizes = torch.Tensor([image.size[::-1]])
# # Convert outputs (bounding boxes and class logits) to COCO API
# results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

# i = 0  # Retrieve predictions for the first image for the corresponding text queries
# boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

# coords = {key : {"boxes" : [], "centers" : [], "scores" : []} for key in texts}
# n_instances = 1
# save_img = True

# obj2thresh = {
#     "dark blue eraser" : 0.003,
# }

# for box, score, label in zip(boxes, scores, labels):
#     box = [round(i, 2) for i in box.tolist()]
#     obj_name = texts[label]
#     if score >= obj2thresh[obj_name]:
#         # print(f"Detected {obj_name} with confidence {round(score.item(), 3)} at location {box}")
#         coords[obj_name]["boxes"].append(box)
#         coords[obj_name]["scores"].append(score.item())
#         center = ( round((box[0] + box[2])/2), round((box[1] + box[3])/2) )
#         coords[obj_name]["centers"].append(center)
#         print("score", score)
    
# # extract the top n_instances objects - TODO if needed, allow different max number for each object
# for key in coords:
#     # check if there are more than n_instances 
#     if len(coords[key]["scores"]) > n_instances:
#         scores = coords[key]["scores"]
#         indices = np.argsort(scores)[::-1][:n_instances]
#         # discard all except top scoring ones
#         coords[key]["boxes"] = np.array(coords[key]["boxes"])[indices].tolist()
#         coords[key]["scores"] = np.array(coords[key]["scores"])[indices].tolist()
#         coords[key]["centers"] = np.array(coords[key]["centers"])[indices].tolist()

# if save_img:
#     draw = ImageDraw.Draw(image)
#     colors = ["red", "blue", "green", "purple", "orange", "black", "violet", "teal", "darkgreen"]
#     idx = 0
#     font = ImageFont.truetype('arial.ttf', 24)
#     txt_start_pos = (15, 15)
#     r = 5 # radius of center circle
#     # draw bounding boxes and save image
#     for key in coords:
#         for box, center in zip(coords[key]["boxes"], coords[key]["centers"]):
#             color = colors[idx]
#             draw.rectangle(box, fill=None, outline=color) # draw bounding box
#             x0, y0 = center[0] - r, center[1] - r
#             x1, y1 = center[0] + r, center[1] + r
#             draw.ellipse([x0, y0, x1, y1], fill=color) # draw center coord
#             draw.text((txt_start_pos[0], txt_start_pos[1]+28*idx), key, font = font, align ="left", fill=color) 
#             idx += 1
#     image.save(f"test.png")
#     print(f"Saved test.png")