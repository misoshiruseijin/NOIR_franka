import requests
from PIL import Image, ImageDraw
import torch
import pdb
import numpy as np 

from transformers import OwlViTProcessor, OwlViTForObjectDetection

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open("testimg.png")
texts = [["a photo of a cat", "a photo of a dog"]]
inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)

# image.save("testimg.png")

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]])
# Convert outputs (bounding boxes and class logits) to COCO API
results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

i = 0  # Retrieve predictions for the first image for the corresponding text queries
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

score_threshold = 0.1
bounding_boxes = []
center_coords = []

for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    if score >= score_threshold:
        bounding_boxes.append(box)
        center = ( (box[0] + box[2])/2, (box[1] + box[3])/2 )
        center_coords.append(center)
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

for box, center in zip(bounding_boxes, center_coords):

    # create rectangle image
    img1 = ImageDraw.Draw(image)  
    img1.rectangle(box, fill=None, outline="red")
    r = 5
    x0, y0 = center[0] - r, center[1] - r
    x1, y1 = center[0] + r, center[1] + r
    img1.ellipse([x0, y0, x1, y1], fill="red")
image.show()