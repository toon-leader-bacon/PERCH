from typing import Dict, List
from transformers import pipeline
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw
import requests


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


image_name = "snapshot-00077"
image: Image.Image = Image.open(
  f"./pictures/{image_name}.jpg"
)
# image.show()

# https://huggingface.co/facebook/detr-resnet-50
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs,
                                                  target_sizes=target_sizes,
                                                  threshold=0.35)[0]

draw = ImageDraw.Draw(image)

# pipe = pipeline('image-classification', model="dima806/bird_species_image_detection", device=0)
# pipe = pipeline('image-classification', model='chriamue/bird-species-classifier', device=0)
pipe = pipeline('image-classification', model='gungbgs/bird_species_classifier', device=0)


i = 0
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    i += 1
    str_label = model.config.id2label[label.item()]
    if str_label != "bird":
        continue

    box = [round(i, 2) for i in box.tolist()]

    width, height = image.size
    x1, y1, x2, y2 = box
    width_mod = round(0.02 * width)
    height_mod = round(0.02 * height)
    box = (clamp(x1 - width_mod, 0, width),
           clamp(y1 - height_mod, 0, height),
           clamp(x2 + width_mod, 0, width),
           clamp(y2 + height_mod, 0, height))

    # draw.rectangle(box)
    crop_img = image.crop(box)
    crop_img.save(f"./tmp/{image_name}_{i}.jpg")
    print(i)

    bird_detection_result: List[Dict] = pipe(crop_img)
    print(bird_detection_result)

    final_prediction_label = "unknown"
    final_prediction_score = -1
    for detection_result in bird_detection_result:
        score = detection_result["score"]
        label = detection_result["label"]
        if ('SPARROW' in label) or ("DOVE" in label) or ('PIGEON' in label) or ('CARDINAL' in label):
            # Only consider reasonable predicted labels
            
            if score >= final_prediction_score:
                final_prediction_label = label
                final_prediction_score = score
                # print(f"Bird detected in crop {i}: {label} with confidence {score}")

    if final_prediction_label == "unknown":
        print(f"Unknown bird in crop: {image_name}_{i}.jpg")
        final_prediction_label = bird_detection_result[0]["label"]
        final_prediction_label = bird_detection_result[0]["score"]
    
    print("===========\n")


# image.save(f"./tmp/FullImage_BirdBoxes.jpg")

# 0073, 77 has a dove
# 498 cardinal
# female cardinal 998, 1005