from os import listdir
from os.path import isfile, join
from typing import Dict, List
from PIL import Image, ImageDraw
import torch
from pathlib import Path
from transformers import DetrImageProcessor, DetrForObjectDetection, pipeline


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


def add_or_init_key(birds_found, class_label):
    if class_label not in birds_found:
        birds_found[class_label] = 0
    birds_found[class_label] = birds_found[class_label] + 1


def process_single_image(image: Image.Image, image_name: str,
                         processor, model, pipe):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    # convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs,
                                                      target_sizes=target_sizes,
                                                      threshold=0.35)[0]

    birds_found = {}
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
        # print(i)

        bird_detection_result: List[Dict] = pipe(crop_img)
        # print(bird_detection_result)

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

        # if final_prediction_label == "unknown":
        #     # print(f"Unknown bird in crop: {image_name}_{i}.jpg")
        #     final_prediction_label = bird_detection_result[0]["label"]
        #     final_prediction_score = bird_detection_result[0]["score"]

        add_or_init_key(birds_found, final_prediction_label)
    return birds_found


def merge_dictionaries(a: Dict[str, int], b: Dict[str, int]):
    result = a.copy()
    for key, value in b.items():
        if key not in result:
            result[key] = 0
        result[key] = result[key] + value
    return result


def process_images_in_directory():
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    pipe = pipeline('image-classification', model='gungbgs/bird_species_classifier', device=0)

    all_birds = {}

    path_to_images = "./pictures/"
    frames_with_birds = 0
    all_files = [f for f in listdir(path_to_images) if isfile(join(path_to_images, f))]
    for file in all_files:
        if "DS_Store" in file:
            continue
        print(f"Considering image: {file}")
        file_name = Path(f"{path_to_images}/{file}")
        file_stem = file_name.stem  # Just the file name without ".jpg"

        image = Image.open(file_name)
        birds_in_img = process_single_image(image, file_stem, processor, model, pipe)
        all_birds = merge_dictionaries(all_birds, birds_in_img)
        if len(birds_in_img) > 0:
            frames_with_birds += 1

    print(all_birds)
    print(f"frames_with_birds: {frames_with_birds}")


if __name__ == "__main__":
    process_images_in_directory()


# {'ROCK DOVE': 194, 'HOUSE SPARROW': 1238, 0.950392484664917: 1, 'ZEBRA DOVE': 72, 'CHIPPING SPARROW': 256,
#     0.831638514995575: 1, 'MOURNING DOVE': 42, 0.8688039779663086: 1, 'BLACK-THROATED SPARROW': 120, 'JACOBIN PIGEON': 125, }


{
    'ROCK DOVE': 194,
    'HOUSE SPARROW': 1238,
    'unknown': 838,
    'ZEBRA DOVE': 72,
    'CHIPPING SPARROW': 256,
    'MOURNING DOVE': 42,
    'BLACK-THROATED SPARROW': 120,
    'JACOBIN PIGEON': 125,
    'FRILL BACK PIGEON': 140,
    'NORTHERN CARDINAL': 80,
    'GREEN WINGED DOVE': 25,
    'JAVA SPARROW': 2
}
