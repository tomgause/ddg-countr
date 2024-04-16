import argparse
import json
import os
import sys
from PIL import Image, ImageDraw
from datetime import datetime
import numpy as np
import torch
from config import *
sys.path.append(DINOV_LIB_PATH)
from dinov.BaseModel import BaseModel
from dinov import build_model
from utils.arguments import load_opt_from_config_file
from dinov_custom.utils import task_openset


def process_and_save_results(model, images, reference_images, results_path):
    """Process the masks from inference and save the resulting images with bounding boxes."""
    output_json = {}
    for i, (img, image_path) in enumerate(images):
        image, masks = inference(model, *reference_images, img)

        inf_result = Image.fromarray(image)
        save_path = os.path.join(results_path, 'out', f'{i}.jpg')
        os.makedirs(os.path.join(results_path,'out'), exist_ok=True)
        inf_result.save(save_path)
        
        original_img = Image.open(image_path)
        draw = ImageDraw.Draw(original_img)

        # Determine the thresholds for the bbox dimensions
        max_width = original_img.width * BOX_THRESHOLD
        max_height = original_img.height * BOX_THRESHOLD

        boxes = []

        for mask in masks:
            y_indices, x_indices = np.where(mask)

            if not y_indices.size or not x_indices.size:
                continue

            min_x, max_x = x_indices.min() *  original_img.width / 640, x_indices.max() * original_img.width / 640
            min_y, max_y = y_indices.min() * original_img.height / 640, y_indices.max() * original_img.height / 640
            width = max_x - min_x + 1
            height = max_y - min_y + 1

            # Filter out bboxes that are too large
            if width <= max_width and height <= max_height:
                draw.rectangle([min_x, min_y, max_x , max_y], outline="white", width=2)
                boxes.append([[min_x, min_y], [max_x , max_y]])

        # Save the modified image
        save_path = os.path.join(results_path, 'boxes', f'{i}.jpg')
        os.makedirs(os.path.join(results_path,'boxes'), exist_ok=True)
        original_img.save(save_path)
        print(f"Saved processed image to {save_path}")
        output_json[os.path.basename(image_path)] = boxes

    with open(os.path.join(results_path, 'data.json'), "w") as json_file:
        json.dump(output_json, json_file)


def parse_option():
    """Parse input options."""
    parser = argparse.ArgumentParser('DINOv Demo', add_help=False)
    parser.add_argument('--conf_files', default=DINOV_LIB_PATH + "/configs/dinov_sam_coco_swinl_train.yaml", metavar="FILE", help='path to config file')
    parser.add_argument('--ckpt', default=DINOV_CKPT_PATH, metavar="FILE", help='path to checkpoint')
    return parser.parse_args()


@torch.no_grad()
def inference(model, *args, **kwargs):
    """Perform inference using the provided model and arguments."""
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        results = task_openset(model, *args, **kwargs)
        return results


def load_model():
    """Load the model from predefined configurations."""
    args = parse_option()
    opt = load_opt_from_config_file(args.conf_files)
    model = BaseModel(opt, build_model(opt)).from_pretrained(args.ckpt).eval().cuda()
    return model


def main():
    model = load_model()

    formatted_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    os.makedirs(f"run_{formatted_time}", exist_ok=True)
    os.makedirs(f"run_{formatted_time}/dinov", exist_ok=True)
    os.makedirs(f"run_{formatted_time}/dinov/ref-images", exist_ok=True)

    # Load the existing birds.pnt data
    with open(BIRDS_PNT_PATH, 'r') as file:
        data = json.load(file)['points']

    inference_images = []
    reference_images = []
    count = 0
    for image_name, annotations in data.items():
        if len(annotations['bird']) > 6:
            image_path = os.path.join(IMAGE_DIR, image_name)
            with Image.open(image_path) as img:
                if count > 7:
                    inference_images.append((img.convert('RGB'), image_path))
                    count += 1
                    if count > MAX_COUNT: break
                    continue 

                overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))  # Fully transparent
                draw = ImageDraw.Draw(overlay)
                for bird in annotations['bird']:
                    left_up_point = (bird['x'] - CIRCLE_RADIUS, bird['y'] - CIRCLE_RADIUS)
                    right_down_point = (bird['x'] + CIRCLE_RADIUS, bird['y'] + CIRCLE_RADIUS)
                    draw.ellipse([left_up_point, right_down_point], fill=COLOR, outline=COLOR)
                img_with_overlay = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
                img_with_overlay.save(os.path.join(f"run_{formatted_time}/dinov/ref-images", image_name))

                reference_images.append({
                    "image": img.convert('RGB'),
                    "mask": overlay
                })

                count += 1

    # Inference on collected images and save results
    process_and_save_results(model, inference_images, reference_images, results_path=f"run_{formatted_time}/dinov")

if __name__ == "__main__":
    main()
