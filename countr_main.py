import argparse
import json
import os
import sys
from datetime import datetime
import numpy as np

import torch
import timm
from config import COUNTR_MODEL_PATH, DEFAULT_DEVICE, COUNTR_LIB_PATH, DEFAULT_EVAL_JSON, IMAGE_DIR
from countr_custom.utils import load_image, run_one_image

# Ensure timm version compatibility
assert "0.4.5" <= timm.__version__ <= "0.4.9", "timm version must be between 0.4.5 and 0.4.9"

# Append the path for custom libraries
sys.path.append(COUNTR_LIB_PATH)

from models_mae_cross import mae_vit_base_patch16  # type: ignore

def parse_option():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="DDG CounTR Addon", add_help=False)
    parser.add_argument("--ckpt", default=COUNTR_MODEL_PATH, help="Path to the checkpoint file.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Device for computation.")
    parser.add_argument("--input", default="eval", help="Run mode: single image or evaluation.")
    parser.add_argument("--json", default=None, required=True, help="Path to evaluation JSON. Results are saved in same dir.")
    parser.add_argument("--boxes", default=None, help="Bounding boxes in [[[x1,y1],[x2,y2]],...] format for single image mode.")
    return parser.parse_args()

def main():
    args = parse_option()

    # Set up device and model
    device = torch.device(args.device)
    model = mae_vit_base_patch16(norm_pix_loss="store_true")
    model.to(device)
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    print(f"Resumed checkpoint {args.ckpt}")

    model.eval()

    # Prepare for result storage
    results_dir = f"{args.json.split('/')[-3]}/countr"
    results = []

    num_boxes = []
    if args.input == "eval":  # Evaluation mode
        with open(args.json, "r") as file:
            data = json.load(file)
        for key, value in data.items():
            if value:
                print(key)
                im_path = os.path.join(IMAGE_DIR, key)
                samples, boxes, pos = load_image(im_path, value)
                num_boxes.append(len(boxes))
                samples, boxes = samples.unsqueeze(0).to(device), boxes.unsqueeze(0).to(device)
                result, elapsed_time, points = run_one_image(samples, boxes, pos, model, os.path.basename(im_path), results_dir)
                print(result, elapsed_time.duration)
                results.append(result)
    else:  # Single image mode
        im_path = args.input
        samples, boxes, pos = load_image(im_path, eval(args.boxes))
        samples, boxes = samples.unsqueeze(0).to(device), boxes.unsqueeze(0).to(device)
        result, elapsed_time, points = run_one_image(samples, boxes, pos, model, os.path.basename(im_path), results_dir)
        print(result, elapsed_time.duration)
        results.append(result)

        # Save prediction tensor
        tensor_np = points.cpu().numpy()
        np.savetxt(f"{results_dir}/tensor.txt", tensor_np, fmt='%f')

    # Save count results
    with open(f"{results_dir}/result.txt", "w") as file:
        for element, boxes in zip(results,num_boxes):
            file.write(f"{element},{boxes}\n")

if __name__ == "__main__":
    main()
