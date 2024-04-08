import argparse
import json
import os
import sys
from datetime import datetime
import numpy as np

import torch
import timm
from config import COUNTR_MODEL_PATH, DEFAULT_DEVICE, COUNTR_LIB_PATH, DEFAULT_EVAL_JSON
from utils import load_image, run_one_image

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
    parser.add_argument("--boxes", default=None, help="Bounding boxes in [[[x1,y1],[x2,y2]],...] format for single image mode.")
    parser.add_argument("--results_path", default="run", help="Path for saving results.")
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
    formatted_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results = []

    if args.input == "eval":  # Evaluation mode
        with open(DEFAULT_EVAL_JSON, "r") as file:
            data = json.load(file)
        for key, value in data.items():
            if value:
                print(key)
                im_path = os.path.join("data", "images", key)
                samples, boxes, pos = load_image(im_path, value)
                samples, boxes = samples.unsqueeze(0).to(device), boxes.unsqueeze(0).to(device)
                result, elapsed_time, points = run_one_image(samples, boxes, pos, model, os.path.basename(im_path), f"run_{formatted_time}")
                print(result, elapsed_time.duration)
                results.append(result)
    else:  # Single image mode
        im_path = args.input
        samples, boxes, pos = load_image(im_path, eval(args.boxes))
        samples, boxes = samples.unsqueeze(0).to(device), boxes.unsqueeze(0).to(device)
        result, elapsed_time, points = run_one_image(samples, boxes, pos, model, os.path.basename(im_path), f"run_{formatted_time}")
        print(result, elapsed_time.duration)
        results.append(result)

        # Save prediction tensor
        tensor_np = points.cpu().numpy()
        np.savetxt(f"run_{formatted_time}/tensor.txt", tensor_np, fmt='%f')

    # Save count results
    with open(f"run_{formatted_time}/result.txt", "w") as file:
        for element, points in zip(results, points):
            file.write(f"{element}\n")

if __name__ == "__main__":
    main()
