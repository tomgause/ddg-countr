import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from config import BIRDS_PNT_PATH


def parse_option():
    parser = argparse.ArgumentParser(description="DDG CounTR Addon")
    parser.add_argument("--input", help="Path to evaluation run.", required=True)
    parser.add_argument("--pnt", help="Path to birds pnt file.", default=BIRDS_PNT_PATH)
    return parser.parse_args()


def main():
    args = parse_option()

    # Load birds data
    with open(args.pnt, "r") as file:
        birds_data = json.load(file)

    directory_path = os.path.join(args.input, "countr", "images")
    file_path = os.path.join(args.input, "countr", "result.txt")

    # Read results and list JPEG files
    with open(file_path, "r") as file:
        lines = file.readlines()
    jpg_files = [f for f in os.listdir(directory_path) if f.endswith(".jpg")]

    if len(jpg_files) != len(lines):
        print(
            "Warning: The number of JPEG files and lines in results.txt do not match."
        )

    # Data extraction
    gt_values = [
        len(birds_data["points"][jpg_file]["bird"])
        for jpg_file in jpg_files
        if jpg_file in birds_data["points"]
    ]
    pred_values = [float(line.strip().split(",")[0]) for line in lines]
    num_boxes = [float(line.strip().split(",")[1]) for line in lines]

    # Calculate overall RMSE
    differences = np.abs(np.array(gt_values) - np.array(pred_values))
    overall_rmse = np.sqrt(np.mean(differences**2))

    print(f"Overall RMSE: {overall_rmse}")
    print(f"Total GT: {sum(gt_values)}")
    print(f"Total pred: {sum(pred_values)}")
    print(f"Error: {abs(sum(pred_values) - sum(gt_values))}")

    # RMSE by number of boxes
    rmse_by_num_boxes = defaultdict(list)
    for i, num_box in enumerate(num_boxes):
        rmse_by_num_boxes[num_box].append(differences[i])

    rmse_per_num_boxes = {
        num_box: np.sqrt(np.mean(np.array(differences_list) ** 2))
        for num_box, differences_list in rmse_by_num_boxes.items()
    }

    # Plotting
    threshold = 1
    num_boxes_filtered, rmse_filtered = zip(
        *[
            (num_box, rmse)
            for num_box, rmse in rmse_per_num_boxes.items()
            if num_box >= threshold
        ]
    )
    num_boxes_sorted, rmse_sorted = zip(*sorted(zip(num_boxes_filtered, rmse_filtered)))

    plt.figure(figsize=(10, 6))
    plt.plot(num_boxes_sorted, rmse_sorted, marker="o", linestyle="-")
    plt.title("RMSE vs. Number of Boxes")
    plt.xlabel("Number of Boxes")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.savefig(os.path.join(args.input, "boxes_loss.png"))


if __name__ == "__main__":
    main()
