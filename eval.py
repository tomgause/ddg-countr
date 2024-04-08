import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from config import BIRDS_PNT_PATH

def parse_option():
    parser = argparse.ArgumentParser(description="DDG CounTR Addon", add_help=False)
    parser.add_argument(
        "--input", help="Path to evaluation run.", required=True
    )
    parser.add_argument(
        "--pnt", help="Path to birds pnt file.", default=BIRDS_PNT_PATH
    )

    return parser.parse_args()


args = parse_option()

with open(args.pnt, "r") as file:
    birds_data = json.load(file)
directory_path = os.path.join(args.input, "images")
file_path = os.path.join(args.input, "result.txt")
with open(file_path, 'r') as file:
    lines = file.readlines()
jpg_files = [f for f in os.listdir(directory_path) if f.endswith('.jpg')]
if len(jpg_files) != len(lines):
    print("Warning: The number of JPEG files and lines in results.txt do not match.")


gt_values = [len(birds_data['points'][jpg_file]['bird']) for jpg_file in jpg_files]
pred_values = [float(line.strip()) for line in lines]

# Compute differences and RMSE
differences = np.abs(np.array(gt_values) - np.array(pred_values))
rmse = np.sqrt(np.mean(differences ** 2))

print(f"RMSE: {rmse}")

# Create a histogram of the absolute differences
plt.hist(differences, bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Absolute Differences')
plt.xlabel('Absolute Difference')
plt.ylabel('Frequency')
plt.savefig('absolute_differences_histogram.png')
