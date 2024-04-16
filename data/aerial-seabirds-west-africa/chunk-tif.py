import os
import pandas as pd
import rasterio
from rasterio.windows import Window
from PIL import Image
import json
import random
import matplotlib.pyplot as plt
import shutil


# Helper function to convert spatial coordinates to pixel coordinates
def convert_spatial_to_pixel(x_spatial, y_spatial, im_width_px, im_width_m, im_height_px):
    conversion_factor = im_width_px / im_width_m
    x_pixel = x_spatial * conversion_factor
    y_pixel = im_height_px - (y_spatial * conversion_factor)
    return int(x_pixel), int(y_pixel)


# Function to chunk large images into smaller tiles
def chunk_image(image_path, image_chunk_path, chunk_size=576, overlap=32):
    os.makedirs(image_chunk_path, exist_ok=True)
    with rasterio.open(image_path) as src:
        for j in range(0, src.height, chunk_size - overlap):
            for i in range(0, src.width, chunk_size - overlap):
                window = Window(i, j, chunk_size, chunk_size)
                transform = src.window_transform(window)
                out_image = src.read(window=window)
                out_meta = src.meta.copy()
                out_meta.update({
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": transform,
                    "compress": "lzw",
                })
                out_path = os.path.join(image_chunk_path, f"{i}_{j}.tif")
                with rasterio.open(out_path, "w", **out_meta) as dest:
                    dest.write(out_image)


# Function to chunk corresponding labels for the image chunks
def chunk_labels(labels_path, label_chunk_path, chunk_size, overlap, im_width_px, im_width_m, im_height_px):
    os.makedirs(label_chunk_path, exist_ok=True)
    labels = pd.read_csv(labels_path)
    labels[["X_pixel", "Y_pixel"]] = labels.apply(
        lambda row: convert_spatial_to_pixel(row["X"], row["Y"], im_width_px, im_width_m, im_height_px),
        axis=1,
        result_type="expand"
    )
    with rasterio.open(image_path) as src:
        for j in range(0, src.height, chunk_size - overlap):
            for i in range(0, src.width, chunk_size - overlap):
                chunk_labels = labels[
                    (labels["X_pixel"] >= i) & (labels["X_pixel"] < i + chunk_size) &
                    (labels["Y_pixel"] >= j) & (labels["Y_pixel"] < j + chunk_size)
                ].copy()
                chunk_labels["X_pixel"] -= i
                chunk_labels["Y_pixel"] -= j
                chunk_labels.to_csv(os.path.join(label_chunk_path, f"{i}_{j}.csv"), index=False)

# Main function to execute the script
def main(image_path, labels_path):
    image_chunk_path = image_path.replace(".tif", "-chunks")
    label_chunk_path = labels_path.replace(".csv", "-chunks")

    with rasterio.open(image_path) as src:
        im_width_px, im_width_m, im_height_px = src.width, src.width * 0.01, src.height
        chunk_labels(labels_path, label_chunk_path, 576, 32, im_width_px, im_width_m, im_height_px)
        chunk_image(image_path, image_chunk_path, 576, 32)


# Example of running the script
if __name__ == "__main__":
    image_path = "path_to_your_tif_file.tif"
    labels_path = "path_to_your_labels_file.csv"
    main(image_path, labels_path)
