import os
import pandas as pd
import rasterio
from rasterio.windows import Window
from PIL import Image
import json
from tqdm import tqdm


# Constants
CHUNK_SIZE = 576
OVERLAP = 32
IM_WIDTH_PX = 27569
IM_WIDTH_M = 292.48
IM_HEIGHT_PX = 28814

# Set up paths
base_dir = r"aerial-seabirds-west-africa"

# YOU MUST DOWNLOAD THESE TO CONTINUE. https://lila.science/datasets/aerial-seabirds-west-africa/
image_path = os.path.join(base_dir, "seabirds_rgb.tif")
labels_path = os.path.join(base_dir, "labels_birds_full.csv")

tif_dest_dir = os.path.join(base_dir, "image-chunks")
label_dest_dir = os.path.join(base_dir, "label-chunks")
jpg_dest_dir = os.path.join(base_dir, "image-chunks-jpg")
empty_pnt_path = "birds_empty.pnt"
annotated_pnt_path = os.path.join(base_dir, "birds.pnt")


def convert_spatial_to_pixel(x_spatial, y_spatial):
    conversion_factor = IM_WIDTH_PX / IM_WIDTH_M
    x_pixel = x_spatial * conversion_factor
    y_pixel = y_spatial * conversion_factor
    y_pixel = IM_HEIGHT_PX - y_pixel
    return x_pixel, y_pixel


def chunk_image(image_path, image_chunk_path):
    os.makedirs(image_chunk_path, exist_ok=True)
    with rasterio.open(image_path) as src:
        height_steps = range(0, src.height, CHUNK_SIZE - OVERLAP)
        width_steps = range(0, src.width, CHUNK_SIZE - OVERLAP)
        for j in tqdm(height_steps, desc="Chunking images"):
            for i in width_steps:
                window = Window(i, j, CHUNK_SIZE, CHUNK_SIZE)
                transform = src.window_transform(window)
                out_image = src.read(window=window)
                out_meta = src.meta.copy()
                out_meta.update(
                    {
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": transform,
                        "compress": "lzw",
                    }
                )
                out_path = os.path.join(image_chunk_path, f"{i}_{j}.tif")
                with rasterio.open(out_path, "w", **out_meta) as dest:
                    dest.write(out_image)


def chunk_labels(labels_path, label_chunk_path, image_shape):
    labels = pd.read_csv(labels_path)
    labels[["X_pixel", "Y_pixel"]] = labels.apply(
        lambda row: convert_spatial_to_pixel(row["X"], row["Y"]),
        axis=1,
        result_type="expand",
    )
    os.makedirs(label_chunk_path, exist_ok=True)
    height_steps = range(0, image_shape[0], CHUNK_SIZE - OVERLAP)
    width_steps = range(0, image_shape[1], CHUNK_SIZE - OVERLAP)
    for j in tqdm(height_steps, desc="Processing label chunks"):
        for i in width_steps:
            chunk_labels = labels[
                (labels["X_pixel"] >= i)
                & (labels["X_pixel"] < i + CHUNK_SIZE)
                & (labels["Y_pixel"] >= j)
                & (labels["Y_pixel"] < j + CHUNK_SIZE)
            ].copy()
            chunk_labels["X_pixel"] -= i
            chunk_labels["Y_pixel"] -= j
            chunk_labels.to_csv(
                os.path.join(label_chunk_path, f"{i}_{j}.csv"), index=False
            )


def convert_tif_to_jpg(tif_dest_dir, jpg_dest_dir):
    os.makedirs(jpg_dest_dir, exist_ok=True)
    tif_files = [f for f in os.listdir(tif_dest_dir) if f.endswith(".tif")]
    for filename in tqdm(tif_files, desc="Converting TIF to JPG"):
        img_path = os.path.join(tif_dest_dir, filename)
        img = Image.open(img_path)
        jpg_filename = os.path.splitext(filename)[0] + ".jpg"
        jpg_path = os.path.join(jpg_dest_dir, jpg_filename)
        img.save(jpg_path, "JPEG")


def build_pnt(empty_pnt_path, csv_dir, jpg_dir, annotated_pnt_path):
    with open(empty_pnt_path, "r") as f:
        data = json.load(f)
    data["points"] = {}
    jpg_files = [f for f in os.listdir(jpg_dir) if f.endswith(".jpg")]
    for jpg_file in tqdm(jpg_files, desc="Building .pnt file"):
        csv_file = os.path.join(csv_dir, jpg_file.replace(".jpg", ".csv"))
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            points = [
                {"x": row["X_pixel"], "y": row["Y_pixel"]}
                for index, row in df.iterrows()
            ]
            data["points"][jpg_file] = {"bird": points}
    with open(annotated_pnt_path, "w") as f:
        json.dump(data, f, indent=4)


print("Processing Aerial Seabirds West Africa Dataset.")
with rasterio.open(image_path) as src:
    image_shape = (src.height, src.width)
print(f"TIF Shape: {image_shape}")

chunk_labels(labels_path, label_dest_dir, image_shape)

chunk_image(image_path, tif_dest_dir)

convert_tif_to_jpg(tif_dest_dir, jpg_dest_dir)

build_pnt(empty_pnt_path, label_dest_dir, jpg_dest_dir, annotated_pnt_path)
