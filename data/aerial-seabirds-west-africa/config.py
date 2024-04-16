import os

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
