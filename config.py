import os

# Paths
COUNTR_LIB_PATH = "/mnt/c/Users/tomsg/Work/open-source/DotDotGoose/CounTR"
COUNTR_MODEL_PATH = os.path.join(COUNTR_LIB_PATH, "FSC147.pth")
BIRDS_PNT_PATH = "data/aerial-seabirds-west-africa/aerial-seabirds-west-africa/birds.pnt"
DINOV_LIB_PATH = "/mnt/c/Users/tomsg/Work/leverege/tpi/core-gap/DINOv"
DINOV_CKPT_PATH = "/mnt/c/Users/tomsg/Work/leverege/tpi/core-gap/DINOv/model_swinL.pth"
IMAGE_DIR = "data/aerial-seabirds-west-africa/aerial-seabirds-west-africa/image-chunks-jpg"

# Virtual environment activation commands
ACTIVATE_DINOV = "source /mnt/c/Users/tomsg/Work/leverege/tpi/core-gap/DINOv/dinov/.venv/bin/activate"
ACTIVATE_COUNTR = "source /mnt/c/Users/tomsg/Work/open-source/DotDotGoose/CounTR/.venv/bin/activate"

# Evaluation defaults
DEFAULT_EVAL_JSON = "data/converted_data.json"
DEFAULT_DEVICE = "cuda"

# Image processing settings
CIRCLE_RADIUS = 15
COLOR = (255, 255, 255, 128)  # Semi-transparent white
BOX_THRESHOLD = 1/8
MAX_COUNT = 1000
