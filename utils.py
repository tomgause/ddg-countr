
import os
import sys
import time
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import timm

from config import COUNTR_LIB_PATH

# Append the path for custom libraries
sys.path.append(COUNTR_LIB_PATH)

from util.misc import make_grid

# Check timm version compatibility
assert "0.4.5" <= timm.__version__ <= "0.4.9", "timm version must be between 0.4.5 and 0.4.9"

device = torch.device("cuda")

class MeasureTime:
    """Context manager for measuring execution time."""
    def __enter__(self):
        self.start = time.perf_counter_ns()
        return self

    def __exit__(self, typ, value, traceback):
        self.duration = time.perf_counter_ns() - self.start

def load_image(im_path, bboxes):
    image = Image.open(im_path)
    image.load()
    W, H = image.size

    # Resize the image size so that the height is 384
    new_H = 384
    new_W = 16 * int((W / H * 384) / 16)
    scale_factor_H = float(new_H) / H
    scale_factor_W = float(new_W) / W
    image = transforms.Resize((new_H, new_W))(image)
    Normalize = transforms.Compose([transforms.ToTensor()])
    image = Normalize(image)

    boxes, rects = None, None

    if bboxes:
        boxes = list()
        rects = list()
        for bbox in bboxes:
            x1 = int(bbox[0][0] * scale_factor_W)
            y1 = int(bbox[0][1] * scale_factor_H)
            x2 = int(bbox[1][0] * scale_factor_W)
            y2 = int(bbox[1][1] * scale_factor_H)
            rects.append([y1, x1, y2, x2])
            bbox = image[:, y1 : y2 + 1, x1 : x2 + 1]
            bbox = transforms.Resize((64, 64))(bbox)
            boxes.append(bbox.numpy())

        boxes = np.array(boxes)
        boxes = torch.Tensor(boxes)

    return image, boxes, rects


def run_one_image(samples, boxes, pos, model, save_image_name=None, save_dir="run"):
    _, _, h, w = samples.shape

    s_cnt = 0
    for rect in pos:
        if rect[2] - rect[0] < 10 and rect[3] - rect[1] < 10:
            s_cnt += 1
    if False: #  s_cnt >= 100:  # TODO: determine effectiveness of 3x3 run mode
        r_densities = []
        r_images = []
        r_images.append(TF.crop(samples[0], 0, 0, int(h / 3), int(w / 3)))  # 1
        r_images.append(TF.crop(samples[0], 0, int(w / 3), int(h / 3), int(w / 3)))  # 3
        r_images.append(
            TF.crop(samples[0], 0, int(w * 2 / 3), int(h / 3), int(w / 3))
        )  # 7
        r_images.append(TF.crop(samples[0], int(h / 3), 0, int(h / 3), int(w / 3)))  # 2
        r_images.append(
            TF.crop(samples[0], int(h / 3), int(w / 3), int(h / 3), int(w / 3))
        )  # 4
        r_images.append(
            TF.crop(samples[0], int(h / 3), int(w * 2 / 3), int(h / 3), int(w / 3))
        )  # 8
        r_images.append(
            TF.crop(samples[0], int(h * 2 / 3), 0, int(h / 3), int(w / 3))
        )  # 5
        r_images.append(
            TF.crop(samples[0], int(h * 2 / 3), int(w / 3), int(h / 3), int(w / 3))
        )  # 6
        r_images.append(
            TF.crop(samples[0], int(h * 2 / 3), int(w * 2 / 3), int(h / 3), int(w / 3))
        )  # 9

        pred_cnt = 0
        with MeasureTime() as et:
            for r_image in r_images:
                r_image = transforms.Resize((h, w))(r_image).unsqueeze(0)
                density_map = torch.zeros([h, w])
                density_map = density_map.to(device, non_blocking=True)
                start = 0
                prev = -1
                with torch.no_grad():
                    while start + 383 < w:
                        (output,) = model(
                            r_image[:, :, :, start : start + 384], boxes, 3
                        )
                        output = output.squeeze(0)
                        b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
                        d1 = b1(output[:, 0 : prev - start + 1])
                        b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
                        d2 = b2(output[:, prev - start + 1 : 384])

                        b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
                        density_map_l = b3(density_map[:, 0:start])
                        density_map_m = b1(density_map[:, start : prev + 1])
                        b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                        density_map_r = b4(density_map[:, prev + 1 : w])

                        density_map = (
                            density_map_l
                            + density_map_r
                            + density_map_m / 2
                            + d1 / 2
                            + d2
                        )

                        prev = start + 383
                        start = start + 128
                        if start + 383 >= w:
                            if start == w - 384 + 128:
                                break
                            else:
                                start = w - 384

                pred_cnt += torch.sum(density_map / 60).item()
                r_densities += [density_map]
    else:
        density_map = torch.zeros([h, w])
        density_map = density_map.to(device, non_blocking=True)
        start = 0
        prev = -1
        with MeasureTime() as et:
            with torch.no_grad():
                while start + 383 < w:
                    (output,) = model(
                        samples[:, :, :, start : start + 384], boxes, len(boxes)
                    )
                    output = output.squeeze(0)
                    b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
                    d1 = b1(output[:, 0 : prev - start + 1])
                    b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
                    d2 = b2(output[:, prev - start + 1 : 384])

                    b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
                    density_map_l = b3(density_map[:, 0:start])
                    density_map_m = b1(density_map[:, start : prev + 1])
                    b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                    density_map_r = b4(density_map[:, prev + 1 : w])

                    density_map = (
                        density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2
                    )

                    prev = start + 383
                    start = start + 128
                    if start + 383 >= w:
                        if start == w - 384 + 128:
                            break
                        else:
                            start = w - 384

            pred_cnt = torch.sum(density_map / 60).item()

    fig = samples[0]
    box_map = torch.zeros([fig.shape[1], fig.shape[2]], device=device)
    marked_points = []  # To store the coordinates of all marked points

    for rect in pos:
        for i in range(rect[2] - rect[0] + 1):
            y_min = min(rect[0] + i, fig.shape[1] - 1)
            for x in [rect[1], rect[3]]:
                x_clamped = min(x, fig.shape[2] - 1)
                box_map[y_min, x_clamped] = 10
                if [x_clamped, y_min] not in marked_points:
                    marked_points.append([x_clamped, y_min])

        for i in range(rect[3] - rect[1] + 1):
            x_min = min(rect[1] + i, fig.shape[2] - 1)
            for y in [rect[0], rect[2]]:
                y_clamped = min(y, fig.shape[1] - 1)
                box_map[y_clamped, x_min] = 10
                if [x_min, y_clamped] not in marked_points:
                    marked_points.append([x_min, y_clamped])

    box_map = box_map.unsqueeze(0).repeat(3, 1, 1)
    pred = (density_map.unsqueeze(0).repeat(3, 1, 1) if s_cnt < 1 else make_grid(r_densities, h, w).unsqueeze(0).repeat(3, 1, 1))

    os.makedirs(save_dir, exist_ok=True)
    if save_image_name:
        image_dir = os.path.join(save_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        fig = fig + box_map + pred / 2
        fig = torch.clamp(fig, 0, 1)
        torchvision.utils.save_image(fig, os.path.join(image_dir, save_image_name))

    # GT map needs coordinates for all GT dots, which is hard to input and is not a must for the demo. You can provide it yourself.
    return pred_cnt, et, pred
