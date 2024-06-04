import os
import random
from PIL import Image
import os.path
import torch
import torchvision
import torchvision.transforms as trn
from torchvision.utils import save_image
import torch.utils.data as data
import numpy as np
import PIL
import collections
import argparse
from corruptions import (
    gaussian_noise,
    shot_noise,
    impulse_noise,
    defocus_blur,
    zoom_blur,
    brightness,
    contrast,
    elastic_transform,
    jpeg_compression,
    speckle_noise,
    gaussian_blur,
    spatter,
    saturate,
)

IMG_SIZE = 640
IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm"]

from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as transforms
import torch.nn.functional as F


def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class Read_Dataset(data.Dataset):
    def __init__(self, root, avoid_subfolder=None, transform=True):
        self.root = root
        self.transform = transform
        self.avoid_subfolder = (
            os.path.abspath(avoid_subfolder) if avoid_subfolder is not None else None
        )
        self.resize = transforms.Resize((640, 640))
        self.pil_to_tensor = transforms.PILToTensor()

        # avoid subfolder
        self.image_paths = []
        for dirpath, _, filenames in os.walk(root):
            if self.avoid_subfolder and self.avoid_subfolder in os.path.abspath(
                dirpath
            ):
                continue
            for filename in filenames:
                if filename.lower().endswith(("png", "jpg", "jpeg")):
                    self.image_paths.append(os.path.join(dirpath, filename))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:

            h, w = image.size
            if h < w:  # se verticale
                padding = ((w - h) // 2, (w - h) - (w - h) // 2, 0, 0)
            else:  # se orizzontale
                padding = (0, 0, (h - w) // 2, (h - w) - (h - w) // 2)

            image = self.pil_to_tensor(image)
            image = image / 255
            image = F.pad(image, padding, mode="constant", value=0.5)
            image = self.resize(image)

        return image


class DistortImageFolder(data.Dataset):
    def __init__(
        self,
        root,
        methods,
        avoid_subfolder,
        transform=True,
        loader=default_loader,
        severity=2,
        debug=False,
    ):
        self.root = root
        self.methods = methods
        self.transform = transform
        self.loader = loader
        self.avoid_subfolder = os.path.abspath(avoid_subfolder)
        self.severity = severity
        self.debug = debug
        self.resize = transforms.Resize((IMG_SIZE, IMG_SIZE))

        # self.image_paths = [os.path.join(root, f) for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
        self.image_paths = []
        for dirpath, _, filenames in os.walk(root):
            if self.avoid_subfolder in os.path.abspath(dirpath):
                continue
            for filename in filenames:
                if filename.lower().endswith(("png", "jpg", "jpeg")):
                    self.image_paths.append(os.path.join(dirpath, filename))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        original_image = self.loader(image_path)

        method_name, method = random.choice(list(self.methods.items()))

        corrupted_image = method(original_image, severity=self.severity)

        if (
            type(corrupted_image) == PIL.JpegImagePlugin.JpegImageFile
            or type(corrupted_image) == PIL.Image.Image
        ):
            corrupted_torch = trn.ToTensor()(corrupted_image)

        if type(corrupted_image) == np.ndarray:
            corrupted_torch = torch.from_numpy(corrupted_image)
            corrupted_torch = corrupted_torch.permute(2, 0, 1)
            corrupted_torch = corrupted_torch.float() / 255

        # Apply padding
        if self.transform:
            w, h = corrupted_torch.shape[1], corrupted_torch.shape[2]
            if h < w:  # se verticale
                padding = ((w - h) // 2, (w - h) - (w - h) // 2, 0, 0)
            else:  # se orizzontale
                padding = (0, 0, (h - w) // 2, (h - w) - (h - w) // 2)

            corrupted_torch = F.pad(
                corrupted_torch, padding, mode="constant", value=0.5
            )
            corrupted_torch = self.resize(corrupted_torch)

        if self.debug and idx % 100 == 0:
            print(idx, method_name)
            rel_path = os.path.relpath(image_path, self.root)
            corrupted_image_path = os.path.join("corrupted_dataset_debug", rel_path)
            os.makedirs(os.path.dirname(corrupted_image_path), exist_ok=True)
            torchvision.utils.save_image(corrupted_torch, corrupted_image_path)

        return corrupted_torch


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="Calculate FID for corrupted images.")
    parser.add_argument(
        "--severity", type=int, default=3, help="Severity level for image corruption"
    )
    args = parser.parse_args()

    d = collections.OrderedDict()
    d["Gaussian Noise"] = gaussian_noise
    d["Shot Noise"] = shot_noise
    d["Impulse Noise"] = impulse_noise
    d["Defocus Blur"] = defocus_blur
    # d['Glass Blur'] = glass_blur
    # d['Motion Blur'] = motion_blur
    d["Zoom Blur"] = zoom_blur
    # d['Snow'] = snow
    # d['Frost'] = frost
    # d['Fog'] = fog
    d["Brightness"] = brightness
    d["Contrast"] = contrast
    d["Elastic"] = elastic_transform
    # d['Pixelate'] = pixelate
    d["JPEG"] = jpeg_compression

    d["Speckle Noise"] = speckle_noise
    d["Gaussian Blur"] = gaussian_blur
    d["Spatter"] = spatter
    d["Saturate"] = saturate

    original_dataset = Read_Dataset(
        root="/home/e3da/code/validate_robustness/recipes/object_detection/datasets/VOC/images",
        avoid_subfolder="/home/e3da/code/validate_robustness/recipes/object_detection/datasets/VOC/images/VOCdevkit",
    )

    corrupted_dataset = DistortImageFolder(
        root="/home/e3da/code/micromind/recipes/object_detection/datasets/VOC/images",
        methods=d,
        avoid_subfolder="/home/e3da/code/micromind/recipes/object_detection/datasets/VOC/images/VOCdevkit/",
        severity=args.severity,
    )

    original_data_loader = data.DataLoader(
        original_dataset, batch_size=100, shuffle=True, num_workers=4
    )
    corrupted_data_loader = data.DataLoader(
        corrupted_dataset, batch_size=100, shuffle=True, num_workers=4
    )

    original_batch = next(iter(original_data_loader))
    corrupted_batch = next(iter(corrupted_data_loader))
    # print("batch shape original: ", original_batch.shape)
    # print("batch shape original: ", corrupted_batch.shape)
    # print("numero batch nel data loader: ", len(original_data_loader))
    # save_image(original_batch, "original_batch.jpg")
    # save_image(corrupted_batch, "corrupted_batch.jpg")

    _ = torch.manual_seed(123)
    fid = FrechetInceptionDistance(feature=64)#.to(device)

    print(f"Calculating FID with SEVERIY = {args.severity}")
    fids = []
    for _ in range(len(corrupted_data_loader)):
        original_batch = next(iter(original_data_loader))#.to(device)
        corrupted_batch = next(iter(corrupted_data_loader))#.to(device)

        original_batch = (original_batch * 255).to(torch.uint8)
        corrupted_batch = (corrupted_batch * 255).to(torch.uint8)

        fid.update(original_batch, real=True)
        fid.update(corrupted_batch, real=False)
        fid_value = fid.compute()
        fids.append(fid_value.item())
        print(f"batch: {_}, FID:", fid_value.item())

    mean_fid = sum(fids) / len(fids)
    print("FID: ", mean_fid)
    with open("fid_scores.txt", "a") as fid_file:
        fid_file.write(f"Severity: {args.severity}, Mean FID: {mean_fid:.4f}\n")
