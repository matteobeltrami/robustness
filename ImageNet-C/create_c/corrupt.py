import os
import random
from PIL import Image
import os.path
import time
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as trn
import torch.utils.data as data
import numpy as np
import collections
import PIL
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
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


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
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class DistortImageFolder(data.Dataset):
    def __init__(self, root, methods, avoid_subfolder, transform=None, loader=default_loader):
        self.root = root
        self.methods = methods
        self.transform = transform
        self.loader = loader
        self.avoid_subfolder = os.path.abspath(avoid_subfolder)

        #self.image_paths = [os.path.join(root, f) for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
        self.image_paths = []
        for dirpath, _, filenames in os.walk(root):
            if self.avoid_subfolder in os.path.abspath(dirpath):
                continue
            for filename in filenames:
                if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.image_paths.append(os.path.join(dirpath, filename))
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        original_image = self.loader(image_path)

        method_name, method = random.choice(list(self.methods.items()))
        print(method_name)

        corrupted_image = method(original_image, severity=2)

        if type(corrupted_image) == PIL.JpegImagePlugin.JpegImageFile or type(corrupted_image) == PIL.Image.Image:
            corrupted_torch = trn.ToTensor()(corrupted_image)

        if type(corrupted_image) == np.ndarray:
            corrupted_torch = torch.from_numpy(corrupted_image)
            corrupted_torch = corrupted_torch.permute(2, 0, 1)
            corrupted_torch = corrupted_torch.float() / 255

        # Apply transformation if provided
        if self.transform is not None:
            corrupted_image = self.transform(corrupted_image)

        rel_path = os.path.relpath(image_path, self.root)
        corrupted_image_path = os.path.join("corrupted_dataset_test", rel_path)
        os.makedirs(os.path.dirname(corrupted_image_path), exist_ok=True)

        torchvision.utils.save_image(corrupted_torch, corrupted_image_path)

        return corrupted_torch

if __name__ == "__main__":
    
    d = collections.OrderedDict()
    d['Gaussian Noise'] = gaussian_noise
    d['Shot Noise'] = shot_noise
    d['Impulse Noise'] = impulse_noise
    d['Defocus Blur'] = defocus_blur
    # d['Glass Blur'] = glass_blur
    # d['Motion Blur'] = motion_blur
    d['Zoom Blur'] = zoom_blur
    # d['Snow'] = snow
    # d['Frost'] = frost
    # d['Fog'] = fog
    d['Brightness'] = brightness
    d['Contrast'] = contrast
    d['Elastic'] = elastic_transform
    # d['Pixelate'] = pixelate
    d['JPEG'] = jpeg_compression

    d['Speckle Noise'] = speckle_noise
    d['Gaussian Blur'] = gaussian_blur
    d['Spatter'] = spatter
    d['Saturate'] = saturate


    distorted_dataset = DistortImageFolder(
        root="/home/e3da/code/micromind/recipes/object_detection/datasets/VOC/images", 
        methods=d, 
        avoid_subfolder="/home/e3da/code/micromind/recipes/object_detection/datasets/VOC/images/VOCdevkit/", 
    )

    for _ in distorted_dataset: pass