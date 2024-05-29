import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

from torchvision.utils import save_image

from PIL import Image
import os


class Read_Dataset(data.Dataset):
    def __init__(self, root, avoid_subfolder=None, transform=True):
        self.root = root
        self.transform = transform
        self.avoid_subfolder = os.path.abspath(avoid_subfolder) if avoid_subfolder is not None else None
        self.resize = transforms.Resize((640, 640))
        self.pil_to_tensor = transforms.PILToTensor()

        # avoid subfolder
        self.image_paths = []
        for dirpath, _, filenames in os.walk(root):
            if self.avoid_subfolder and self.avoid_subfolder in os.path.abspath(dirpath):
                continue
            for filename in filenames:
                if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.image_paths.append(os.path.join(dirpath, filename))
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:

            h, w = image.size
            if h < w: # se verticale
                padding = ((w - h) // 2, (w - h) - (w - h) // 2, 0, 0)
            else:     # se orizzontale
                padding = (0, 0, (h - w) // 2, (h - w) - (h - w) // 2)

            image = self.pil_to_tensor(image)
            image = image / 255
            image = F.pad(image, padding, mode='constant', value=0.5)
            image = self.resize(image)

        return image


if __name__ == "__main__":
    _ = torch.manual_seed(123)
    fid = FrechetInceptionDistance(feature=64)

    original_dataset = Read_Dataset(
        root="/home/e3da/code/validate_robustness/recipes/object_detection/datasets/VOC/images",
        avoid_subfolder="/home/e3da/code/validate_robustness/recipes/object_detection/datasets/VOC/images/VOCdevkit",
    )
    corrupted_dataset = Read_Dataset(
        root="/home/e3da/code/validate_robustness/recipes/object_detection/datasets/corruptedVOC/images",
        avoid_subfolder=None,
    )

    original_data_loader = data.DataLoader(
        original_dataset, batch_size=100, shuffle=True, num_workers=4
    )
    corrupted_data_loader = data.DataLoader(
        corrupted_dataset, batch_size=100, shuffle=True, num_workers=4
    )

    original_batch = next(iter(original_data_loader))
    corrupted_batch = next(iter(corrupted_data_loader))
    print("batch shape original: ", original_batch.shape)
    print("batch shape original: ", corrupted_batch.shape)
    print("numero batch nel data loader: ", len(original_data_loader))

    #save_image(corrupted_batch, "img.png")

    for _ in range(len(corrupted_data_loader)-1):
        original_batch = next(iter(original_data_loader))
        corrupted_batch = next(iter(corrupted_data_loader))

        original_batch = (original_batch * 255).to(torch.uint8)
        corrupted_batch = (corrupted_batch * 255).to(torch.uint8)

        fid.update(original_batch, real=True)
        fid.update(corrupted_batch, real=False)
        fid_value = fid.compute()
        print(fid_value)
