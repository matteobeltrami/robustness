import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

from torchvision.utils import save_image

from PIL import Image
import os


class Read_Dataset(data.Dataset):
    def __init__(self, root, avoid_subfolder, transform=True):
        self.root = root
        self.transform = transform
        self.avoid_subfolder = os.path.abspath(avoid_subfolder)
        self.resize = transforms.Resize((640, 640))
        self.pil_to_tensor = transforms.PILToTensor()

        # self.image_paths = [os.path.join(root, f) for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
        # avoid subfolder
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

    dataset = Read_Dataset(
        root="/home/e3da/code/robustness/ImageNet-C/create_c/corrupted_dataset",
        avoid_subfolder="/home/e3da/code/robustness/ImageNet-C/create_c/corrupted_dataset/val2012",
    )

    data_loader = data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=4
    )

    for img in data_loader:
        image = img
        print(img.shape)
        break

    save_image(image, "img.png")

    breakpoint()

    # generate two slightly overlapping image intensity distributions
    imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
    imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
    breakpoint()
    fid.update(imgs_dist1, real=True)
    fid.update(imgs_dist2, real=False)
    fid_value = fid.compute()
    print(fid_value)
    breakpoint()