import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MVTecDataset(Dataset):
    def __init__(self, root_dir, category, split="train", transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform

        data_dir = os.path.join(root_dir, category)

        if split == "train":
            self.image_paths = sorted(
                glob.glob(os.path.join(data_dir, "train", "good", "*.png"))
            )

        elif split == "test":
            self.image_paths = []
            self.labels = []
            self.mask_paths = []

            test_dir = os.path.join(data_dir, "test")
            for subdir in sorted(os.listdir(test_dir)):
                subdir_path = os.path.join(test_dir, subdir)
                if not os.path.isdir(subdir_path):
                    continue

                is_anomaly = subdir != "good"

                for img_path in sorted(glob.glob(os.path.join(subdir_path, "*.png"))):
                    self.image_paths.append(img_path)
                    self.labels.append(1 if is_anomaly else 0)

                    if is_anomaly:
                        img_name = os.path.splitext(os.path.basename(img_path))[0]
                        mask_path = os.path.join(
                            data_dir, "ground_truth", subdir, f"{img_name}_mask.png"
                        )
                        self.mask_paths.append(mask_path if os.path.exists(mask_path) else None)
                    else:
                        self.mask_paths.append(None)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        if self.split == "train":
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(0, dtype=torch.long)

        label = self.labels[idx]
        mask_path = self.mask_paths[idx]

        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.fromarray(np.zeros((h, w), dtype=np.uint8))

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask, torch.tensor(label, dtype=torch.long)


def get_dataloaders(root_dir, category, img_size=256, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    train_dataset = MVTecDataset(root_dir, category, split="train",
                                  transform=transform, mask_transform=mask_transform)
    test_dataset = MVTecDataset(root_dir, category, split="test",
                                 transform=transform, mask_transform=mask_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
