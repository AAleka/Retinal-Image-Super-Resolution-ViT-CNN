import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ABDataset(Dataset):
    def __init__(self, root_a, transform=None):
        self.root_a = root_a
        self.transform = transform

        self.a_images = os.listdir(root_a)
        self.length_dataset = len(self.a_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        a_img = self.a_images[index % self.length_dataset]
        a_path = os.path.join(self.root_a, a_img)
        a_img = np.array(Image.open(a_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=a_img)
            a_img = augmentations["image"]

        return a_img
