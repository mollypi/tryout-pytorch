import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import pandas as pd
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)


    def __getitem__(self, idx):
        print(self.img_labels)
        print(self.img_dir)
        img_path = os.path.join(self.img_dir, self.img_labels['filepaths'][idx])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

if __name__ == '__main__':
    c = CustomImageDataset("/Users/md3282/Desktop/tryout-pytorch/data/cards.csv", "/Users/md3282/Desktop/tryout-pytorch/data/")
    print(c.__len__())
    print(c.__getitem__(3))