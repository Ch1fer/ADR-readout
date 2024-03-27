import torch
from torch.utils import data
from torch.utils.data import Dataset
import torch as nn
import pandas as pd
import numpy as np
import torchvision as tv
import cv2 as cv
import os
import matplotlib.pyplot as plt
from PIL import Image


class CustomDataset(Dataset):  # My custom dataset class
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


image_dir = 'data/images'
label_dir = 'data/label.csv'
countsOfRows = 100


"""___PREPARE_DATA_AND_TRANSFORM___"""
files = os.listdir(image_dir)  # Load images
images = []
fileCounter = 0
for file in files:
    if fileCounter >= countsOfRows:
        break
    fileCounter += 1

    if file.endswith('.jpg'):
        image_path = os.path.join(image_dir, file)
        img = cv.imread(image_path)
        img = cv.resize(img, (64, 64), interpolation=cv.INTER_AREA)  # resize image
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # transform images to gray scale, only one channel of gray
        clahe = cv.createCLAHE(clipLimit=0.05, tileGridSize=(4, 4))  # increase image contrast
        img = clahe.apply(img)
        images.append(img)

labels = pd.read_csv(label_dir, nrows=countsOfRows)  # Load labels
labels = torch.tensor(labels.values, dtype=torch.int64)

transform = tv.transforms.ToTensor()
myDataset = CustomDataset(images, labels, transform=transform)  # Create custom dataset from previously images & labels

row = 5
plt.imshow(myDataset[row][0].numpy()[0], cmap="gray")  # picture
plt.xlabel(f"{myDataset[row][1][0]} : {myDataset[row][1][1]}")  # label
plt.show()  # visualise the first picture and label


"""___Dataloader___"""
batch_size = 16
dataloader = torch.utils.data.DataLoader(myDataset, batch_size=batch_size, shuffle=True)

for img, l in dataloader:
    print(img.shape)
    print(l.shape)
    break


# """___MODEL___"""
# class MyModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         linear1 = nn.Linear(3 * 300 * 300, )  # 270k :)
