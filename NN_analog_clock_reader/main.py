import torch
from torch.utils import data
from torch.utils.data import Dataset
import torch.nn as nn
import pandas as pd
import torchvision as tv
import cv2 as cv
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


def show_8_samples(dataset, size, start):
    if size <= start + 8:
        print("wrong start!")
        return
    plt.subplot(2, 4, 1)
    plt.imshow(dataset[start][0].permute(1, 2, 0).numpy())
    plt.xlabel(f"{dataset[start][1][0]} : {dataset[start][1][1]}")

    plt.subplot(2, 4, 2)
    plt.imshow(dataset[start + 1][0].permute(1, 2, 0).numpy())
    plt.xlabel(f"{dataset[start + 1][1][0]} : {dataset[start + 1][1][1]}")

    plt.subplot(2, 4, 3)
    plt.imshow(dataset[start + 2][0].permute(1, 2, 0).numpy())
    plt.xlabel(f"{dataset[start + 2][1][0]} : {dataset[start + 2][1][1]}")

    plt.subplot(2, 4, 4)
    plt.imshow(dataset[start + 3][0].permute(1, 2, 0).numpy())
    plt.xlabel(f"{dataset[start + 3][1][0]} : {dataset[start + 3][1][1]}")

    plt.subplot(2, 4, 5)
    plt.imshow(dataset[start + 4][0].permute(1, 2, 0).numpy())
    plt.xlabel(f"{dataset[start + 4][1][0]} : {dataset[start + 4][1][1]}")

    plt.subplot(2, 4, 6)
    plt.imshow(dataset[start + 5][0].permute(1, 2, 0).numpy())
    plt.xlabel(f"{dataset[start + 5][1][0]} : {dataset[start + 5][1][1]}")

    plt.subplot(2, 4, 7)
    plt.imshow(dataset[start + 6][0].permute(1, 2, 0).numpy())
    plt.xlabel(f"{dataset[start + 6][1][0]} : {dataset[start + 6][1][1]}")

    plt.subplot(2, 4, 8)
    plt.imshow(dataset[start + 7][0].permute(1, 2, 0).numpy())
    plt.xlabel(f"{dataset[start + 7][1][0]} : {dataset[start + 7][1][1]}")
    plt.show()
# function for displaying 8 photos after transformation

class CustomDataset(Dataset):
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
# My custom dataset class

image_dir = 'data/images'
label_dir = 'data/label.csv'
countsOfRows = 1000
resizeVal = 64


"""___PREPARE_DATA_AND_TRANSFORM___"""
files = os.listdir(image_dir)  # Load images
images = []
fileCounter = 0
for file in files:
    if fileCounter >= countsOfRows:
        break
    fileCounter += 1

    if file.endswith('.jpg'):
        image_path = os.path.join(image_dir, file)  # create directory path
        img = cv.imread(image_path)
        img = cv.resize(img, (resizeVal, resizeVal), interpolation=cv.INTER_AREA)  # resize image
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # transform images to gray scale, only one channel of gray
        images.append(img)


labels = pd.read_csv(label_dir, nrows=countsOfRows)  # Load labels
labels = torch.tensor(labels.values, dtype=torch.float32)  # transform to tensor

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=12)
# dividing the dataset into training and test

transform = tv.transforms.ToTensor()
datasetTrain = CustomDataset(X_train, y_train, transform=transform)  # Create train dataset
datasetTest = CustomDataset(X_test, y_test, transform=transform)  # Create test dataset

print(f"train: {len(datasetTrain)}")
print(f"test: {len(datasetTest)}")


show_8_samples(datasetTrain, countsOfRows, 0)  # displays 8 images starting from the index "start"


"""___DATALOADER___"""
batch_size = 16
dataloader = torch.utils.data.DataLoader(datasetTrain, batch_size=batch_size, shuffle=True, drop_last=True)


"""___MODEL___"""
class ShallowNetwork(nn.Module):
    def __init__(self):
        super(ShallowNetwork, self).__init__()
        self.sharpen_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).unsqueeze(
            0).unsqueeze(0)
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=0)
        self.conv.weight.data = self.sharpen_kernel
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear((resizeVal - 2) * (resizeVal - 2), 256)
        self.linear2 = nn.Linear(256, 60)
        self.linear3 = nn.Linear(60, 1)
        self.act = nn.ReLU()
        self.outF = nn.ReLU()

    def forward(self, x):
        # plt.imshow(np.transpose(x[0].numpy(), (1, 2, 0)))
        # plt.axis('off')
        # plt.show()
        x = self.conv(x)
        # plt.imshow(np.transpose(x[0].detach().numpy(), (1, 2, 0)))
        # plt.axis('off')
        # plt.show()
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.act(x)
        x = self.linear3(x)
        x = self.outF(x)
        return x


class ConvolutionNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionNetwork, self).__init__()
        self.act = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv0 = nn.Conv2d(1, 32, 3, stride=1, padding=0)
        self.conv1 = nn.Conv2d(32, 64, 3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=0)

        self.adaptive = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, 12)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):


        return x

learning_rate = 0.05
# loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.MSELoss()

# model = ShallowNetwork()
model = ConvolutionNetwork()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


"""___TRAINING___"""
epochs = 100

for epoch in range(epochs):
    loss_val = 0
    for img, label in dataloader:
        optimizer.zero_grad()
        predict = model(img)

        hourOneHot = nn.functional.one_hot(label[:, 0].long(), 12)

        loss = loss_fn(predict, hourOneHot.float())
        loss.backward()
        optimizer.step()

        loss_val += loss.item()

    print(loss_val / len(dataloader))


"""___TESTING___"""

sizeTest = len(datasetTest)
countCorrect = 0

for img, label in datasetTest:

    predict = model(img)

    label_time_in_minutes = label.detach().numpy()[0] * 60
    predict_time_in_minutes = predict.detach().numpy()[0][0] * 12 * 60
    # print(f"{label_time_in_minutes} - {predict_time_in_minutes:.2f}")

    difference = abs(label_time_in_minutes - predict_time_in_minutes)
    if difference < 5:
        countCorrect += 1


accuracy = countCorrect / sizeTest
print(f"{countCorrect} / {sizeTest}")
print(accuracy)
