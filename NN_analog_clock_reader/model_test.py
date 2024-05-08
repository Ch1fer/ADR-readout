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
import torch.nn.init as init
import numpy as np

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

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 50, kernel_size=5, stride=2)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(50)

        self.conv2 = nn.Conv2d(50, 100, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(100)

        self.conv3 = nn.Conv2d(100, 150, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(150)

        self.conv4 = nn.Conv2d(150, 200, kernel_size=3, stride=1)
        self.dropout = nn.Dropout(0.4)

        self.flatten = nn.Flatten()

        self.hour_fc1 = nn.Linear(800, 144)
        self.hour_fc2 = nn.Linear(144, 144)
        self.hour_fc3 = nn.Linear(144, 12)

        self.minute_fc1 = nn.Linear(800, 100)
        self.minute_fc2 = nn.Linear(100, 200)
        self.minute_fc3 = nn.Linear(200, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.dropout(x)

        x = self.flatten(x)

        hour = self.hour_fc1(x)
        hour = self.relu(hour)
        hour = self.hour_fc2(hour)
        hour = self.relu(hour)
        hour = self.hour_fc3(hour)

        minute = self.minute_fc1(x)
        minute = self.relu(minute)
        minute = self.minute_fc2(minute)
        minute = self.relu(minute)
        minute = self.minute_fc3(minute)

        return hour, minute


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CustomModel().to(device)
model.load_state_dict(torch.load('my_model.pth', map_location=device))

image_dir = 'data/images'
label_dir = 'data/label.csv'
countsOfRows = 10000
resizeVal = 100


"""___PREPARE_DATA_AND_TRANSFORM___"""
files = os.listdir(image_dir)  # Load images
images = []
fileCounter = 0
for file in sorted(files, key=lambda x: int(x.split('.')[0])):
    if fileCounter >= countsOfRows:
        break
    fileCounter += 1

    if file.endswith('.jpg'):
        image_path = os.path.join(image_dir, file)  # create directory path
        print(image_path)
        img = cv.imread(image_path)
        img = cv.resize(img, (resizeVal, resizeVal), interpolation=cv.INTER_AREA)  # resize image
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # transform images to gray scale, only one channel of gray
        images.append(img)


labels = pd.read_csv(label_dir, nrows=countsOfRows)  # Load labels
labels = torch.tensor(labels.values, dtype=torch.float32)  # transform to tensor

transform = tv.transforms.ToTensor()
datasetTest = CustomDataset(images, labels, transform=transform)  # Create test dataset

batch_size = 32
dataloader_test = torch.utils.data.DataLoader(datasetTest, batch_size=batch_size, shuffle=True, drop_last=True)

"""___TESTING___"""
sizeTest = len(datasetTest)
countCorrect = 0

for img, label in dataloader_test:
    hour, minute = model(img.to(device))
    for i in range(len(hour)):

        hourPredict = hour[i]  # 12 classes for hours

        fire_hour = torch.argmax(hourPredict).item()

        if fire_hour == label[i][0] and abs(minute[i] - label[i][1]) < 5:
            countCorrect += 1
        # else:
        #     plt.imshow(img[i].permute(1, 2, 0).numpy())
        #     plt.xlabel(f"котра година: {label[i][0]} : {label[i][1]} , нейронка каже {fire_hour}: {float(minute[i])}")
        #     plt.show()

torch.save(model.state_dict(), 'my_model.pth')
accuracy = countCorrect / sizeTest
print(f"{countCorrect} / {sizeTest}")
print(f"accuracy: {accuracy}")