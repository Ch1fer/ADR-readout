import torch
from torch.utils import data
from torch.utils.data import Dataset
import torch.nn as nn
import pandas as pd
import torchvision as tv
import cv2 as cv
import os
import matplotlib.pyplot as plt



def show_8_samples(dataset, size, start):
    if size <= start + 8:
        print("wrong start!")
        return

    plt.subplot(2, 4, 1)
    plt.imshow(dataset[start][0].numpy()[0], cmap="gray")
    plt.xlabel(f"{dataset[start][1][0]} : {dataset[start][1][1]}")

    plt.subplot(2, 4, 2)
    plt.imshow(dataset[start + 1][0].numpy()[0], cmap="gray")
    plt.xlabel(f"{dataset[start + 1][1][0]} : {dataset[start + 1][1][1]}")

    plt.subplot(2, 4, 3)
    plt.imshow(dataset[start + 2][0].numpy()[0], cmap="gray")
    plt.xlabel(f"{dataset[start + 2][1][0]} : {dataset[start + 2][1][1]}")

    plt.subplot(2, 4, 4)
    plt.imshow(dataset[start + 3][0].numpy()[0], cmap="gray")
    plt.xlabel(f"{dataset[start + 3][1][0]} : {dataset[start + 3][1][1]}")

    plt.subplot(2, 4, 5)
    plt.imshow(dataset[start + 4][0].numpy()[0], cmap="gray")
    plt.xlabel(f"{dataset[start + 4][1][0]} : {dataset[start + 4][1][1]}")

    plt.subplot(2, 4, 6)
    plt.imshow(myDataset[start + 5][0].numpy()[0], cmap="gray")
    plt.xlabel(f"{dataset[start + 5][1][0]} : {dataset[start + 5][1][1]}")

    plt.subplot(2, 4, 7)
    plt.imshow(dataset[start + 6][0].numpy()[0], cmap="gray")
    plt.xlabel(f"{dataset[start + 6][1][0]} : {dataset[start + 6][1][1]}")

    plt.subplot(2, 4, 8)
    plt.imshow(myDataset[start + 7][0].numpy()[0], cmap="gray")
    plt.xlabel(f"{dataset[start + 7][1][0]} : {dataset[start + 7][1][1]}")
    plt.show()

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
countsOfRows = 1000


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
        img = cv.resize(img, (48, 48), interpolation=cv.INTER_AREA)  # resize image

        clahe = cv.createCLAHE(clipLimit=0.1, tileGridSize=(4, 4))  # increase image contrast
        b, g, r = cv.split(img)
        img = cv.merge((clahe.apply(b), clahe.apply(g), clahe.apply(r)))

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # transform images to gray scale, only one channel of gray
        images.append(img)
        # plt.imshow(img, cmap="gray")
        # plt.show()


labels = pd.read_csv(label_dir, nrows=countsOfRows)  # Load labels
labels = torch.tensor(labels.values, dtype=torch.float32)

transform = tv.transforms.ToTensor()
myDataset = CustomDataset(images, labels, transform=transform)  # Create custom dataset from previously images & labels

show_8_samples(myDataset, countsOfRows, 0)  # displays 8 images starting from the index "start"


# plt.imshow(myDataset[row][0].numpy()[0], cmap="gray")  # picture
# plt.xlabel(f"{myDataset[row][1][0]} : {myDataset[row][1][1]}")  # label
# plt.show()  # visualise the first picture and label


"""___Dataloader___"""
batch_size = 16
dataloader = torch.utils.data.DataLoader(myDataset, batch_size=batch_size, shuffle=True)

for img, l in dataloader:
    print(img.shape)
    print(l.shape)
    break


# """___MODEL___"""
#
#
# class ShallowNetwork(nn.Module):
#     def __init__(self):
#         super(ShallowNetwork, self).__init__()
#
#         self.flatten = nn.Flatten()
#         self.linear1 = nn.Linear(48 * 48, 100)
#         self.linear2 = nn.Linear(100, 50)
#         self.linear3 = nn.Linear(50, 2)
#         self.act = nn.ReLU()
#         self.outF = nn.Softmax()
#
#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.linear1(x)
#         x = self.act(x)
#         x = self.linear2(x)
#         x = self.act(x)
#         x = self.linear3(x)
#         x = self.outF(x)
#         return x
#
#
# learning_rate = 0.01
# model = ShallowNetwork()
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#
#
#
# """___TRAINING___"""
# epochs = 100
#
# for epoch in range(epochs):
#     loss_val = 0
#     for img, label in dataloader:
#         optimizer.zero_grad()
#
#         # one_hot_first_column = nn.functional.one_hot(label[:, 0], 12)
#         # one_hot_second_column = nn.functional.one_hot(label[:, 1], 60)
#         #
#         # one_hot_label = torch.cat((one_hot_first_column, one_hot_second_column), dim=1)
#         predict = model(img)
#
#         loss = loss_fn(predict, label)
#         loss.backward()
#
#         loss_val += loss.item()
#
#         optimizer.step()
#
#     print(loss_val / len(dataloader))
