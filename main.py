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
    plt.imshow(dataset[start + 5][0].numpy()[0], cmap="gray")
    plt.xlabel(f"{dataset[start + 5][1][0]} : {dataset[start + 5][1][1]}")

    plt.subplot(2, 4, 7)
    plt.imshow(dataset[start + 6][0].numpy()[0], cmap="gray")
    plt.xlabel(f"{dataset[start + 6][1][0]} : {dataset[start + 6][1][1]}")

    plt.subplot(2, 4, 8)
    plt.imshow(dataset[start + 7][0].numpy()[0], cmap="gray")
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
countsOfRows = 10000
resizeVal = 32


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
        img = cv.resize(img, (resizeVal, resizeVal), interpolation=cv.INTER_AREA)  # resize image

        # clahe = cv.createCLAHE(clipLimit=0.1, tileGridSize=(4, 4))  # increase image contrast
        # b, g, r = cv.split(img)
        # img = cv.merge((clahe.apply(b), clahe.apply(g), clahe.apply(r)))

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # transform images to gray scale, only one channel of gray
        # _, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

        images.append(img)
        # plt.imshow(img, cmap="gray")
        # plt.show()


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


# plt.imshow(myDataset[row][0].numpy()[0], cmap="gray")  # picture
# plt.xlabel(f"{myDataset[row][1][0]} : {myDataset[row][1][1]}")  # label
# plt.show()  # visualise the first picture and label


"""___Dataloader___"""
batch_size = 16
dataloader = torch.utils.data.DataLoader(datasetTrain, batch_size=batch_size, shuffle=True)

# for img, l in dataloader:
#     print(img.shape)
#     print(l.shape)
#     break


"""___MODEL___"""

class ShallowNetwork(nn.Module):
    def __init__(self):
        super(ShallowNetwork, self).__init__()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(resizeVal * resizeVal, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 72)
        self.act = nn.ReLU()
        self.outF = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.act(x)
        x = self.linear3(x)
        x = self.outF(x)
        return x


learning_rate = 0.02
model = ShallowNetwork()
loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(model.parameters())



"""___TRAINING___"""
epochs = 10

for epoch in range(epochs):
    loss_val = 0
    for img, label in dataloader:
        optimizer.zero_grad()

        predict = model(img)

        hourOneHot = nn.functional.one_hot(label[:, 0].long(), 12)  # create hot tensor for hours
        minuteOneHot = nn.functional.one_hot(label[:, 1].long(), 60)  # create hot tensor for minutes

        labelOneHot = torch.cat((hourOneHot, minuteOneHot), dim=1)  # combining minutes and hours into one output tensor
        # print(f"shape of labelOneHot{labelOneHot.shape}")

        loss = loss_fn(predict, labelOneHot.float())
        loss.backward()

        loss_val += loss.item()

        optimizer.step()
    print(loss_val / len(dataloader))

"""___TESTING___"""

sizeTest = len(datasetTest)
countCorrect = 0

for img, label in datasetTest:

    predict = model(img)

    hourOneHot = nn.functional.one_hot(label[0].long(), 12)  # create hot tensor for hours
    hourOneHot = hourOneHot.type(torch.int)
    minuteOneHot = nn.functional.one_hot(label[1].long(), 60)  # create hot tensor for minutes
    minuteOneHot = minuteOneHot.type(torch.int)


    labelOneHot = torch.cat((hourOneHot, minuteOneHot), dim=0)

    hourPredict = predict[0][:12]
    hourPredict = hourPredict == hourPredict.max()
    hourPredict = hourPredict.type(torch.int)

    minutePredict = predict[0][12:]
    minutePredict = minutePredict == minutePredict.max()
    minutePredict = minutePredict.type(torch.int)

    predictOneHot = torch.cat((minutePredict, hourPredict), dim=0)
    countCorrect += (torch.all(predictOneHot == labelOneHot)).item()

accuracy = countCorrect / sizeTest
print(f"{countCorrect} / {sizeTest}")
print(accuracy)
