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


# Use CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

image_dir = 'data/images'
label_dir = 'data/label.csv'
countsOfRows = 10000
resizeVal = 48


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
        images.append(img)


labels = pd.read_csv(label_dir, nrows=countsOfRows)  # Load labels
labels = torch.tensor(labels.values, dtype=torch.float32)  # transform to tensor

X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.4, random_state=12)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=23)

# dividing the dataset into training, test and validation

transform = tv.transforms.ToTensor()
datasetTrain = CustomDataset(X_train, y_train, transform=transform)  # Create train dataset
datasetTest = CustomDataset(X_test, y_test, transform=transform)  # Create test dataset
datasetValidate = CustomDataset(X_val, y_val, transform=transform)  # Create validate dataset

print(f"train: {len(datasetTrain)}")
print(f"test: {len(datasetTest)}")
print(f"validate: {len(datasetValidate)}")


# show_8_samples(datasetTrain, countsOfRows, 0)  # displays 8 images starting from the index "start"


"""___DATALOADER___"""
batch_size = 16
dataloader_train = torch.utils.data.DataLoader(datasetTrain, batch_size=batch_size, shuffle=True, drop_last=False)
dataloader_valid = torch.utils.data.DataLoader(datasetValidate, batch_size=batch_size, shuffle=True, drop_last=False)
dataloader_test = torch.utils.data.DataLoader(datasetTest, batch_size=batch_size, shuffle=True, drop_last=False)


"""___MODEL___"""


class ConvolutionNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionNetwork, self).__init__()
        self.actF = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.softmax = nn.Softmax(dim=1)

        #                       3-channels    64 filters     kernel = 3 stride = 1  pad = 0
        self.conv0 = nn.Conv2d(3, 16, 3, 1, 0)
        self.conv1 = nn.Conv2d(16, 32, 3, 1, 0)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0)

        self.flatten = nn.Flatten()
        # full connected feed forward

        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 12)

    def forward(self, x):
        x = self.conv0(x)
        # x = self.actF(x)
        x = self.maxpool(x)


        x = self.conv1(x)
        # x = self.actF(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        # x = self.actF(x)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.actF(x)

        x = self.fc2(x)
        x = self.actF(x)

        x = self.fc3(x)
        x = self.softmax(x)

        return x


learning_rate = 0.05
loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.MSELoss()

# model = ShallowNetwork()
model = ConvolutionNetwork().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


"""___TRAINING___"""
epochs = 100
train_stats = [[], []]
validate_stats = [[], []]


for epoch in range(epochs):
    loss_train = 0
    loss_valid = 0
    for img, label in dataloader_train:
        optimizer.zero_grad()

        predict = model(img.to(device))
        hourOneHot = nn.functional.one_hot(label[:, 0].long(), 12).to(device)

        loss = loss_fn(predict, hourOneHot.float())
        loss.backward()
        optimizer.step()

        loss_train += loss.item()

    avg_loss_train = loss_train / len(dataloader_train)

    train_stats[0].append(avg_loss_train)
    train_stats[1].append(epoch)
    print(f"epoch: [{epoch}/{epochs}], loss: {avg_loss_train}")

    if epoch % 3 == 0:
        for img, label in dataloader_valid:
            predict = model(img.to(device))
            hourOneHot = nn.functional.one_hot(label[:, 0].long(), 12).to(device)

            loss = loss_fn(predict, hourOneHot.float())

            loss_valid += loss.item()

        avg_loss_valid = loss_valid / len(dataloader_valid)

        validate_stats[0].append(avg_loss_valid)
        validate_stats[1].append(epoch)

plt.plot(train_stats[1], train_stats[0], linestyle='-', color='blue', label='train_data')
plt.plot(validate_stats[1], validate_stats[0], linestyle='--', color='green', label='valid_data')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid(True)
plt.show()

"""___TESTING___"""
sizeTest = len(datasetTest)
countCorrect = 0

for img, label in dataloader_test:
    predict = model(img.to(device))
    hourOneHot = nn.functional.one_hot(label[0][0].long(), 12).to(device)  # create hot tensor for hours
    hourOneHot = hourOneHot.type(torch.float)

    hourPredict = predict[0][:12]  # 12 classes for hours
    hourPredict = hourPredict == hourPredict.max()  # replace the max value with 1, and all other values with 0

    countCorrect += (torch.all(hourPredict == hourOneHot)).item()


accuracy = countCorrect / sizeTest
print(f"{countCorrect} / {sizeTest}")
print(accuracy)
