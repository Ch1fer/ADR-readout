import torch
import torch.nn as nn
import torchvision as tv
import cv2 as cv


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


image_dir = 'data/images/41835.jpg'

img = cv.imread(image_dir)
img = cv.resize(img, (100, 100), interpolation=cv.INTER_AREA)  # resize image
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # transform images to gray scale, only one channel of gray

transform = tv.transforms.ToTensor()

img = transform(img)
img = img.unsqueeze(0)

hour, minute = model(img.to(device))
hour = torch.argmax(hour).item()
minute = round(minute.item())

print(f" {hour} : {minute}")
