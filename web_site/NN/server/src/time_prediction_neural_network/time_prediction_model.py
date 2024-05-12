from typing import NamedTuple

import torch
import torch.nn as nn
import torchvision as tv
import cv2 as cv
from pathlib import Path
from web_site.NN.server.src.utils import get_file_path_in_module


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


class Time(NamedTuple):
    hour: int
    minute: int


def get_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def implement_model(trained_model: Path, device: torch.device) -> CustomModel:
    model = CustomModel().to(device)
    model.load_state_dict(torch.load(trained_model, map_location=device))
    return model


def process_image(image: Path) -> torch.Tensor:
    uploaded_image = cv.imread(str(image))
    resized_image = cv.resize(uploaded_image, (100, 100), interpolation=cv.INTER_AREA)  # resize image
    grey_scale_image = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)  # transform images to gray scale, only one channel of gray
    transform = tv.transforms.ToTensor()
    transformed_image = transform(grey_scale_image)
    processed_image = transformed_image.unsqueeze(0)
    return processed_image


def predict_time(implemented_model: CustomModel, processed_image: torch.Tensor) -> Time:
    hour, minute = implemented_model(processed_image.to(get_device()))
    hour = torch.argmax(hour).item()
    minute = round(minute.item())
    predicted_time = Time(hour=hour, minute=minute)
    return predicted_time


def get_prediction(image: Path) -> Time:
    device = get_device()
    path_for_model = get_file_path_in_module("model.pth", Path(__file__))
    model = implement_model(path_for_model, device)
    processed_image = process_image(image.absolute())
    prediction_time = predict_time(model, processed_image)
    return prediction_time


if __name__ == "__main__":
    print((get_prediction(Path("../endpoints/client_files/image")))._asdict())
