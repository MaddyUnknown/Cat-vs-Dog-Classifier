from torch import load, from_numpy
import torch.nn as nn
import numpy as np
from PIL import Image
import sys
import os

class ToTorch(object):
    def __call__(self, image):
        image = image.resize((120,120), Image.ANTIALIAS)
        image = np.array(image)
        image = image.transpose((2,0,1))
        image = image.reshape((1,3,120,120))
        return from_numpy(image)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.batch1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.batch2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv3 = nn.Conv2d(64, 128 , 3)
        self.batch3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.fc1 = nn.Linear(13*13*128, 128)
        self.relu_fc1 = nn.ReLU()
        self.batch_fc1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 80)
        self.relu_fc2 = nn.ReLU()

        self.batch_fc2 = nn.BatchNorm1d(80)
        self.fc3 = nn.Linear(80, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, choice=0):
        x = x/225
        x = self.conv1(x)
        x = self.pool1(self.batch1(self.relu1(x)))
        x = self.conv2(x)
        x = self.pool2(self.batch2(self.relu2(x)))
        x = self.conv3(x)
        x = self.pool3(self.batch3(self.relu3(x)))
        x = x.view(-1, 13*13*128)
        x = self.batch_fc1(self.relu_fc1(self.fc1(x)))
        x = self.batch_fc2(self.relu_fc2(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))

        return x


model = Net()
model.load_state_dict(load(r'Model\Model_new_v4\Model_new_v4'))
model.eval()

def predict(name):
    image = Image.open(name)
    image = ToTorch()(image)
    output = float(model(image.float()))>0.5
    if int(output)==0:
        print(name+" is a Cat")
    else:
        print(name+" is a Dog")

for i in sys.argv[1:]:
    if(i.startswith("--")):         #opening folder
        image_folder = i[2:]
        if os.path.isdir(image_folder):
            for img in os.listdir(image_folder):
                predict(os.path.join(image_folder, img))
    elif (i.startswith("-")):           #reading a txt file
        with open(i[1:], 'r') as file:
            names = file.read().splitlines()
            for i in names:
                predict(i)
    else:
        predict(i)
