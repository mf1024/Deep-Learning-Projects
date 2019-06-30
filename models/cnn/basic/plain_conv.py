
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from pytorch_datasets.fashion_mnist import FashionMnistDataset

import matplotlib
matplotlib.colors.Colormap('inferno')

BATCH_SIZE = 16
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-4

train_fashion_mnist_dataset = FashionMnistDataset("./data", train=True, download = True)
test_fashion_mnist_dataset = FashionMnistDataset("./data", train=False, download = True)

train_dataloader = DataLoader(train_fashion_mnist_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_dataloader = DataLoader(test_fashion_mnist_dataset, batch_size = BATCH_SIZE, shuffle = True)

in_features = 28 * 28

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduce_size = False):
        super(ConvBlock, self).__init__()

        stride = 1
        if reduce_size:
            stride = 2

        self.conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)

        return x


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = ConvBlock(in_channels=1, out_channels=16, reduce_size=True)
        self.conv2 = ConvBlock(in_channels=16, out_channels=16, reduce_size=False)
        self.conv3 = ConvBlock(in_channels=16, out_channels=16, reduce_size=False)
        self.conv4 = ConvBlock(in_channels=16, out_channels=16, reduce_size=False)

        self.conv5 = ConvBlock(in_channels=16, out_channels=32, reduce_size=True)
        self.conv6 = ConvBlock(in_channels=32, out_channels=32, reduce_size=False)
        self.conv7 = ConvBlock(in_channels=32, out_channels=32, reduce_size=False)
        self.conv8 = ConvBlock(in_channels=32, out_channels=32, reduce_size=False)

        self.conv9 = ConvBlock(in_channels=32, out_channels=64, reduce_size=True)
        self.conv10 = ConvBlock(in_channels=64, out_channels=64, reduce_size=False)
        self.conv11 = ConvBlock(in_channels=64, out_channels=64, reduce_size=False)
        self.conv12 = ConvBlock(in_channels=64, out_channels=64, reduce_size=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(64,10)

    def forward(self, x):

        x = torch.unsqueeze(x, dim=1)

        x = self.conv1.forward(x)
        x = self.conv2.forward(x)
        x = self.conv3.forward(x)
        x = self.conv4.forward(x)
        x = self.conv5.forward(x)
        x = self.conv6.forward(x)
        x = self.conv7.forward(x)
        x = self.conv8.forward(x)
        x = self.conv9.forward(x)
        x = self.conv10.forward(x)
        x = self.conv11.forward(x)
        x = self.conv12.forward(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), x.size(1))

        x = self.fc(x)

        x = torch.softmax(x, dim=1)

        return x


model =  ConvNet()
optimizer = torch.optim.Adam(params = model.parameters(), lr = LEARNING_RATE)

for epoch in range(NUM_EPOCHS):

    #Training

    sum_correct = 0.0
    sum_samples = 0.0
    sum_loss = 0.0

    model = model.train()

    for labels, images in train_dataloader:

        # plt.imshow(images[0], cmap="inferno")
        # plt.show()

        x = torch.tensor(images).float()
        y = torch.tensor(labels).long()

        y_pred = model.forward(x) # We get prediction for each class for image in the batch so [BATCH_SIZE, CLASSES]
        y_one_hot = torch.zeros(x.shape[0],10)
        y_one_hot = y_one_hot.scatter_(1, y.unsqueeze(dim=1), 1)

        loss = torch.sum(- y_one_hot * torch.log(y_pred))
        sum_loss += loss.detach().to('cpu')

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        right_predictions = y_pred.detach().to('cpu').argmax(dim=1) == y.detach().to('cpu')
        sum_correct += right_predictions.sum()
        sum_samples += len(right_predictions)

    print(f"Epoch {epoch} train accuracy is {float(sum_correct)/float(sum_samples)}")
    print(f"Sum train loss is {sum_loss}")


    # Testing
    sum_correct = 0.0
    sum_samples = 0.0
    sum_loss = 0.0

    model = model.eval()
    with torch.no_grad():

        for labels, images in test_dataloader:

            x = torch.tensor(images).float()
            y = torch.tensor(labels).long()

            y_pred = model.forward(x) # We get prediction for each class for image in the batch so [BATCH_SIZE, CLASSES]
            y_one_hot = torch.zeros(x.shape[0],10)
            y_one_hot = y_one_hot.scatter_(1, y.unsqueeze(dim=1), 1)

            loss = torch.sum(- y_one_hot * torch.log(y_pred))
            sum_loss += loss.detach().to('cpu')

            right_predictions = y_pred.detach().to('cpu').argmax(dim=1) == y.detach().to('cpu')
            sum_correct += right_predictions.sum()
            sum_samples += len(right_predictions)

    print(f"Epoch {epoch} test accuracy is {float(sum_correct)/float(sum_samples)}")
