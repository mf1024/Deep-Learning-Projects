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

class FashionPerceptronModel(nn.Module):
    def __init__(self):
        super(FashionPerceptronModel, self).__init__()

        self.deep_layer_size = 512

        self.fc1 = nn.Linear(in_features = 28 * 28, out_features = self.deep_layer_size)
        self.bn1 = nn.BatchNorm1d(self.deep_layer_size)
        self.fc2 = nn.Linear(in_features = self.deep_layer_size, out_features = self.deep_layer_size)
        self.bn2 = nn.BatchNorm1d(self.deep_layer_size)
        self.fc3 = nn.Linear(in_features = self.deep_layer_size, out_features = 10)
        self.bn3 = nn.BatchNorm1d(10)

    def forward(self, input):
        # input size is [BATCH_SIZE, 28, 28]
        # we want to transform that into [BATCH_SIZE, 28 * 28]

        x = input.reshape(input.shape[0], 28 * 28)

        x = self.fc1.forward(x)
        x = self.bn1.forward(x)
        x = torch.relu(x)
        x = self.fc2.forward(x)
        x = self.bn2.forward(x)
        x = torch.relu(x)
        x = self.fc3.forward(x)
        x = self.bn3.forward(x)

        x = torch.softmax(x, dim=1)

        return x

model =  FashionPerceptronModel()
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
