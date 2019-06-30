import torch
from torch import nn

class VggBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VggBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )

    def forward(self, x):

        x = self.conv(x)
        x = torch.relu(x)

        return x

class Vgg19(nn.Module):
    def __init__(self, output_classes):
        super(Vgg19, self).__init__()

        self.output_classes = output_classes

        self.conv1_blocks = 2
        self.conv2_blocks = 2
        self.conv3_blocks = 4
        self.conv4_blocks = 4
        self.conv5_blocks = 4


        self.max_pool_22 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            padding=0)


        in_channels = 3
        out_channels = 3

        self.conv1 = nn.Sequential()
        for layer_idx in range(self.conv1_blocks):

            if layer_idx == 0:
                out_channels = 64

            self.conv1.add_module(
                f'conv1_{layer_idx}',
                VggBlock(
                    in_channels = in_channels,
                    out_channels = out_channels
                )
            )
            in_channels = out_channels


        self.conv2 = nn.Sequential()
        for layer_idx in range(self.conv2_blocks):

            if layer_idx == 0:
                out_channels = 128

            self.conv2.add_module(
                f'conv2_{layer_idx}',
                VggBlock(
                    in_channels = in_channels,
                    out_channels = out_channels
                )
            )
            in_channels = out_channels

        self.conv3 = nn.Sequential()
        for layer_idx in range(self.conv3_blocks):

            if layer_idx == 0:
                out_channels = 256

            self.conv3.add_module(
                f'conv3_{layer_idx}',
                VggBlock(
                    in_channels = in_channels,
                    out_channels = out_channels
                )
            )
            in_channels = out_channels

        self.conv4 = nn.Sequential()
        for layer_idx in range(self.conv4_blocks):

            if layer_idx == 0:
                out_channels = 512

            self.conv4.add_module(
                f'conv4_{layer_idx}',
                VggBlock(
                    in_channels = in_channels,
                    out_channels = out_channels
                )
            )
            in_channels = out_channels

        self.conv5 = nn.Sequential()
        for layer_idx in range(self.conv4_blocks):
            self.conv5.add_module(
                f'conv5_{layer_idx}',
                VggBlock(
                    in_channels = in_channels,
                    out_channels = out_channels
                )
            )


        self.fc1 = nn.Linear(7 * 7 * 512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.output_classes)


    def forward(self, x):

        x = self.conv1.forward(x)
        x = self.max_pool_22(x)

        x = self.conv2.forward(x)
        x = self.max_pool_22(x)

        x = self.conv3.forward(x)
        x = self.max_pool_22(x)

        x = self.conv4.forward(x)
        x = self.max_pool_22(x)

        x = self.conv5.forward(x)
        x = self.max_pool_22(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = torch.relu(x)

        x = self.fc3(x)
        x = torch.softmax(x, dim=1)

        return x
