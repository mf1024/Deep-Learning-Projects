import torch
from torch import nn
from torchsummary import summary

class ConvBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.bottleneck_channels = bottleneck_channels
        self.out_channels = out_channels

        self.bn11 = nn.BatchNorm2d(in_channels)
        self.relu11 = nn.ReLU()
        self.conv11 = nn.Conv2d(
            in_channels = self.in_channels,
            out_channels = self.bottleneck_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )

        self.bn33 = nn.BatchNorm2d(self.bottleneck_channels)
        self.relu33 = nn.ReLU()
        self.conv33 = nn.Conv2d(
            in_channels = self.bottleneck_channels,
            out_channels = self.out_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )

    def forward(self, x):

        x = self.bn11(x)
        x = self.relu11(x)
        x = self.conv11(x)

        x = self.bn33(x)
        x = self.relu33(x)
        x = self.conv33(x)

        return x


class DenseBlock(nn.Module):
    def __init__(self, conv_block_num, in_channels, k):
        super().__init__()

        self.conv_block_list = nn.ModuleList()

        running_in_channels = in_channels
        for conv_block_idx in range(conv_block_num):
            self.conv_block_list.append(
                ConvBlock(
                    in_channels = running_in_channels,
                    bottleneck_channels= 4 * k,
                    out_channels = k
                )
            )

            running_in_channels += k

    def forward(self, x):
        collective_knowledge = x
        for conv_block in self.conv_block_list:
            x = conv_block.forward(collective_knowledge)
            collective_knowledge = torch.cat((collective_knowledge, x), dim = 1)

        return collective_knowledge


class DenseNet121(nn.Module):
    def __init__(self, class_num):
        super().__init__()

        self.class_num = class_num
        self.compression_factor = 0.5
        self.k = 32

        self.blocks_layers_num = [6, 12, 24, 16]
        self.num_dense_blocks = len(self.blocks_layers_num)

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()


        self.conv1 = nn.Conv2d(
            in_channels = 3,
            out_channels = 2 * self.k,
            kernel_size=7,
            stride=2,
            padding=3
        )

        channels = 2 * self.k

        self.max_pool = nn.MaxPool2d(
            kernel_size = 3,
            stride = 2,
            padding = 1
        )

        for block_idx in range(self.num_dense_blocks):

            self.dense_blocks.append(

                DenseBlock(
                    conv_block_num = self.blocks_layers_num[block_idx],
                    in_channels = channels,
                    k = self.k,
                )
            )
            channels = channels + self.k * self.blocks_layers_num[block_idx]

            if block_idx < self.num_dense_blocks - 1:

                self.transition_layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels = channels,
                            out_channels = int(channels * self.compression_factor),
                            kernel_size = 1,
                            stride = 1,
                            padding = 0
                        ),
                        nn.AvgPool2d(
                            kernel_size = 2,
                            stride = 2
                        )
                    )
                )
                channels = int(channels * self.compression_factor)

        self.global_avg_pooling = nn.AvgPool2d(kernel_size=7)
        self.fully_connected = nn.Linear(channels, self.class_num)

    def forward(self, x):

        x = self.conv1(x)
        x = self.max_pool(x)

        for block_idx in range(self.num_dense_blocks):
            x = self.dense_blocks[block_idx](x)
            if (block_idx) < self.num_dense_blocks - 1:
                x = self.transition_layers[block_idx](x)

        x = self.global_avg_pooling(x)
        x = torch.squeeze(x)
        x = self.fully_connected(x)

        return x


