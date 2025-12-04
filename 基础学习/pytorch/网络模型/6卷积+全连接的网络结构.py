# import torch.nn as nn
from torch import nn
import torch
from torchsummary import summary


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 32, 3)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(32, 64, 3)
        # 为什么main函数里输入为224，这里为220？
        # 因为卷积操作会减小特征图的尺寸，这里卷积步长默认为1，输出尺寸 = 输入尺寸 - （卷积核大小 - 1）
        # 输入: (5, 3, 224, 224)           # 224×224
        # 经过 conv_1 (3×3卷积核，无padding):
        # 输出: (5, 32, 222, 222)          # 224 - (3-1) = 224 - 2 = 222
        # 经过 conv_2 (3×3卷积核，无padding):
        # 输出: (5, 64, 220, 220)          # 222 - (3-1) = 222 - 2 = 220
        self.fc1 = nn.Linear(64 * 220 * 220, 512)
        self.fc2 = nn.Linear(512, 10)
        # 最后的输出维度为1
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.relu(x)
        x = x.view(-1, 64 * 220 * 220)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# 激活函数sigmoid


if __name__ == "__main__":
    model = ConvNet()

    input = torch.randn((5, 3, 224, 224))
    output = model(input)
    print(output.shape)
    print(output)

    summary(model, input_size=(3, 224, 224))
