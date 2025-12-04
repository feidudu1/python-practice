# import torch.nn as nn
from torch import nn
import torch
from torchsummary import summary


class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        # 3是输入通道数，32是输出通道数
        # 最后的3表示卷积核大小事3*3
        self.conv_1 = nn.Conv2d(3, 32, 3)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(32, 64, 3)
        self.conv_3 = nn.Conv2d(64, 128, 3)
        # 反卷积
        self.conv_4 = nn.ConvTranspose2d(128, 64, 3)
        # 3表示3通道，通常是rgb。64表示64通道/特征
        self.conv_5 = nn.ConvTranspose2d(64, 32, 3)
        self.conv_6 = nn.ConvTranspose2d(32, 10, 3)
        # 最后的输出维度为1
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.conv_3(x)
        x = self.relu(x)
        x = self.conv_4(x)
        x = self.relu(x)
        x = self.conv_5(x)
        x = self.relu(x)
        x = self.conv_6(x)
        x = self.relu(x)
        x = self.softmax(x)
        return x


# 激活函数sigmoid


if __name__ == "__main__":
    model = FCN()

    input = torch.randn((10, 3, 224, 224))
    output = model(input)
    print(output.shape)
    print(output)

    summary(model, input_size=(3, 224, 224))
