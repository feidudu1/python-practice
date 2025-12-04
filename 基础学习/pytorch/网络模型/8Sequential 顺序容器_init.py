import torch.nn as nn
import torch
from torchsummary import summary


class MyBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x_1 = x
        x = self.conv1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn_2(x)
        x = self.relu(x)
        # 将x + x_1时，因为两者shape不一致，前面是224*224，后者为220，所以需要加padding
        result = x + x_1
        return result


class MainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.block1 = MyBlock(64, 64)
        self.block2 = MyBlock(64, 64)
        self.block3 = MyBlock(64, 64)
        self.block4 = MyBlock(64, 64)
        self.fc1 = nn.Linear(64 * 220 * 220, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(-1, 64 * 220 * 220)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    input = torch.rand((5, 3, 224, 224))
    model = MainNet()
    # model = MyBlock(3, 3)  # 这里的值不是根据图来的，是自己随机定义的
    output = model(input)

    print(output.shape)
    print(output)
    print(model)

    summary(model, input_size=(3, 224, 224))
