# import torch.nn as nn
from torch import nn
import torch
from torchsummary import summary


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1000, 100)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 5)
        # 最后的输出维度为1
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = NeuralNetwork()

    # torch.randn((1000,)) 和 torch.randn(1000) 在功能上完全相同，都会创建一个形状为 (1000,) 的一维张量。
    # 4表示batch size
    input = torch.randn((4, 1000))
    output = model(input)
    print(output.shape)
    print(output)

    summary(model, input_size=(1000,))
