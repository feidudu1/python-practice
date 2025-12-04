# import torch.nn as nn
from torch import nn
import torch
from torchsummary import summary


# 这是一个全连接层（Linear Layer），用于将 10 个输入特征映射为 1 个输出值。
# 权重和偏置是模型参数，由 nn.Linear 自动创建
# 每次创建模型时，这些参数会被随机初始化
# 通过训练（如反向传播），这些参数会被更新为最优值
# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 输入特征10个，输出特征1个
#         self.fc = nn.Linear(10, 1)

#     def forward(self, x):
#         x = self.fc(x)
#         return x


# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(10, 5, bias=False)
#         self.fc2 = nn.Linear(5, 1, bias=False)

#     def forward(self, x):
#         x = self.fc1(x)
#         # 这里的x是1*5
#         x = self.fc2(x)
#         # 这里的fc2的输入要求是5*1
#         return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(100, 10, bias=False)
        self.fc1 = nn.Linear(10, 5, bias=False)
        self.fc2 = nn.Linear(5, 1, bias=False)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        # 这里的x是10*5
        x = self.fc2(x)
        # 这里的fc2的输入要求是5*1
        return x


# 激活函数sigmoid


if __name__ == "__main__":
    model = Model()

    input = torch.randn(100)
    output = model(input)
    print(output)

    summary(model, input_size=(100,))
