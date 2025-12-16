from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == "__main__":
    # 数据都保存在 "data" 文件夹中
    writer = SummaryWriter(log_dir="data")
    model = NeuralNetwork()
    # 添加tensorboard的模型记录
    init_img = torch.zeros((1, 3, 28, 28))
    writer.add_graph(model, init_img)
    writer.close()
