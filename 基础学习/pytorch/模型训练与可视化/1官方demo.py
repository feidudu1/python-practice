import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 解释在 https://www.yuque.com/feifei-z1hpi/acw30f/ezw9lz72da0wdm6y/edit?toc_node_uuid=Vfk3zeyoixIfg4AN

# 指定数据集
training_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)

# 设置数据加载器
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# for x, y in test_dataloader:
#     print("x.shape:", x.shape, "y:", y)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"使用设备: {device}")


# 创建模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x


model = NeuralNetwork().to(device)
print(model)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        # 计算预测结果和输出结果相差多少
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        # 优化器归零，类似称东西，每次称前去皮归零
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    # 总样本数
    size = len(dataloader.dataset)
    # dataloader是一个DataLoader对象，将数据集分成多个batch进行迭代，每个batch 包含batch_size个样本
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            # 测试时不需要计算梯度，只需要推理，不用更新参数等
            x, y = x.to(device), y.to(device)
            pred = model(x)
            # 在循环中累加每个 batch 的损失
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    # 计算测试集的平均损失
    test_loss /= num_batches
    # test_loss = test_loss / num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


# epochs = 5
# for t in range(epochs):
#     print(f"Epoch {t}")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn)
# print("Done!")

# torch.save(model.state_dict(), "model.pth")
# print("保存成功")

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("./model.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# 将模型切换到验证模式
model.eval()
# 这里的x为图像的tensor信息，y为标签（数字）
# test_data[0] 代表第几个样本，[0]代表图像信息，【1】代表标签
x, y = test_data[0][0], test_data[0][1]

# 禁用梯度计算
# 推理阶段不需要梯度:
# 训练：需要梯度来更新参数
# 推理/测试：只需要预测结果，不需要梯度
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
