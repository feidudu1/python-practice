import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm  # pip install tqdm

import matplotlib.pyplot as plt

# 指定训练集
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# 指定测试集
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# 创建数据加载器
batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# 指定设备
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device} device")


# 定义模型
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
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()
# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# # 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    # 初始化训练数据集的大小和批次数量
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # 设置模型为训练模式
    model.train()
    # 初始化总损失和正确预测数量
    loss_total = 0
    correct = 0
    # 遍历数据加载器中的所有数据批次
    for X, y in tqdm(dataloader):
        # 将数据和标签移动到指定设备（例如GPU）
        X, y = X.to(device), y.to(device)
        # 使用模型进行预测
        pred = model(X)
        # 计算正确预测的数量
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        # 计算预测结果和真实结果之间的损失
        loss = loss_fn(pred, y)
        # 累加总损失
        loss_total += loss.item()
        # 执行反向传播，计算梯度
        loss.backward()
        # 更新模型参数
        optimizer.step()
        # 清除梯度信息
        optimizer.zero_grad()

    # 计算平均损失和准确率
    loss_avg = loss_total / num_batches
    correct /= size
    # 返回准确率和平均损失，保留三位小数
    return round(correct, 3), round(loss_avg, 3)


# 定义测试函数
def test(dataloader, model, loss_fn):
    # 初始化测试数据集的大小和批次数量
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # 设置模型为评估模式
    model.eval()

    # 初始化测试损失和正确预测数量
    test_loss, correct = 0, 0

    # 不计算梯度，以提高计算效率并减少内存使用
    with torch.no_grad():
        # 遍历数据加载器中的所有数据批次
        for X, y in tqdm(dataloader):
            # 将数据和标签移动到指定设备（例如GPU）
            X, y = X.to(device), y.to(device)
            # 使用模型进行预测
            pred = model(X)
            # 累加预测损失
            test_loss += loss_fn(pred, y).item()
            # 累加正确预测的数量
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # 计算平均测试损失和准确率
    test_loss /= num_batches
    correct /= size

    # 返回准确率和平均测试损失，保留三位小数
    return round(correct, 3), round(test_loss, 3)


train_acc_list = []
train_loss_list = []

test_acc_list = []
test_loss_list = []


# 定义循环次数，每次循环里面，先训练，再测试
epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_acc, train_loss = train(train_dataloader, model, loss_fn, optimizer)
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)
    test_acc, test_loss = test(test_dataloader, model, loss_fn)
    test_acc_list.append(test_acc)
    test_loss_list.append(test_loss)
print("Done!")


x_list = [i + 1 for i in range(len(train_acc_list))]
plt.plot(x_list, train_acc_list, label="Train")
plt.plot(x_list, test_acc_list, label="Test")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.plot(x_list, train_loss_list, label="Train")
plt.plot(x_list, test_loss_list, label="Test")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
