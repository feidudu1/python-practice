# Copyright 2021 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
# Author: longpeng
# Email: longpeng2008to2012@gmail.com

# coding:utf8
import numpy as np
import matplotlib.pyplot as plt

# 输入数据
X = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
# 标签
Y = np.array([[0, 1, 1, 0]])
# np.random.random()：生成 [0, 1) 区间的随机浮点数
# (3, 4)：形状参数，表示生成 3 行 4 列的二维数组
V = (
    np.random.random((3, 4)) - 0.5
) * 2  # 第一个网络层参数矩阵，初始化输入层权值,取值范围-1到1
W = (
    np.random.random((4, 1)) - 0.5
) * 2  # 第二个网络层参数矩阵，初始化输出层权值,取值范围-1到1


def get_show():
    # 正样本
    all_positive_x = [0, 1]
    all_positive_y = [0, 1]
    # 负样本
    all_negative_x = [0, 1]
    all_negative_y = [1, 0]

    plt.figure()
    # b代表blue，o代表圆形。y代表黄色
    plt.plot(all_positive_x, all_positive_y, "bo")
    plt.plot(all_negative_x, all_negative_y, "yo")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


get_show()

lr = 0.11


# 激活函数(从0～1）
def sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return x


# 激活函数的导数，f'(x)=f(x)(1-f(x)),dsigmoid(x)=sigmoid(x)*(1-sigmoid(x))
def dsigmoid(x):
    x = x * (1 - x)
    return x


# 更新权值（2个权值矩阵，V和W）
def update():
    global X, Y, W, V, lr
    L1 = sigmoid(np.dot(X, V))  # 隐藏层输出(4*3)×(3*4)=(4,4)
    L2 = sigmoid(np.dot(L1, W))  # 输出层输出(4,4)×(4*1)=(4,1)

    # ========== 误差公式解释 ==========
    # 误差传播基于链式法则（Chain Rule）：
    #
    # 1. 输出层误差 L2_delta：
    #    - (Y.T - L2) 是预测误差（损失函数对输出的梯度）
    #    - dsigmoid(L2) 是激活函数的导数（链式法则的一部分）
    #    - 两者逐元素相乘（*）得到完整的梯度
    #    数学公式：∂E/∂L2 = (Y - L2) * sigmoid'(L2)
    #    其中 E 是损失函数（均方误差），sigmoid' 是激活函数的导数
    # 输出层的误差=下一层的误差*激活函数导数*与下一层的连接权重矩阵（全为1）
    L2_delta = (Y.T - L2) * dsigmoid(L2)

    # 2. 隐藏层误差 L1_delta：
    #    - L2_delta.dot(W.T) 将输出层误差反向传播到隐藏层
    #    - W.T 是权重矩阵的转置（因为误差要从输出层传播到隐藏层）
    #    - * dsigmoid(L1) 再次应用链式法则，乘以激活函数的导数
    #    数学公式：∂E/∂L1 = (∂E/∂L2) * W^T * sigmoid'(L1)
    #    这是链式法则的体现：误差通过权重矩阵反向传播
    # 隐藏层的误差=下一层的误差*激活函数导数*与下一层的连接权重矩阵
    L1_delta = L2_delta.dot(W.T) * dsigmoid(L1)

    # ========== 参数更新解释 ==========
    # 参数更新使用矩阵乘法（.dot），不是点乘（*）
    #
    # 1. 输出层参数更新 W_C：
    #    - L1.T 是隐藏层输出的转置 (4×4) -> (4×4)
    #    - L2_delta 是输出层误差 (4×1)
    #    - L1.T.dot(L2_delta) 得到 (4×1)，对应 W 的梯度
    #    数学原理：
    #    - 对于每个样本，梯度 = 误差 × 上一层的激活值
    #    - 矩阵形式：∇W = L1^T × L2_delta（对所有样本求和）
    #    - 形状：(4×4)^T × (4×1) = (4×1)，正好匹配 W 的形状
    # 输出层参数更新值=学习率*误差*上一层的激活值
    W_C = lr * L1.T.dot(L2_delta)

    # 2. 隐藏层参数更新 V_C：
    #    - X.T 是输入数据的转置 (3×4)
    #    - L1_delta 是隐藏层误差 (4×4)
    #    - X.T.dot(L1_delta) 得到 (3×4)，对应 V 的梯度
    #    数学原理：∇V = X^T × L1_delta
    #    形状：(3×4) × (4×4) = (3×4)，正好匹配 V 的形状
    # 隐藏层参数更新=学习率*误差*上一层的激活值
    V_C = lr * X.T.dot(L1_delta)

    # 梯度下降更新：新参数 = 旧参数 + 学习率 × 梯度
    W = W + W_C
    V = V + V_C
    # 返回 L1 和 L2，避免重复计算
    return L1, L2


errors = []  # 记录误差
L1, L2 = None, None  # 初始化，用于保存最终的输出
for i in range(100000):
    L1, L2 = update()  # 更新权值，并获取当前输出
    if i % 1000 == 0:  # 输出误差
        errors.append(np.mean(np.abs(Y.T - L2)))
        print("Error:", np.mean(np.abs(Y.T - L2)))
plt.plot(errors)
plt.ylabel("errors")
plt.show()

# 使用训练完成后的最终输出（已在循环中计算，无需重复计算）
print(L2)


def classify(x):
    if x > 0.5:
        return 1
    else:
        return 0


# L2 是形状为 (4, 1) 的预测输出数组，每个值在 [0, 1] 之间
# classify(x) 将值转换为 0 或 1（阈值 0.5）
# map(classify, L2) 对 L2 的每个元素应用 classify
# for i in map(...) 遍历结果并打印
for i in map(classify, L2):  # L2一共四个数
    print(i)
