import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# 定义函数
def f(x):
    return x**2


# 定义导数
def df(x):
    return 2 * x


# 梯度下降参数
lr = 0.1
n_iterations = 10  # 迭代次数
x1 = 2.5  # 初始值

# 绘制原始函数
x = np.linspace(-3, 3, 100)
y = f(x)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, label="f(x) = x^2")
(point,) = ax.plot([], [], "ro", label="Gradient Descent")
value_display = ax.text(0.7, 0.02, "", transform=ax.transAxes)


def init():
    point.set_data([], [])
    value_display.set_text("")
    return point, value_display


def update(i):
    global x1
    gradient = df(x1)
    x1 -= lr * gradient
    point.set_data([x1], [f(x1)])
    value_display.set_text("Min = {:.2f}, {:.2f}".format(x1, f(x1)))
    return point, value_display


ani = FuncAnimation(
    fig, update, frames=np.arange(0, n_iterations), init_func=init, blit=True
)

ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Function and Gradient Descent Animation")
ax.grid(True)
plt.show()
