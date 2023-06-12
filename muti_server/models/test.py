"""
Created by xiedong
@Date: 2023/6/12 20:33
"""


def compute_gradient_w1():
    return 1


def compute_gradient_w2():
    return 2


w1 = 1
w2 = 2
# 假设模型中有两个参数：w1和w2
# 初始化梯度为0
w1_grad = 0
w2_grad = 0

# 第一次反向传播计算梯度
w1_grad += compute_gradient_w1()  # 假设第一次计算得到的梯度为1
w2_grad += compute_gradient_w2()  # 假设第一次计算得到的梯度为2

# 第二次反向传播计算梯度
w1_grad += compute_gradient_w1()  # 假设第二次计算得到的梯度为3
w2_grad += compute_gradient_w2()  # 假设第二次计算得到的梯度为4

# 参数更新
learning_rate = 0.1
w1 -= learning_rate * w1_grad  # 根据梯度更新参数w1
w2 -= learning_rate * w2_grad  # 根据梯度更新参数w2

# 下一轮迭代前，需要将梯度清零
w1_grad = 0
w2_grad = 0

# 继续进行下一轮迭代...
