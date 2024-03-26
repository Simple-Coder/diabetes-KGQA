import matplotlib.pyplot as plt

from matplotlib.font_manager import findSystemFonts

# 获取可用字体列表
available_fonts = findSystemFonts()

# 打印可用字体列表
for font in available_fonts:
    print(font)

# 设置中文字体为宋体，英文字体为新罗马
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置为宋体
plt.rcParams['font.family'] = 'Times New Roman'  # 英文字体设置为新罗马字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 迭代次数
iterations = [i for i in range(5, 201, 10)]

# 模型A的MRR值
mrr_model_A = [0.638, 0.678, 0.628, 0.680, 0.685, 0.705, 0.680, 0.692, 0.665, 0.683, 0.691, 0.699, 0.710, 0.705, 0.713,
               0.714, 0.694, 0.692, 0.689, 0.715]

mrr_model_B = [0.599, 0.638, 0.648, 0.640, 0.638, 0.639, 0.645, 0.624, 0.645, 0.647, 0.650, 0.652, 0.672, 0.671, 0.695,
               0.681, 0.660, 0.632, 0.668, 0.645]

mrr_model_C = [0.610, 0.615, 0.639, 0.640, 0.625, 0.628, 0.633, 0.634, 0.672, 0.649, 0.631, 0.621, 0.629, 0.642, 0.655,
               0.655, 0.650, 0.645, 0.621, 0.672]

# 绘制MRR随迭代次数变化的图表
plt.plot(iterations[:len(mrr_model_A)], mrr_model_A, color='#FF5733', label='RLKRMIPI')  # 模型A的折线，红色
plt.plot(iterations[:len(mrr_model_B)], mrr_model_B, color='#3399FF', label='RLKRMIPI-ActionDropout')  # 模型B的折线，绿色虚线
plt.plot(iterations[:len(mrr_model_C)], mrr_model_C, color='#FFD700', label='RLKRMIPI-Reward')  # 模型C的折线，蓝色点划线

plt.title('NELL-995')
# plt.xlabel('迭代次数')
plt.xlabel('Iterations')
plt.ylabel('MRR')
plt.legend()  # 添加图例
plt.xticks(ticks=range(0, 201, 50))  # 设置x轴刻度，每隔50个单位显示一个刻度
plt.yticks(ticks=[0.4, 0.5, 0.6, 0.7, 0.8])  # 设置y轴刻度，只显示指定刻度
plt.grid(False)  # 移除网格线
plt.xlim(0, 200)  # 设置x轴范围
plt.xlim(0, iterations[-1])  # 设置x轴范围为最后一个迭代次数
# 将图例放在右下角
plt.legend(loc='lower right')

# 保存图像到D盘
plt.savefig('./NELL-995_Pic.png')
plt.show()
