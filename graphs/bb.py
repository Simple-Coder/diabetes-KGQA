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
mrr_model_A = [0.310, 0.330, 0.350, 0.325, 0.340, 0.355, 0.370, 0.365, 0.368, 0.380, 0.410, 0.440, 0.410, 0.420, 0.430,
               0.425, 0.415, 0.434, 0.433, 0.433]

mrr_model_B = [0.280, 0.285, 0.290, 0.310, 0.300, 0.295, 0.320, 0.325, 0.330, 0.360, 0.392, 0.395, 0.370, 0.355, 0.380,
               0.375, 0.385, 0.390, 0.381, 0.392]

mrr_model_C = [0.270, 0.275, 0.270, 0.285, 0.295, 0.310, 0.305, 0.315, 0.320, 0.330, 0.370, 0.379, 0.389, 0.363, 0.376,
               0.382, 0.374, 0.364, 0.386, 0.388]

# 绘制MRR随迭代次数变化的图表
plt.plot(iterations[:len(mrr_model_A)], mrr_model_A, color='#FF5733', label='RLKRMIPI')  # 模型A的折线，红色
plt.plot(iterations[:len(mrr_model_B)], mrr_model_B, color='#3399FF', label='RLKRMIPI-ActionDropout')  # 模型B的折线，绿色虚线
plt.plot(iterations[:len(mrr_model_C)], mrr_model_C, color='#FFD700', label='RLKRMIPI-Reward')  # 模型C的折线，蓝色点划线

plt.title('FB15k-237')
# plt.xlabel('迭代次数')
plt.xlabel('Iterations')
plt.ylabel('MRR')
plt.legend()  # 添加图例
plt.xticks(ticks=range(0, 201, 50))  # 设置x轴刻度，每隔50个单位显示一个刻度
plt.yticks(ticks=[0.2, 0.3, 0.4, 0.5, 0.6])  # 设置y轴刻度，只显示指定刻度
plt.grid(False)  # 移除网格线
plt.xlim(0, 200)  # 设置x轴范围
plt.xlim(0, iterations[-1])  # 设置x轴范围为最后一个迭代次数
# 将图例放在右下角
plt.legend(loc='lower right')

# 保存图像到D盘
plt.savefig('./FB15k-237_pic.png')
plt.show()
