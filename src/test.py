import matplotlib.pyplot as plt
import numpy as np

# 假设你有以下点的坐标、分类和名称
points = np.array([[1, 2], [2, 3], [3, 1], [5, 4], [4, 6], [6, 5]])
categories = np.array([0, 0, 1, 1, 0, 1])  # 点的分类
names = np.array(['A', 'B', 'C', 'D', 'E', 'F'])  # 点的名称

# 获取每个类别的索引
category_0_idx = np.where(categories == 0)
category_1_idx = np.where(categories == 1)

# 绘制不同类别的点
plt.scatter(points[category_0_idx, 0], points[category_0_idx, 1], color='red', label='Category 0')
plt.scatter(points[category_1_idx, 0], points[category_1_idx, 1], color='blue', label='Category 1')

# 给第一个类别的点标注名称
for i in category_0_idx[0]:
    plt.text(points[i, 0], points[i, 1], names[i], fontsize=9, ha='right')

# 设置图例和标题
plt.legend()
plt.title('Scatter plot with categorized points and names')
plt.xlabel('X axis')
plt.ylabel('Y axis')

# 显示图像
plt.show()
