import pandas as pd
import matplotlib.pyplot as plt

# Load data
Ns = [500, 1000, 2000]
plt.figure(figsize=(10, 6))
colors = ['navy', 'royalblue', 'skyblue'] # 颜色设定
markers = ['o', 'x', '^']  # 点设定
for i, N in enumerate(Ns):
    file_path = f'./OutcomeData/model2_output_{N}.txt'
    data = pd.read_csv(file_path, sep=' ', names=['K', 'r'])
    # 添加数据
    plt.plot(data['K'], data['r'], label=f'N={N}', color=colors[i], marker=markers[i])

# 绘制图像
plt.title('Synchronization vs. Coupling Strength for Different System Sizes')
plt.xlabel('Coupling Strength (K)')
plt.ylabel('Order Parameter (r)')
plt.legend()
plt.grid(True)

# 保存图像
output_path = './Output/synchronization_vs_coupling_strength_all_sizes.png'
plt.savefig(output_path)
plt.show()
