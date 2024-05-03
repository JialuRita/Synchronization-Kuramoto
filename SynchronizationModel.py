import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from scipy.integrate import odeint

# 基于Kuramoto库进行模型复现
class Kuramoto:

    def __init__(self, coupling=1, dt=0.01, T=10, n_nodes=None, natfreqs=None):
        '''
        coupling: float
            Coupling strength. Default = 1. Typical values range between 0.4-2
        dt: float
            Delta t for integration of equations.
        T: float
            Total time of simulated activity.
            From that the number of integration steps is T/dt.
        n_nodes: int, optional
            Number of oscillators.
            If None, it is inferred from len of natfreqs.
            Must be specified if natfreqs is not given.
        natfreqs: 1D ndarray, optional
            Natural oscillation frequencies.
            If None, then new random values will be generated and kept fixed
            for the object instance.
            Must be specified if n_nodes is not given.
            If given, it overrides the n_nodes argument.
        '''
        if n_nodes is None and natfreqs is None:
            raise ValueError("n_nodes or natfreqs must be specified")

        self.dt = dt
        self.T = T
        self.coupling = coupling

        if natfreqs is not None:
            self.natfreqs = natfreqs
            self.n_nodes = len(natfreqs)
        else:
            self.n_nodes = n_nodes
            self.natfreqs = np.random.normal(size=self.n_nodes)

    def init_angles(self):
        '''
        Random initial random angles (position, "theta").
        '''
        return 2 * np.pi * np.random.random(size=self.n_nodes)

    def derivative(self, angles_vec, t, adj_mat, coupling):
        '''
        Compute derivative of all nodes for current state, defined as

        dx_i    natfreq_i + k  sum_j ( Aij* sin (angle_j - angle_i) )
        ---- =             ---
         dt                M_i

        t: for compatibility with scipy.odeint
        '''
        assert len(angles_vec) == len(self.natfreqs) == len(adj_mat), \
            'Input dimensions do not match, check lengths'

        angles_i, angles_j = np.meshgrid(angles_vec, angles_vec)
        interactions = adj_mat * np.sin(angles_j - angles_i)  # Aij * sin(j-i)

        dxdt = self.natfreqs + coupling * interactions.sum(axis=0)  # sum over incoming interactions
        return dxdt

    # 修改intergrate函数
    def integrate(self, angles_vec, adj_mat):
        '''Updates all states by integrating state of all nodes'''
        # Coupling term (k / Mj) is constant in the integrated time window.
        # Compute it only once here and pass it to the derivative function
        # n_interactions = (adj_mat != 0).sum(axis=0)  # number of incoming interactions
        # coupling = self.coupling / n_interactions  # normalize coupling by number of interactions

        coupling=self.coupling/self.n_nodes  # lamda定义为K/N
        t = np.linspace(0, self.T, int(self.T/self.dt))
        timeseries = odeint(self.derivative, angles_vec, t, args=(adj_mat, coupling))
        return timeseries.T  # transpose for consistency (act_mat:node vs time)

    # 修改run函数
    def run(self, adj_mat=None, angles_vec=None):
        '''
        adj_mat: 2D nd array
            Adjacency matrix representing connectivity.
        angles_vec: 1D ndarray, optional
            States vector of nodes representing the position in radians.
            If not specified, random initialization [0, 2pi].

        Returns
        -------
        act_mat: 2D ndarray
            Activity matrix: node vs time matrix with the time series of all
            the nodes.
        '''
        if angles_vec is None:
            angles_vec = self.init_angles()

        act_mat = self.integrate(angles_vec, adj_mat)

        # 计算每个时间步相位的序参量
        r_values = [self.phase_coherence(act_mat[:, step]) for step in range(act_mat.shape[1])]

        return act_mat, r_values  # 返回序参量

    # 定义序参量
    @staticmethod
    def phase_coherence(angles_vec):
        '''
        Compute global order parameter R_t - mean length of resultant vector
        '''
        suma = sum([(np.e ** (1j * i)) for i in angles_vec])
        return abs(suma / len(angles_vec))

    def mean_frequency(self, act_mat, adj_mat):
        '''
        Compute average frequency within the time window (self.T) for all nodes
        '''
        assert len(adj_mat) == act_mat.shape[0], 'adj_mat does not match act_mat'
        _, n_steps = act_mat.shape

        # Compute derivative for all nodes for all time steps
        dxdt = np.zeros_like(act_mat)
        for time in range(n_steps):
            dxdt[:, time] = self.derivative(act_mat[:, time], None, adj_mat)

        # Integrate all nodes over the time window T
        integral = np.sum(dxdt * self.dt, axis=1)
        # Average across complete time window - mean angular velocity (freq.)
        meanfreq = integral / self.T
        return meanfreq


if __name__=='__main__':
    # 设置参数
    N_values = [100, 200, 300, 500, 1000]  # BA网络的规模
    lambda_values = np.arange(0, 101, 1)  # 耦合强度的范围
    dt = 0.01  # 时间步长
    T = 40  # 总模拟时间
    # 对于每个规模，计算耦合强度和同步参数
    results = {N: {} for N in N_values}

    for N in N_values:
        G = nx.barabasi_albert_graph(N, m=3)  # 创建BA网络
        adj_mat = nx.to_numpy_array(G)  # 转换为邻接矩阵
        natfreq = np.random.uniform(-0.5, 0.5, N)  # 生成N个介于-0.5到0.5之间的随机数
        # 计算每个耦合强度的同步参数
        for lambda_val in lambda_values:
            kuramoto = Kuramoto(coupling=lambda_val, dt=dt, T=T, n_nodes=N,natfreqs=natfreq)  # 使用修改的Kuramoto代码
            _, r_values = kuramoto.run(adj_mat=adj_mat)
            # 计算最终时间步的同步参数
            sync_param = np.mean(r_values[-int(len(r_values)*0.1):])
            results[N][lambda_val] = sync_param  # 存储结果
        # 计算平均度
        total_degree = sum(dict(G.degree()).values())
        num_nodes = len(G)
        average_degree = total_degree / num_nodes
        # 计算度的平方的平均值
        degree_squared_sum = sum(degree ** 2 for node, degree in G.degree())
        average_degree_squared = degree_squared_sum / len(G)
        Kc = 2 * average_degree / (np.pi * average_degree_squared)
        
        #绘制图像---序参量r随耦合强度λ的变化
        plt.figure(figsize=(10, 6))
        lambda_normalized = np.array(list(results[N].keys())) / N  # 标准化lambda值
        sync_values = list(results[N].values())  # 获取同步参数值
        color_map = {100: 'royalblue', 200: 'lightgreen', 300: 'red', 500: 'purple', 1000: 'yellow'}
        plt.scatter(lambda_normalized, sync_values, color=color_map[N], label=f'N={N}')
        plt.vlines(Kc, 0, 1, color='orange', linestyle='--', label=f'λc={Kc:.3f}')  # 添加垂直线表示Kc
        # 设置坐标轴标签和标题
        plt.xlabel('$\lambda$', fontsize=12)
        plt.ylabel('$r$', fontsize=12)
        plt.title('Synchronization vs. Coupling Strength for Different System Sizes', fontsize=14)
        # 添加图例
        plt.legend(title='System Size', fontsize=10, title_fontsize=11, loc='best')
        # 添加网格
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        # 保存图像
        plt.savefig(f'{N}_{Kc}.png')
        #plt.show()

    # 绘制总体散点图
    plt.figure(figsize=(10, 6))
    for N in results:
        lambda_normalized = np.array(list(results[N].keys())) / N  # 标准化lambda值
        sync_values = list(results[N].values())  # 获取同步参数值
        color_map = {100: 'royalblue', 200: 'lightgreen', 300: 'red', 500: 'purple', 1000: 'yellow'}
        plt.scatter(lambda_normalized, sync_values, color=color_map[N], label=f'N={N}')
    plt.xlabel('$\lambda$')
    plt.ylabel('$r$')
    plt.title('Synchronization vs. Coupling strength for different system sizes')
    plt.legend()
    plt.grid(True)
    #plt.show()