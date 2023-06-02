import numpy as np

# 定义初始条件和边界条件
L = 1
T = 0.1
nx = 100
nt = 100
dx = L / (nx - 1)
dt = T / nt
alpha = 0.5
u = np.zeros((nt, nx))
u[0, :] = np.sin(np.pi * np.linspace(0, L, nx))

# 使用差分法求解偏微分方程
for n in range(0, nt-1):
    for i in range(1, nx-1):
        u[n+1, i] = u[n, i] + alpha * dt / dx**2 * (u[n, i+1] - 2*u[n, i] + u[n, i-1])

# 可视化结果
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)
X, T = np.meshgrid(x, t)
ax.plot_surface(X, T, u)
plt.show()
