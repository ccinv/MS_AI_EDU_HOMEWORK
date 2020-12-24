import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

N = 1000
data = pd.read_csv("mlm.csv")

x, y = [], []
for i in range(0, N):
	x.append([1, data.values[i][0], data.values[i][1]])
	y.append(data.values[i][2])
x = np.array(x)
y = np.array(y).reshape(N, 1)
w = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(x), x)), np.transpose(x)), y)
print("W:", w)

x_p, y_p, z_p = [], [], []
for i in range(0, N):
	x_p.append(data.values[i][0])
	y_p.append(data.values[i][1])
	z_p.append(data.values[i][2])
ax = plt.axes(projection="3d")
ax.scatter3D(x_p, y_p, z_p)

x_drawing = np.linspace(0, 100)
y_drawing = np.linspace(0, 100)
X_drawing, Y_drawing = np.meshgrid(x_drawing, y_drawing)
ax.plot_surface(X = X_drawing, Y = Y_drawing, Z = X_drawing * w[1] + Y_drawing * w[2] + w[0], color = 'r', alpha = 0.3)

ax.view_init(elev = 30, azim = 30)
plt.show()
