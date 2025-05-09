import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

size_x = 60
x = np.arange(0, size_x, 1.0)
y = x.copy()
X,Y = np.meshgrid(x, y)
Z = 10*np.sin(np.pi*X/20)
# set all 0.22*Y < 0 to 0
Z[Z < 0] = 0
Z += 10 + 0.22*Y

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_box_aspect((1, 1, Z.max()/size_x))
surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm,linewidth=0,antialiased=False)

plt.show()