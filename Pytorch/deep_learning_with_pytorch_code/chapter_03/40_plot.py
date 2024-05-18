import matplotlib.pyplot as plt
import torch

# create tensors
x = torch.linspace(-1, 1, 100)
y = torch.linspace(-2, 2, 100)
# create the surface
xx, yy = torch.meshgrid(x, y, indexing="xy")  # xy-indexing is matching numpy
z = torch.sqrt(1 - xx**2 - (yy/2)**2)
print(xx)

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection="3d")
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([0, 2])
ax.plot_surface(xx, yy, z, cmap="cividis")
ax.view_init(45, 35)
plt.show()
