#minimize 2-d functions

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize




# Define the function that we are interested in
def function(x):
    return (4 - 2.1*x[0]**2 + 2 * x[1] **2)



############Plot color plot in 2D


# Make a grid to evaluate the function (for plotting)
x = np.linspace(-2, 2)
y = np.linspace(-1, 1)
xg, yg = np.meshgrid(x, y)

plt.figure()
plt.imshow(function([xg, yg]), extent=[-2, 2, -1, 1], origin="lower")
plt.colorbar()
plt.show()

########## 3D plot 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xg, yg, function([xg, yg]), rstride=1, cstride=1,
                       cmap=plt.cm.jet, linewidth=0, antialiased=False)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('Surfaceplot function')
plt.show()

############Minimize
x_min = optimize.minimize(function, x0=[0, 0])

plt.figure()



# Show the function in 2D
plt.imshow(function([xg, yg]), extent=[-2, 2, -1, 1], origin="lower")
plt.colorbar()
plt.savefig('surfaceplot.png')
# And the minimum that we've found:
plt.scatter(x_min.x[0], x_min.x[1])

plt.show()
