import numpy as np
import matplotlib.pyplot as plt

def surf_den(x, T):
  return np.exp(-x / T) / (x * T**1.5)

x = np.logspace(-1,1.5,100)

y1 = surf_den(x, 1)
y2 = surf_den(x, 1.5)
y3 = surf_den(x, 2)
y4 = surf_den(x,4)

plt.plot(x,y1, label='T=1')
plt.plot(x,y2, label='T=1.5')
plt.plot(x,y3, label='T=2')
plt.plot(x,y4, label='T=4')
plt.xscale('log')
plt.yscale('log')
xlim1,xlim2,ylim1,ylim2 = plt.axis()
plt.axis((xlim1,10**1.5,10**-5,10))
plt.xlabel('x')
plt.ylabel('S')
plt.legend()

plt.savefig("q2d.eps")

plt.show()
