import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('linreg_data.csv', delimiter=',')
x, y = data[:, 0], data[:, 1]
degree = 2
p = np.poly1d(np.polyfit(x, y, degree))
plt.scatter(x, y)
plt.plot(np.sort(x), p(np.sort(x)), color='red')
plt.show()
