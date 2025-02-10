import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('linreg_data.csv', delimiter=',')
x = data[:, 0]
y = data[:, 1]
slope, intercept = np.polyfit(x, y, 1)
mymodel = slope * x + intercept
plt.scatter(x, y, label='Data Points')
plt.plot(x, mymodel, color='red', label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')
plt.show()