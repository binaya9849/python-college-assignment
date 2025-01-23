import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-10, 10, 400)
y1 = 2*x + 1
y2 = 2*x + 2
y3 = 2*x + 3
plt.plot(x, y1, linestyle='-', color='#4287f5', label='y = 2x + 1')
plt.plot(x, y2, linestyle='--', color='#2e76e8', label='y = 2x + 2')
plt.plot(x, y3, linestyle='-.', color='#1d5cbf', label='y = 2x + 3')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Graph of y = 2x + c for c = 1, 2, 3')
plt.grid(True) 
plt.legend()
plt.show()