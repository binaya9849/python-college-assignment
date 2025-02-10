import numpy as np
import matplotlib.pyplot as plt

n_values = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for n in n_values:
    dice1 = np.random.randint(1, 7, n)
    dice2 = np.random.randint(1, 7, n)
    sums = dice1 + dice2
    h, h2 = np.histogram(sums, range(2, 14))
    plt.bar(h2[:-1], h / n, width=0.8, alpha=0.5, color='red' ) 
    plt.title(f"Histogram of Dice Sums for n={n}")
    plt.xlabel("Sum of Dice")
    plt.ylabel("Frequency")
    plt.show()

print("The histogram of the sums of six dice throws gradually smooths out and becomes symmetric as we throw more dice, and ultimately converges into a triangular distribution with peak 7, which is the most likely sum of 2 fair dice thrown. With fewer data points (like 500 or 1000 throws), we see that these frequencies vary quite a bit, adding a lot of noise to the distribution and making it less clear cut. But as samples increase this behavior reduces and the distribution approaches the theoretically expected behaviour.")

print("In statistics, regression toward the mean is the phenomenon where if one sample of a random variable is extreme, the next sampling of the same random variable is likely to be closer to its mean.")
