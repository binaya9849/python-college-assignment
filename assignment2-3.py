import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filePath = 'weight-height.csv'
data = pd.read_csv(filePath)


length_in_inches = data['Height'].values  
weight_in_pounds = data['Weight'].values  


length_in_cm = length_in_inches * 2.54  
weight_in_kg = weight_in_pounds * 0.453592 


mean_length_cm = np.mean(length_in_cm)
mean_weight_kg = np.mean(weight_in_kg)


plt.figure(figsize=(8, 6))
plt.hist(length_in_cm, bins=20, color='yellow', edgecolor='black', alpha=0.7)
plt.title('Histogram of Heights (cm)', fontsize=10)
plt.xlabel('Height (cm)', fontsize=8)
plt.ylabel('Frequency', fontsize=8)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


print(f"Mean Height: {mean_length_cm:.2f} cm")
print(f"Mean Weight: {mean_weight_kg:.2f} kg")