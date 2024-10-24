import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
data = pd.read_csv('scatter.csv')  # Update with the correct file path

# Extract SafeCount and DangerousCount
safe_counts = data['SafeCount']
dangerous_counts = data['DangereousCount']

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(safe_counts, dangerous_counts, color='blue', marker='x', label='Data Points')

# Line of best fit
m, b = np.polyfit(safe_counts, dangerous_counts, 1)
plt.plot(safe_counts, [m*x + b for x in safe_counts], color='red', label=f'Line of Best Fit (y = {m:.2f}x + {b:.2f})')

# Labels and title
plt.xlabel('Safe Count')
plt.ylabel('Dangerous Count')
plt.title('Relationship between Safe and Dangerous Fire Starting Behaviors')
plt.legend()

# Show plot
plt.grid(True)
plt.show()
