import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("dataset.csv")

# Display first few rows
print("Dataset Preview:")
print(data.head())

# Basic statistics
print("\nStatistical Summary:")
print(data.describe())

# Correlation analysis
correlation = data.corr()
print("\nCorrelation Matrix:")
print(correlation)

# Visualization - correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Healthcare Data Correlation Heatmap")
plt.show()

# Outcome distribution
sns.countplot(x="Outcome", data=data)
plt.title("Disease Outcome Distribution")
plt.show()