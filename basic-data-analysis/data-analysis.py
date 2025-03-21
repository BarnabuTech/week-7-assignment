import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
try:
    # Load the Iris dataset from sklearn
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    # Display first few rows
    print("First 5 rows of the dataset:")
    print(df.head())
    
    # Check data types and missing values
    print("\nDataset Info:")
    print(df.info())
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Clean data if necessary (no missing values in this dataset)
except Exception as e:
    print(f"Error loading dataset: {e}")

# Task 2: Basic Data Analysis
print("\nBasic statistics:")
print(df.describe())

# Grouping by species and calculating mean
species_mean = df.groupby('species').mean()
print("\nAverage values per species:")
print(species_mean)

# Task 3: Data Visualization
plt.figure(figsize=(12, 8))

# Line Plot: Trend of Sepal Length over indices
plt.subplot(2, 2, 1)
plt.plot(df.index, df['sepal length (cm)'], marker='o', linestyle='-')
plt.title("Trend of Sepal Length")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")

# Bar Chart: Average petal length per species
plt.subplot(2, 2, 2)
sns.barplot(x=species_mean.index, y=species_mean['petal length (cm)'])
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")

# Histogram: Distribution of Sepal Width
plt.subplot(2, 2, 3)
plt.hist(df['sepal width (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")

# Scatter Plot: Sepal Length vs. Petal Length
plt.subplot(2, 2, 4)
sns.scatterplot(x=df['sepal length (cm)'], y=df['petal length (cm)'], hue=df['species'], palette='bright')
plt.title("Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")

plt.tight_layout()
plt.show()

# Findings:
# - Setosa species has the shortest petal length on average.
# - Virginica species has the highest petal length and sepal length.
# - The histogram shows most sepal widths are between 2.5 and 3.5 cm.
# - Scatter plot indicates a positive correlation between sepal length and petal length.