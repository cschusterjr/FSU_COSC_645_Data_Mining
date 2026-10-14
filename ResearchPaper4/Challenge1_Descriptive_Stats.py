# Descriptive statistics and distribution plot for the Iris dataset

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# ---------------------------------------------------------------
# Step 1: Load the dataset
# ---------------------------------------------------------------
# as_frame=True returns the data as a pandas DataFrame,
# which makes statistical calculations easy.
iris = load_iris(as_frame=True)
df = iris.frame  # Table containing the measurements and the species code

# Replace the numeric species codes (0, 1, 2) with readable names
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
df = df.drop(columns="target")

print("=" * 60)
print("IRIS DATASET OVERVIEW")
print("=" * 60)
print(f"Number of observations: {len(df)}")
print(f"Number of variables:    {df.shape[1]}")
print("\nFirst 5 rows:")
print(df.head())

# ---------------------------------------------------------------
# Step 2: Select the numerical variables
# ---------------------------------------------------------------
# The 'species' column is categorical, so we keep only numeric columns.
numeric_df = df.select_dtypes(include="number")

# ---------------------------------------------------------------
# Step 3: Calculate descriptive statistics
# ---------------------------------------------------------------
# .agg() applies several functions to every numeric column at once.
stats = numeric_df.agg(["mean", "median", "std", "min", "max"]).T
stats.columns = ["Mean", "Median", "Std Dev", "Min", "Max"]

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS (all measurements in cm)")
print("=" * 60)
print(stats.round(2).to_string())

# ---------------------------------------------------------------
# Step 4: Print a short interpretation of one variable
# ---------------------------------------------------------------
variable = "sepal length (cm)"
mean_val = stats.loc[variable, "Mean"]
median_val = stats.loc[variable, "Median"]
std_val = stats.loc[variable, "Std Dev"]

print(f"\nExample interpretation for '{variable}':")
print(f"  The average value is {mean_val:.2f} cm and the median is {median_val:.2f} cm.")
print(f"  A typical observation lies about {std_val:.2f} cm from the mean.")

# ---------------------------------------------------------------
# Step 5: Visualize the distribution of one variable
# ---------------------------------------------------------------
# A histogram shows how often values fall into each range (bin).
plt.figure(figsize=(8, 5))
plt.hist(numeric_df[variable], bins=15, color="steelblue", edgecolor="black")

# Add vertical lines for the mean and median for context
plt.axvline(mean_val, color="red", linestyle="--", label=f"Mean = {mean_val:.2f}")
plt.axvline(median_val, color="green", linestyle="-", label=f"Median = {median_val:.2f}")

# Label the chart so it is easy to read
plt.title("Distribution of Sepal Length in the Iris Dataset")
plt.xlabel("Sepal length (cm)")
plt.ylabel("Frequency (number of flowers)")
plt.legend()
plt.tight_layout()
plt.show()