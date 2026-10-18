from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Locate project directories
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data" / "raw"
results_dir = project_dir / "results"

# Ensure the results directory exists
results_dir.mkdir(parents=True, exist_ok=True)

#Load cancer classification labels
labels = pd.read_csv(data_dir / "labels.csv", index_col=0)

#Count the number of samples in each cancer class
class_counts = labels["Class"].value_counts()

#Create a bar chart
plt.figure(figsize=(10, 6))

sns.barplot(
    x=class_counts.index,
    y=class_counts.values,
    color="steelblue"
)

plt.title("Distribution of Cancer Types in the Gene Expression Dataset")
plt.xlabel("Cancer Type")
plt.ylabel("Number of Samples")

#Display sample counts above each bar
for index, count in enumerate(class_counts.values):
    plt.text(index, count + 3, str(count), ha='center', va='bottom')

plt.ylim(0, class_counts.max() * 1.15)
plt.tight_layout()

#Save the chart for the final report
output_path = results_dir / "cancer_type_distribution.png"
plt.savefig(output_path, dpi=300)
plt.close()

print("EXPLORATORY DATA ANALYSIS")
print("-" * 30)
print("Cancer Type Distribution:")
print(class_counts)

print(f"\nChart saved to: {output_path}")


# Load the gene-expression measurements
data = pd.read_csv(data_dir / "data.csv", index_col=0)

print("\nGENE EXPRESSION ANALYSIS")
print("-" * 30)

print(f"Total gene features: {data.shape[1]}")
print(f"Total samples: {data.shape[0]}")

# Calculate summary statistics across all gene-expression values
expression_values = data.to_numpy().flatten()

print(f"Mean Expression: {expression_values.mean():.4f}")
print(f"Median Expression: {pd.Series(expression_values).median():.4f}")
print(f"Standard Deviation: {expression_values.std():.4f}")
print(f"Minimum Expression: {expression_values.min():.4f}")
print(f"Maximum Expression: {expression_values.max():.4f}")

# Create a histogram of gene-expression values
plt.figure(figsize=(10, 6))

plt.hist(
    expression_values,
    bins=50,
    color="steelblue",
    edgecolor="black"
)

plt.title("Distribution of Gene Expression Values")
plt.xlabel("Gene Expression Value")
plt.ylabel("Frequency")

plt.tight_layout()

# Save the visualization
output_path_hist = results_dir / "gene_expression_distribution.png"
plt.savefig(output_path_hist, dpi=300)
plt.close()

print(f"\nHistogram saved to: {output_path_hist}")