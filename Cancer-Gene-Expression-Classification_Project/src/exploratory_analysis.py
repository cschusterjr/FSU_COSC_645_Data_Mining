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