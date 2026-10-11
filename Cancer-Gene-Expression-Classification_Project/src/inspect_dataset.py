from pathlib import Path
import pandas as pd

#Locate the project and dataset folders
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data" / "raw"

#Load the datasets 
data = pd.read_csv(data_dir / "data.csv", index_col=0)
labels = pd.read_csv(data_dir / "labels.csv", index_col=0)

#Display basic information about the dataset
print("CANCER GENE EXPRESSION DATASET")
print("-" * 40)

print(f"Number of samples: {data.shape[0]}")
print(f"Number of gene features: {data.shape[1]}")
print(f"Number of missing values: {data.isnull().sum().sum()}")

print("\nCancer Type Distribution:")
print(labels["Class"].value_counts())

print("\nFirst 5 rows of gene expression dataset:")
print(data.iloc[:5, :5])

# Verify that sample identifiers match
print("\nSample identifiers match:",
      data.index.equals(labels.index))



# Additional data quality checks
print("\nADDITIONAL DATA QUALITY CHECKS")
print("-" * 40)

# Check for duplicate sample identifiers
duplicate_ids = data.index.duplicated().sum()
print(f"Duplicate sample identifiers: {duplicate_ids}")

# Check for duplicate gene-expression profiles
duplicate_profiles = data.duplicated().sum()
print(f"Duplicate gene-expression profiles: {duplicate_profiles}")

# Check for gene features with no variation
constant_features = (data.nunique() <= 1).sum()
print(f"Constant gene features: {constant_features}")

# Check for infinite values
import numpy as np

infinite_values = np.isinf(data.to_numpy()).sum()
print(f"Infinite values: {infinite_values}")

# Check the overall range of gene-expression values
print(f"Minimum expression value: {data.min().min():.4f}")
print(f"Maximum expression value: {data.max().max():.4f}")
