from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

# Locate project directories
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data" / "raw"

#Load the gene-expression data and labels
data = pd.read_csv(data_dir / "data.csv", index_col=0)
labels = pd.read_csv(data_dir / "labels.csv", index_col=0)

# Verify that sample identifiers match
assert data.index.equals(labels.index), "Sample IDs do not match!"

# Separate features and target labels
X = data
y = labels["Class"]

# Create a stratified training and testing split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("DATA PREPROCESSING")
print("-" * 30)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"Features: {X_train.shape[1]}")

# Identify and remove constant features using training data only
variance_filter = VarianceThreshold(threshold=0.0)

X_train_filtered = variance_filter.fit_transform(X_train)
X_test_filtered = variance_filter.transform(X_test)

print(f"\nOriginal features: {X_train.shape[1]}")
print(f"Filtered features: {X_train_filtered.shape[1]}")
print(f"Removed {X_train.shape[1] - X_train_filtered.shape[1]} constant features.")

# Standardize features using training data only
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_filtered)
X_test_scaled = scaler.transform(X_test_filtered)

print(f"\nStandarization complete.")
print(f"Training data shape: {X_train_scaled.shape}")
print(f"Testing data shape: {X_test_scaled.shape}")

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())