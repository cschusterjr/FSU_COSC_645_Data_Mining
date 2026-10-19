from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Locate project directories
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data" / "raw"
results_dir = project_dir / "results"

results_dir.mkdir(parents=True, exist_ok=True)

# Load gene-expression data and cancer labels
data = pd.read_csv(data_dir / "data.csv", index_col=0)
labels = pd.read_csv(data_dir / "labels.csv", index_col=0)

assert data.index.equals(labels.index), "Sample IDs do not match!"

X = data
y = labels["Class"]

# Create the same stratified split used in Milestone 3
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Remove constant features using training data only
variance_filter = VarianceThreshold(threshold=0.0)

X_train_filtered = variance_filter.fit_transform(X_train)
X_test_filtered = variance_filter.transform(X_test)

# Standardize features using training data only
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_filtered)
X_test_scaled = scaler.transform(X_test_filtered)

print("PRINCIPAL COMPONENT ANALYSIS")
print("-" * 40)
print(f"Training samples: {X_train_scaled.shape[0]}")
print(f"Features after filtering: {X_train_scaled.shape[1]}")

# Fit PCA using all available training components
# This allows us to calculate the 90%, 95%, and 99% thresholds
max_components = min(
    X_train_scaled.shape[0],
    X_train_scaled.shape[1]
)

pca = PCA(
    n_components=max_components,
    svd_solver="full"
)

pca.fit(X_train_scaled)

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)

print("\nExplained Variance Results:")

for threshold in [0.90, 0.95, 0.99]:
    if cumulative_variance[-1] >= threshold:
        components = np.searchsorted(
            cumulative_variance,
            threshold
        ) + 1

        print(
            f"{threshold:.0%} variance: "
            f"{components} components"
        )
    else:
        print(
            f"{threshold:.0%} variance: "
            f"Not reached within {max_components} components"
        )

print(
    f"\nVariance explained by {max_components} components: "
    f"{cumulative_variance[-1]:.2%}"
)

# Plot cumulative explained variance
plt.figure(figsize=(10, 6))

plt.plot(
    range(1, max_components + 1),
    cumulative_variance * 100,
    color="steelblue",
    linewidth=2
)

for threshold in [90, 95, 99]:
    plt.axhline(
        y=threshold,
        color="gray",
        linestyle="--",
        alpha=0.6
    )

plt.title("Cumulative Explained Variance Using PCA")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.ylim(0, 105)
plt.grid(alpha=0.3)

plt.tight_layout()

output_path = results_dir / "pca_explained_variance.png"

plt.savefig(output_path, dpi=300)
plt.close()

print(f"\nPCA chart saved to: {output_path}")


# Visualize the first two principal components
print("\nPCA CANCER TYPE VISUALIZATION")
print("-" * 40)

# Transform training data using the fitted PCA model
X_train_pca = pca.transform(X_train_scaled)

# Create a DataFrame for visualization
pca_df = pd.DataFrame({
    "PC1": X_train_pca[:, 0],
    "PC2": X_train_pca[:, 1],
    "Cancer Type": y_train.to_numpy()
})

# Create a scatter plot
plt.figure(figsize=(10, 7))

for cancer_type in sorted(pca_df["Cancer Type"].unique()):
    subset = pca_df[
        pca_df["Cancer Type"] == cancer_type
    ]

    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        label=cancer_type,
        alpha=0.7,
        s=35
    )

plt.title("Cancer Gene Expression: First Two Principal Components")
plt.xlabel(
    f"Principal Component 1 "
    f"({pca.explained_variance_ratio_[0]:.1%} variance)"
)
plt.ylabel(
    f"Principal Component 2 "
    f"({pca.explained_variance_ratio_[1]:.1%} variance)"
)

plt.legend(title="Cancer Type")
plt.grid(alpha=0.3)
plt.tight_layout()

# Save the visualization
output_path = results_dir / "pca_cancer_scatter.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"PCA scatter plot saved to: {output_path}")
