
"""
COSC 645 Semester Project
Cancer Type Classification Using Gene Expression Data

"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------
# 1. Set up project paths
# --------------------------------------------------

project_dir = Path(__file__).resolve().parents[1]
results_dir = project_dir / "results"

input_file = results_dir / "pca_model_comparison.csv"

print("COSC 645 - Model Results Visualization")
print("-" * 50)


# --------------------------------------------------
# 2. Load model comparison results
# --------------------------------------------------

results = pd.read_csv(input_file)

print(f"Loaded {len(results)} model comparison results.")


# --------------------------------------------------
# 3. Define chart settings
# --------------------------------------------------

models = [
    "Logistic Regression",
    "KNN",
    "Gaussian Naive Bayes",
    "Linear SVM"
]

configurations = [
    "Original Features",
    "PCA 310 Components",
    "PCA 434 Components"
]

colors = ["#4C78A8", "#F58518", "#54A24B"]

x = np.arange(len(models))
width = 0.25


# --------------------------------------------------
# 4. Chart 1: Model accuracy comparison
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 6))

for i, config in enumerate(configurations):

    subset = results[
        results["Configuration"] == config
    ].set_index("Model").loc[models]

    accuracy = subset["Mean CV Accuracy"] * 100

    ax.bar(
        x + (i - 1) * width,
        accuracy,
        width,
        label=config,
        color=colors[i]
    )

ax.set_title(
    "Classification Accuracy With and Without PCA"
)
ax.set_ylabel("Mean Cross-Validation Accuracy (%)")
ax.set_xlabel("Machine-Learning Model")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha="right")
ax.set_ylim(0, 105)
ax.legend()

plt.tight_layout()

output_file = results_dir / "model_accuracy_comparison.png"

plt.savefig(output_file, dpi=300)
plt.close()

print("Saved: model_accuracy_comparison.png")


# --------------------------------------------------
# 5. Chart 2: Model training time comparison
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 6))

for i, config in enumerate(configurations):

    subset = results[
        results["Configuration"] == config
    ].set_index("Model").loc[models]

    fit_times = subset["Mean Fit Time (seconds)"]

    ax.bar(
        x + (i - 1) * width,
        fit_times,
        width,
        label=config,
        color=colors[i]
    )

ax.set_title(
    "Model Fitting Time With and Without PCA"
)
ax.set_ylabel("Mean Pipeline Fit Time (seconds)")
ax.set_xlabel("Machine-Learning Model")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha="right")
ax.legend()

plt.tight_layout()

output_file = results_dir / "model_training_time_comparison.png"

plt.savefig(output_file, dpi=300)
plt.close()

print("Saved: model_training_time_comparison.png")


# --------------------------------------------------
# 6. Chart 3: Accuracy vs. feature count
# --------------------------------------------------

feature_counts = {
    "Original Features": 20254,
    "PCA 310 Components": 310,
    "PCA 434 Components": 434
}

fig, ax = plt.subplots(figsize=(11, 6))

for model in models:

    subset = results[
        results["Model"] == model
    ].copy()

    subset["Feature Count"] = (
        subset["Configuration"].map(feature_counts)
    )

    subset = subset.sort_values("Feature Count")

    ax.plot(
        subset["Feature Count"],
        subset["Mean CV Accuracy"] * 100,
        marker="o",
        linewidth=2,
        label=model
    )

ax.set_title(
    "Classification Accuracy vs. Feature Count"
)
ax.set_xlabel("Number of Features / PCA Components")
ax.set_ylabel("Mean Cross-Validation Accuracy (%)")
ax.set_xscale("log")
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()

output_file = results_dir / "accuracy_vs_feature_count.png"

plt.savefig(output_file, dpi=300)
plt.close()

print("Saved: accuracy_vs_feature_count.png")


# --------------------------------------------------
# 7. Completion message
# --------------------------------------------------

print("\nAll visualizations generated successfully.")
print("Files saved to:", results_dir)
