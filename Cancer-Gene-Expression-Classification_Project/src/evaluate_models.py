
"""
COSC 645 Semester Project
Cancer Type Classification Using Gene Expression Data

Milestone 6: Model Evaluation and Comparison
Part 1: Stratified Cross-Validation
"""

from pathlib import Path
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC


# --------------------------------------------------
# 1. Set up project paths
# --------------------------------------------------

project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data" / "raw"
results_dir = project_dir / "results"

results_dir.mkdir(parents=True, exist_ok=True)

print("COSC 645 - Milestone 6 Model Evaluation")
print("-" * 50)


# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------

data = pd.read_csv(
    data_dir / "data.csv",
    index_col=0
)

labels = pd.read_csv(
    data_dir / "labels.csv",
    index_col=0
)

assert data.index.equals(labels.index), (
    "Sample identifiers do not match."
)

X = data
y = labels["Class"]

print(f"Total samples: {X.shape[0]}")
print(f"Original features: {X.shape[1]}")


# --------------------------------------------------
# 3. Recreate the original train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# --------------------------------------------------
# 4. Define the four classifiers
# --------------------------------------------------

classifiers = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=42
    ),

    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Gaussian Naive Bayes": GaussianNB(),

    "Linear SVM": LinearSVC(
        C=1.0,
        max_iter=10000,
        random_state=42
    )
}


# --------------------------------------------------
# 5. Configure 5-fold stratified cross-validation
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# --------------------------------------------------
# 6. Evaluate each model
# --------------------------------------------------

results = []

for model_name, classifier in classifiers.items():

    print(f"\nEvaluating {model_name}...")
    print("-" * 50)

    pipeline = Pipeline([
        (
            "variance_filter",
            VarianceThreshold(threshold=0.0)
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            classifier
        )
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=1
    )

    mean_accuracy = scores.mean()
    std_accuracy = scores.std()

    print("Fold Accuracy Scores:")

    for i, score in enumerate(scores, start=1):
        print(f"Fold {i}: {score * 100:.2f}%")

    print(f"Mean Accuracy: {mean_accuracy * 100:.2f}%")
    print(f"Standard Deviation: {std_accuracy * 100:.2f}%")

    results.append({
        "Model": model_name,
        "Mean CV Accuracy": mean_accuracy,
        "CV Standard Deviation": std_accuracy
    })


# --------------------------------------------------
# 7. Save cross-validation results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Mean CV Accuracy",
    ascending=False
)

output_file = results_dir / "cross_validation_results.csv"

results_df.to_csv(
    output_file,
    index=False
)

print("\nFinal Cross-Validation Comparison")
print("-" * 50)

print(results_df.to_string(index=False))

print("\nResults saved to:")
print(output_file)

print("\nMilestone 6 - Cross-validation complete.")
