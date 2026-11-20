
"""
COSC 645 Semester Project
Cancer Type Classification Using Gene Expression Data
"""

from pathlib import Path
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC


# --------------------------------------------------
# 1. Set up paths
# --------------------------------------------------

project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data" / "raw"
results_dir = project_dir / "results"

results_dir.mkdir(parents=True, exist_ok=True)

print("COSC 645 - PCA Model Comparison")
print("-" * 50)


# --------------------------------------------------
# 2. Load dataset
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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples reserved: {len(X_test)}")


# --------------------------------------------------
# 3. Define models
# --------------------------------------------------

classifiers = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=42
    ),
    "KNN": KNeighborsClassifier(
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
# 4. Define feature configurations
# --------------------------------------------------

configurations = {
    "Original Features": None,
    "PCA 310 Components": 310,
    "PCA 434 Components": 434
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# --------------------------------------------------
# 5. Compare models
# --------------------------------------------------

results = []

for config_name, n_components in configurations.items():

    for model_name, classifier in classifiers.items():

        print(f"\nEvaluating: {model_name}")
        print(f"Configuration: {config_name}")

        steps = [
            (
                "variance_filter",
                VarianceThreshold(threshold=0.0)
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]

        if n_components is not None:
            steps.append((
                "pca",
                PCA(
                    n_components=n_components,
                    svd_solver="randomized",
                    random_state=42
                )
            ))

        steps.append(("classifier", classifier))

        pipeline = Pipeline(steps)

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
            return_train_score=False
        )

        mean_accuracy = scores["test_score"].mean()
        std_accuracy = scores["test_score"].std()
        mean_fit_time = scores["fit_time"].mean()
        mean_score_time = scores["score_time"].mean()

        print(
            f"Mean Accuracy: {mean_accuracy * 100:.2f}%"
        )
        print(
            f"Standard Deviation: {std_accuracy * 100:.2f}%"
        )
        print(
            f"Mean Fit Time: {mean_fit_time:.2f} seconds"
        )

        results.append({
            "Configuration": config_name,
            "Model": model_name,
            "Mean CV Accuracy": mean_accuracy,
            "CV Standard Deviation": std_accuracy,
            "Mean Fit Time (seconds)": mean_fit_time,
            "Mean Score Time (seconds)": mean_score_time
        })


# --------------------------------------------------
# 6. Save comparison results
# --------------------------------------------------

results_df = pd.DataFrame(results)

output_file = (
    results_dir / "pca_model_comparison.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print("\nFinal PCA Model Comparison")
print("-" * 50)

print(results_df.to_string(index=False))

print("\nResults saved to:")
print(output_file)

