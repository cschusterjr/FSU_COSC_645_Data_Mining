"""
COSC 645 Semester Project
Cancer Type Classification Using Gene Expression Data

Milestone 5: Machine-Learning Model Development
Model 1: Logistic Regression
"""

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data" / "raw"

print("COSC 645 - Logistic Regression Model")
print("-" * 50)


# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------

data = pd.read_csv(data_dir / "data.csv", index_col=0)
labels = pd.read_csv(data_dir / "labels.csv", index_col=0)

# Confirm that the samples match
assert data.index.equals(labels.index), (
    "Sample identifiers do not match."
)

X = data
y = labels["Class"]

print(f"Total samples: {X.shape[0]}")
print(f"Original features: {X.shape[1]}")
print(f"Cancer classes: {sorted(y.unique())}")


# --------------------------------------------------
# 3. Split training and testing data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# --------------------------------------------------
# 4. Create the machine-learning pipeline
# --------------------------------------------------

model = Pipeline([
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
        LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=42
        )
    )
])

# --------------------------------------------------
# 5. Train the model
# --------------------------------------------------

print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)

print("Training complete.")

filtered_features = (
    model.named_steps["variance_filter"]
    .get_support()
    .sum()
)

print(f"Features after filtering: {filtered_features}")


# --------------------------------------------------
# 6. Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 7. Evaluate initial performance
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nLogistic Regression Results")
print("-" * 50)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Accuracy (%): {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# --------------------------------------------------
# 8. Train K-Nearest Neighbors (KNN)
# --------------------------------------------------

print("\nTraining K-Nearest Neighbors...")

knn_model = Pipeline([
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
        KNeighborsClassifier(n_neighbors=5)
    )
])

knn_model.fit(X_train, y_train)

# --------------------------------------------------
# 9. Evaluate KNN
# --------------------------------------------------

knn_pred = knn_model.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_pred)

print("\nKNN Results")
print("-" * 50)

print(f"Test Accuracy: {knn_accuracy:.4f}")
print(f"Test Accuracy (%): {knn_accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        knn_pred,
        zero_division=0
    )
)


# --------------------------------------------------
# 10. Compare initial models
# --------------------------------------------------

print("\nModel Comparison")
print("-" * 50)

print(f"Logistic Regression: {accuracy * 100:.2f}%")
print(f"KNN:                 {knn_accuracy * 100:.2f}%")


# --------------------------------------------------
# 11. Train Gaussian Naive Bayes
# --------------------------------------------------

print("\nTraining Gaussian Naive Bayes...")

nb_model = Pipeline([
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
        GaussianNB()
    )
])

nb_model.fit(X_train, y_train)


# --------------------------------------------------
# 12. Evaluate Gaussian Naive Bayes
# --------------------------------------------------

nb_pred = nb_model.predict(X_test)

nb_accuracy = accuracy_score(y_test, nb_pred)

print("\nGaussian Naive Bayes Results")
print("-" * 50)

print(f"Test Accuracy: {nb_accuracy:.4f}")
print(f"Test Accuracy (%): {nb_accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        nb_pred,
        zero_division=0
    )
)


# --------------------------------------------------
# 13. Compare all three models
# --------------------------------------------------

print("\nUpdated Model Comparison")
print("-" * 50)

print(f"Logistic Regression: {accuracy * 100:.2f}%")
print(f"KNN:                 {knn_accuracy * 100:.2f}%")
print(f"Gaussian Naive Bayes:{nb_accuracy * 100:7.2f}%")



# --------------------------------------------------
# 14. Train Support Vector Machine (SVM)
# --------------------------------------------------

print("\nTraining Support Vector Machine...")

svm_model = Pipeline([
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
        LinearSVC(
            C=1.0,
            max_iter=10000,
            random_state=42
        )
    )
])

svm_model.fit(X_train, y_train)

print("SVM training complete.")


# --------------------------------------------------
# 15. Evaluate SVM
# --------------------------------------------------

svm_pred = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_pred)

print("\nSVM Results")
print("-" * 50)

print(f"Test Accuracy: {svm_accuracy:.4f}")
print(f"Test Accuracy (%): {svm_accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        svm_pred,
        zero_division=0
    )
)


# --------------------------------------------------
# 16. Compare all four models
# --------------------------------------------------

print("\nFinal Baseline Model Comparison")
print("-" * 50)

print(f"Logistic Regression: {accuracy * 100:.2f}%")
print(f"KNN:                 {knn_accuracy * 100:.2f}%")
print(f"Gaussian Naive Bayes:{nb_accuracy * 100:7.2f}%")
print(f"Linear SVM:          {svm_accuracy * 100:.2f}%")


# --------------------------------------------------
# 17. Save baseline model comparison results
# --------------------------------------------------

results_dir = project_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

model_results = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Accuracy": accuracy
    },
    {
        "Model": "K-Nearest Neighbors",
        "Accuracy": knn_accuracy
    },
    {
        "Model": "Gaussian Naive Bayes",
        "Accuracy": nb_accuracy
    },
    {
        "Model": "Linear SVM",
        "Accuracy": svm_accuracy
    }
])

model_results = model_results.sort_values(
    by="Accuracy",
    ascending=False
)

output_file = results_dir / "baseline_model_comparison.csv"

model_results.to_csv(output_file, index=False)

print("\nBaseline model results saved:")
print(output_file)

print("\nFinal Model Comparison:")
print(model_results.to_string(index=False))
