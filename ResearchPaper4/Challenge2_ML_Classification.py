# Decision tree classification of breast tumors (malignant vs. benign)
# Dataset: Breast Cancer Wisconsin (Diagnostic), built into scikit-learn

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------
# Step 1: Load the dataset
# ---------------------------------------------------------------
data = load_breast_cancer()
X = data.data      # 30 numeric features describing cell nuclei (radius, texture, ...)
y = data.target    # Class labels: 0 = malignant, 1 = benign
                   # NOTE: scikit-learn encodes MALIGNANT as 0 and BENIGN as 1.

print("=" * 60)
print("BREAST CANCER WISCONSIN DATASET")
print("=" * 60)
print(f"Observations: {X.shape[0]}")
print(f"Features:     {X.shape[1]}")
print(f"Malignant (0): {(y == 0).sum()}   Benign (1): {(y == 1).sum()}")

# ---------------------------------------------------------------
# Step 2: Split into training and testing sets
# ---------------------------------------------------------------
# 80% of the data trains the model; 20% is held out to test it on unseen data.
# stratify=y keeps the malignant/benign proportions the same in both sets.
# random_state makes the split reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples: {len(X_train)}   Testing samples: {len(X_test)}")

# ---------------------------------------------------------------
# Step 3: Train the decision tree
# ---------------------------------------------------------------
# max_depth limits how deep the tree can grow. An unrestricted tree tends to
# memorize the training data (overfitting), so we cap it for better generalization.
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------------------------------
# Step 4: Make predictions on both sets
# ---------------------------------------------------------------
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# ---------------------------------------------------------------
# Step 5: Evaluate the model
# ---------------------------------------------------------------
# We treat MALIGNANT (label 0) as the "positive" class, because missing a
# malignant tumor is the costly mistake in this problem.
# pos_label=0 tells scikit-learn to compute precision/recall for that class.
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, pos_label=0)
recall = recall_score(y_test, y_test_pred, pos_label=0)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE (malignant = positive class)")
print("=" * 60)
print(f"Training accuracy: {accuracy_score(y_train, y_train_pred):.3f}")
print(f"Test accuracy:     {accuracy:.3f}")
print(f"Test precision:    {precision:.3f}  (of tumors predicted malignant, how many truly were)")
print(f"Test recall:       {recall:.3f}  (of truly malignant tumors, how many we caught)")

# Confusion matrix: rows = actual class, columns = predicted class.
# labels=[0, 1] orders it as malignant first, then benign.
cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
tp, fn = cm[0]   # actual malignant: predicted malignant / predicted benign
fp, tn = cm[1]   # actual benign:    predicted malignant / predicted benign

print("\nConfusion matrix:")
print(f"{'':>18}{'Pred malignant':>16}{'Pred benign':>14}")
print(f"{'Actual malignant':>18}{tp:>16}{fn:>14}")
print(f"{'Actual benign':>18}{fp:>16}{tn:>14}")
print(f"\nTrue positives: {tp}  False negatives: {fn}  "
      f"False positives: {fp}  True negatives: {tn}")

# ---------------------------------------------------------------
# Step 6: Look at which features mattered most
# ---------------------------------------------------------------
importances = sorted(
    zip(data.feature_names, model.feature_importances_),
    key=lambda pair: pair[1],
    reverse=True,
)
print("\nTop 5 most important features:")
for name, score in importances[:5]:
    print(f"  {name:<25}{score:.3f}")

# ---------------------------------------------------------------
# Step 7: Visualize the confusion matrix and the tree
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Left: confusion matrix heatmap
ConfusionMatrixDisplay(cm, display_labels=["Malignant", "Benign"]).plot(
    ax=axes[0], cmap="Blues", colorbar=False
)
axes[0].set_title("Confusion Matrix (Test Set)")

# Right: the learned decision tree
plot_tree(
    model,
    feature_names=data.feature_names,
    class_names=["Malignant", "Benign"],  # order matches label values 0, 1
    filled=True,
    rounded=True,
    fontsize=7,
    ax=axes[1],
)
axes[1].set_title("Decision Tree (max_depth=4)")

plt.tight_layout()
plt.show()