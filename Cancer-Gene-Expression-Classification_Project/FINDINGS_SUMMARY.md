## Project Overview

This project evaluates the effectiveness of machine-learning
algorithms for classifying five cancer types using high-dimensional
gene-expression data.

The dataset contains 801 samples and 20,531 original features.
After removing zero-variance features, 20,254 features remained.

Four classification algorithms were evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Linear Support Vector Machine (SVM)

Model performance was evaluated using a stratified 80/20
training-test split and five-fold cross-validation on the
training data.

## 1. Best-Performing Model

Linear SVM demonstrated the strongest overall performance.

Using the original filtered features, Linear SVM achieved
100% accuracy on the held-out test set and 100% mean accuracy
across five cross-validation folds.

Logistic Regression and KNN also demonstrated strong
classification performance, with mean cross-validation
accuracies of 99.53% and 99.84%, respectively.

Gaussian Naive Bayes achieved 59.69% mean cross-validation
accuracy, substantially below the other classifiers.

Based on these results, Linear SVM was selected as the
best-performing model for this experiment.

## 2. Impact of PCA

Principal Component Analysis (PCA) was evaluated to determine
whether dimensionality reduction could preserve classification
accuracy while reducing the number of input dimensions.

The original filtered dataset contained 20,254 features.

Two PCA configurations were evaluated:

- 310 principal components
- 434 principal components

Using 310 components reduced dimensionality by approximately
98.47%.

Linear SVM achieved 99.84% mean cross-validation accuracy
with 310 components, compared with 100% using the original
features.

Logistic Regression improved slightly from 99.53% to
99.84% mean cross-validation accuracy with PCA.

KNN maintained accuracy above 99% across all configurations.

Gaussian Naive Bayes performed worse with PCA, indicating
that dimensionality reduction did not benefit every algorithm.

Overall, PCA substantially reduced dimensionality while
preserving high classification accuracy for the three
strongest models.

## 3. Computational Efficiency

PCA did not consistently reduce total pipeline fitting time.

For Logistic Regression, KNN, and Gaussian Naive Bayes,
the additional PCA computation increased average fitting time.

Linear SVM showed a different pattern.

Its mean fitting time decreased from approximately
5.79 seconds with the original features to 3.73 seconds
with 310 PCA components.

This represents a reduction of approximately 35.6%.

These measurements were collected from a single experimental
environment, so additional testing would be needed to
establish consistent computational improvements.

## 4. Study Limitations

Several limitations should be considered when interpreting
the results.

First, the dataset contained only 801 samples, which is
relatively small compared with the number of gene-expression
features.

Second, the data came from a single public dataset.
Performance on independent datasets may differ.

Third, the high classification accuracy could reflect
patterns specific to the dataset, including differences
in sample collection or processing.

Fourth, the models were evaluated using predefined
hyperparameters rather than an extensive tuning process.

Finally, the results should not be interpreted as evidence
that the models are ready for clinical diagnosis.

Independent validation and additional testing would be
required before considering real-world medical applications.

## 5. Recommended Improvements

Future work could investigate several improvements:

1. Perform hyperparameter tuning using cross-validation.
2. Evaluate additional feature-selection techniques.
3. Test model performance on independent gene-expression datasets.
4. Investigate model interpretability and influential features.
5. Compare PCA against other dimensionality-reduction methods.

These improvements could help determine whether the observed
performance remains consistent across different datasets
and experimental conditions.

## 6. Conclusion

The results demonstrate that machine-learning algorithms
can classify the five cancer types in this dataset with
high accuracy.

Linear SVM achieved the strongest overall performance,
while Logistic Regression and KNN also produced
excellent results.

PCA reduced the number of input dimensions by approximately
98.47% while preserving high classification accuracy for
the strongest classifiers.

However, PCA did not consistently improve computational
efficiency, and its effectiveness varied by algorithm.

Overall, the findings support the use of machine learning
and dimensionality reduction for analyzing high-dimensional
gene-expression data, while emphasizing the importance
of independent validation and further research.