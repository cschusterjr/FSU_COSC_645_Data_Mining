# COSC 645 Semester Project Status

## Project
Cancer Gene Expression Classification

## Current Milestone
Milestone 1: Environment and GitHub Setup

## Overall Status
In Progress

## Milestone Tracking

| Milestone | Description | Status |
|-----------|-------------|--------|
| 1 | Environment and GitHub Setup | In Progress |
| 2 | Dataset Selection and Acquisition | Not Started |
| 3 | Data Cleaning and Exploration | Not Started |
| 4 | Dimensionality Reduction | Not Started |
| 5 | Machine Learning Development | Not Started |
| 6 | Model Evaluation and Comparison | Not Started |
| 7 | Interpretation and Improvements | Not Started |
| 8 | Final Report and GitHub Release | Not Started |

## Completed Work
- Created the project folder in VS Code.
- Configured Python 3.11.5 virtual environment.
- Initialized Git.
- Created project folders and initial documentation.
- Installed and verified Python dependencies.

## Next Steps
- Publish the project to GitHub.
- Select and download a cancer gene-expression dataset.
- Begin exploratory data analysis.

## Current Issues
None.

## Milestone 2: Dataset Selection and Acquisition

**Status:** In Progress

### Dataset Selection

Selected the Gene Expression Cancer RNA-Seq dataset from the UCI Machine Learning Repository. The dataset contains gene-expression measurements derived from The Cancer Genome Atlas (TCGA) and includes five cancer classifications.

Dataset source: https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq

### Dataset Validation

Created and executed `src/inspect_dataset.py` to load and inspect the dataset using Python and pandas.

The initial validation confirmed:

- 801 samples
- 20,531 gene-expression features
- Five cancer classifications: BRCA, KIRC, LUAD, PRAD, and COAD
- No missing values
- Matching sample identifiers between the gene-expression data and classification labels

### Initial Observations

The dataset contains an uneven distribution of cancer types. BRCA has the largest number of samples (300), while COAD has the fewest (78). This class imbalance will be considered when developing and evaluating classification models.

The dataset is also highly dimensional, containing substantially more gene features than samples. This makes it appropriate for evaluating dimensionality reduction techniques such as Principal Component Analysis (PCA).

### Next Steps

- Complete additional data quality checks.
- Document the dataset source and characteristics.
- Begin exploratory data analysis and visualization.

### Additional Data Quality Findings

Additional validation found no duplicate sample identifiers, duplicate gene-expression profiles, or infinite values. Gene-expression measurements ranged from 0.0000 to 20.7788.

Of the 20,531 gene-expression features, 267 were identified as constant across all 801 samples. These features provide no variation for classification and will be removed during preprocessing. After removing them, 20,264 features would remain.

The original dataset will be preserved, and feature removal will be incorporated into the machine-learning preprocessing pipeline to avoid data leakage.

**Milestone 2 Status: Complete**


Milestone 3: Data Cleaning and Exploratory Data Analysis

Status: Complete

Exploratory Data Analysis

Exploratory data analysis was performed using Python, pandas, matplotlib, and seaborn. Three visualizations were created to examine the dataset:

Cancer Type Distribution: Confirmed an uneven distribution across five cancer classifications, with BRCA having the largest number of samples (300) and COAD having the fewest (78).

Gene Expression Distribution: Examined the distribution of gene-expression measurements. The mean expression was 6.4433, the median was 7.6096, and the standard deviation was 4.0582. The histogram showed a concentration of values near zero and another concentration around 9 to 10.

Gene Variability: Identified the ten most variable gene features. The highest variance was observed for gene_9176 (44.76). The analysis also confirmed 267 constant features across the complete dataset.

Data Preprocessing

Created src/preprocess_data.py to establish a reproducible preprocessing approach.

The dataset was divided into 640 training samples (approximately 80%) and 161 testing samples (approximately 20%) using stratified sampling.

Feature filtering was performed using VarianceThreshold, which removed 277 constant features based on the training dataset. This reduced the number of features from 20,531 to 20,254.

The remaining features were standardized using StandardScaler. Both feature filtering and standardization were fitted using training data only to prevent data leakage.

Key Findings

The dataset is highly dimensional, with substantially more gene features than samples. Additionally, the cancer classifications are imbalanced, and gene-expression measurements show considerable variation.

These findings support the use of dimensionality reduction and multiple classification algorithms to investigate model performance.

Deliverables

src/exploratory_analysis.py

src/preprocess_data.py

results/cancer_type_distribution.png

results/gene_expression_distribution.png

results/top_variable_genes.png

Milestone 3 Status: Complete


## Milestone 4: Dimensionality Reduction Using PCA

**Status: Complete**

### PCA Implementation

Principal Component Analysis (PCA) was implemented using scikit-learn to investigate dimensionality reduction in the cancer gene-expression dataset. PCA was fitted using only the training data after constant-feature removal and standardization to prevent data leakage.

The training dataset contained 640 samples and 20,254 features after preprocessing.

### Explained Variance Analysis

The PCA analysis identified the number of principal components required to preserve different percentages of the total variance:

- **90% explained variance:** 310 principal components
- **95% explained variance:** 434 principal components
- **99% explained variance:** 583 principal components

These results demonstrate that PCA can substantially reduce the dimensionality of the dataset while retaining most of its variation. For example, using 310 components represents approximately a 98.5% reduction in the number of features compared with the filtered dataset.

### PCA Visualization

A two-dimensional scatter plot was created using the first two principal components, which explained approximately 19.4% of the total variance.

The visualization showed some separation between cancer types. KIRC samples formed a relatively distinct cluster, while COAD samples were concentrated toward the left side of the plot. BRCA, LUAD, and PRAD showed greater overlap, although their distributions differed.

These findings suggest that gene-expression measurements contain patterns associated with cancer type, even when represented using only two principal components.

However, PCA is an unsupervised technique, and the observed separation does not establish classification accuracy. Further evaluation using supervised machine-learning algorithms is required.

### Deliverables

- `src/pca_analysis.py`
- `results/pca_explained_variance.png`
- `results/pca_cancer_scatter.png`

### Next Steps

The next milestone will focus on developing machine-learning classification models, including Logistic Regression, K-Nearest Neighbors, Naive Bayes, and Support Vector Machines.

Models will be evaluated using the original filtered gene-expression features and PCA-transformed features to investigate the effect of dimensionality reduction on classification performance.

**Milestone 4 Status: Complete**

Milestone 5: Machine-Learning Model Development

Overview

This milestone focused on developing and evaluating four machine-learning algorithms to classify cancer types using gene-expression data. The objective was to establish baseline classification performance before investigating the effects of dimensionality reduction through Principal Component Analysis (PCA).

The models were developed using Python and scikit-learn. Each model used the same stratified training and testing split to allow for consistent comparisons.

Models Implemented

Four classification algorithms were developed:

Logistic Regression: A linear classification model used to establish baseline performance.

K-Nearest Neighbors (KNN): A distance-based classifier that predicts cancer types using the five nearest training samples.

Gaussian Naive Bayes: A probabilistic classifier that assumes features are conditionally independent within each cancer class.

Linear Support Vector Machine (SVM): A classifier that identifies decision boundaries between cancer types.

Data Preparation

The original dataset contained 801 samples and 20,531 gene-expression features representing five cancer types: BRCA, COAD, KIRC, LUAD, and PRAD.

The dataset was divided into 640 training samples and 161 testing samples using a stratified 80/20 split.

Feature filtering removed 277 constant features, leaving 20,254 features. Standardization was applied to the remaining features.

Preprocessing was implemented through scikit-learn pipelines to ensure that transformations were fitted using training data only, preventing data leakage.

Initial Classification Results

The following results were obtained using the held-out test dataset without PCA.

Model

Test Accuracy

Correct Predictions

Linear SVM

100.00%

161/161

Logistic Regression

98.14%

158/161

K-Nearest Neighbors

97.52%

157/161

Gaussian Naive Bayes

57.14%

92/161

Linear SVM achieved the highest observed accuracy, correctly classifying all 161 test samples. Logistic Regression and KNN also demonstrated strong performance, with accuracy exceeding 97%.

Gaussian Naive Bayes performed considerably worse, suggesting that its statistical assumptions may not be well suited to the relationships present in this gene-expression dataset.

These results represent performance on a single held-out test split and should not be interpreted as evidence of equivalent performance on independent clinical datasets.

Key Findings

The gene-expression dataset contains patterns that can be used to distinguish the five cancer types with high accuracy.

Model selection has a substantial impact on classification performance.

Linear SVM achieved perfect classification on the current test set, although additional validation is needed to assess generalization.

Gaussian Naive Bayes struggled with certain cancer types, particularly colon cancer, where recall was approximately 6%.

The baseline results provide a benchmark for evaluating the impact of PCA in the next milestone.