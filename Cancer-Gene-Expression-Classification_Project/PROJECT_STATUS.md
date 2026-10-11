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