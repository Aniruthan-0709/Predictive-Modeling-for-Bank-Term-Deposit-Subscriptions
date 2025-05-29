# Bank Term Deposit Subscription Prediction

This project uses supervised machine learning to predict whether a client will subscribe to a term deposit based on telemarketing data from a Portuguese bank. The solution improves targeting efficiency, reduces marketing costs, and enhances customer conversion through data-driven decision-making.

## Project Overview

Telemarketing campaigns in banking have low conversion rates (10–12%), leading to wasted resources. This project uses the UCI Bank Marketing dataset to identify potential subscribers and optimize outreach strategies using classification models.

## Objectives

* Predict term deposit subscriptions based on customer and campaign features
* Handle class imbalance effectively
* Evaluate and compare multiple machine learning models
* Improve marketing ROI and customer targeting

## Dataset

* Source: UCI Machine Learning Repository - Bank Marketing Dataset
* Records: 45,211
* Features: 16 input features + 1 binary target (`y`)
* Target Classes: `yes` (subscribed) and `no` (not subscribed), highly imbalanced (\~88% no)

## Workflow Summary

1. Data Cleaning and Preprocessing

   * Treated unknown values
   * Winsorized outliers
   * Encoded categorical variables
   * Standardized numerical features

2. Exploratory Data Analysis

   * Target imbalance analysis
   * Univariate and bivariate visualizations
   * Correlation analysis
   * Seasonality detection

3. Feature Engineering

   * Feature selection using correlation and chi-squared tests
   * Removed data-leakage-prone fields (e.g., duration)

4. Modeling and Evaluation

   * Models used: Logistic Regression, Decision Trees, Random Forest, KNN, XGBoost
   * SMOTE and class weighting for imbalance
   * Metrics: AUC, F1-score, Precision, Recall

## Results

| Model               | AUC  | F1-Score | Remarks                           |
| ------------------- | ---- | -------- | --------------------------------- |
| Logistic Regression | 0.86 | 0.70     | Simple and interpretable          |
| Decision Tree       | 0.83 | 0.68     | Rule-based, prone to overfitting  |
| Random Forest       | 0.89 | 0.74     | Robust with feature importance    |
| KNN                 | 0.76 | 0.63     | Weak in high-dimensional space    |
| XGBoost             | 0.91 | 0.77     | Best performance, GPU-accelerated |

## Tools and Technologies

* Python, pandas, NumPy, scikit-learn
* XGBoost, SMOTE (imbalanced-learn)
* seaborn, matplotlib
* Jupyter Notebook

## Project Structure

```
Predictive-Modeling-for-Bank-Term-Deposit-Subscriptions/
|
├── Plots/            # Visualizations and EDA plots
├── code/             # Python scripts for preprocessing, modeling, etc.
├── data/             # Dataset and any derived data files
├── documents/        # Final report, presentation, and documentation
└── README.md         # Project documentation
```

## How to Run

1. Clone the repository
   `git clone https://github.com/your-username/bank-term-deposit-prediction.git`

2. Navigate to the project directory
   `cd Predictive-Modeling-for-Bank-Term-Deposit-Subscriptions`

3. Install dependencies
   `pip install -r requirements.txt`

4. Open the main notebook or script
   `jupyter notebook` or run Python scripts in the `code/` folder

## Authors

Aniruthan Swaminathan
Suhas Ramachandra
