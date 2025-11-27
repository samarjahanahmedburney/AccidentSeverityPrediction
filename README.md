💥 AccidentSeverityPrediction

Author: Samar jahan burney

Predict the severity of road traffic accidents using machine learning with a complete end-to-end pipeline.

📄 Description

This project uses a dataset collected from Addis Ababa Sub-city police departments (2017–2020) for research purposes. The dataset contains 12,316 accident records with 32 features and a multi-class target variable Accident_severity (Fatal, Serious, Slight). Sensitive information has been excluded, and all data is encoded for ML tasks.

The goal is to build a robust ML pipeline that handles preprocessing, resampling, model training, evaluation, and saving the best model.

⚡ Features of the Pipeline

EDA & Visualizations:

Missing value heatmap

Target distribution chart

Numeric feature histograms and correlations

Boxplots & categorical comparisons vs target

Preprocessing:

Handle missing values (median for numeric, mode for categorical)

One-hot encoding for categorical features

Standard scaling for numeric features

Resampling:

SMOTE + Random Oversampling to handle class imbalance

Models Included:

Logistic Regression (baseline)

Decision Tree

Random Forest

XGBoost (if available)

LightGBM (if available)

Evaluation Metrics:

Weighted F1, Macro F1

Precision, Recall, Classification Report

Confusion Matrix

ROC-AUC (multi-class)

Feature Importance for tree-based models

Model comparison chart

Output:

Saves best-performing model to accident_severity_pipeline.pkl

📈 Model Performance Summary
Model	Weighted F1	Macro F1
Logistic Regression	0.5918	0.3136
Decision Tree	0.7483	0.3981
Random Forest	0.7761	0.3075
XGBoost	0.7971	0.3567
LightGBM	0.7845	0.3284

Observation: XGBoost achieves the highest weighted F1, while Decision Tree shows better macro F1. Imbalanced classes (especially Fatal injury) remain challenging.
