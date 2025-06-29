# Telecom_Churn_Prediction


Telecom Churn Prediction and Customer Segmentation
This project provides a comprehensive analysis of telecommunication customer data to predict churn and segment customers based on their churn probability and other key characteristics. The goal is to identify customers at risk of churning and to understand different customer groups for targeted retention strategies.

Features
Data Loading & Initial Exploration: Loads customer data from a CSV file and performs initial checks (shape, info, missing values, descriptive statistics, churn rate).

Robust Data Cleaning & Preprocessing:

Handles missing values in numerical columns (e.g., TotalCharges) by imputing with the median.

Converts various categorical string columns to numerical representations using explicit mapping (for binary 'Yes'/'No' columns, 'gender') and label encoding (pd.factorize) for multi-category columns like InternetService, Contract, PaymentMethod, etc., to resolve ValueErrors during numerical operations.

Converts the target Churn column to numerical (0/1).

Removes the customerID column.

Exploratory Data Analysis (EDA):

Visualizes churn distribution.

Displays distributions of numerical features.

Shows churn rates across various categorical features.

Generates a correlation matrix for numerical features.

Machine Learning Model Building & Evaluation:

Sets up a robust preprocessing pipeline using StandardScaler for numerical features.

Trains and evaluates three different classification models:

Logistic Regression

Random Forest Classifier

Gradient Boosting Classifier

Evaluates models using key metrics: Accuracy, Precision, Recall, F1-Score, ROC AUC, and Confusion Matrix.

Plots ROC curves for each model.

Performs 5-fold cross-validation to assess model generalization.

Model Explainability (ELI5):

Utilizes Permutation Importance to identify the most influential features for the chosen best model (e.g., Random Forest), explaining why a customer might churn.

Customer Segmentation:

Predicts churn probability for all customers.

Segments customers into categories like 'At Risk', 'Loyal', 'New & Potentially Loyal', 'Low Value Loyal', and 'Stable' based on churn probability, tenure, and monthly charges.

Provides insights into the churn rate and average churn probability for each segment.

Characterizes each segment with descriptive statistics.

Conceptual Recommendations: Outlines actionable strategies for customer retention based on model insights and segmentation.

Requirements
To run this script, you need the following Python libraries:

pandas

numpy

matplotlib

seaborn

scikit-learn

eli5 (You might need to install this: pip install eli5)

shap (Optional, if you wish to use SHAP for explainability; install with pip install shap)

You can install all required libraries using pip:

Bash

pip install pandas numpy matplotlib seaborn scikit-learn eli5
# pip install shap # Uncomment if you want to use SHAP
Usage
Save the script: Save the provided Python code as, for example, churn_analysis.py.

Place the dataset: Ensure your telecom churn dataset is named Telecom_Churn_Prediction.csv and is placed in the same directory as the script.

Run the script: Open a terminal or command prompt, navigate to the directory where you saved the files, and run the script using:

Bash

python churn_analysis.py
The script will print various outputs to the console, including data exploration insights, cleaning steps, model performance metrics, and segmentation results. It will also generate several plots.

Dataset
The script expects a CSV file named Telecom_Churn_Prediction.csv. This file should contain customer data including features related to their services, contract, charges, and a 'Churn' column (binary target variable, e.g., 'Yes'/'No').

Output
The script will produce:

Detailed print statements in the console about data loading, cleaning, model training progress, and evaluation metrics.

Several plots displayed (and closed if running in a non-interactive environment):

Churn Distribution

Histograms of Numerical Features

Count Plots of Categorical Features vs. Churn

Correlation Matrix Heatmap

ROC Curves for each model

An eli5 output for Permutation Importance (if running in a Jupyter Notebook or compatible environment).

