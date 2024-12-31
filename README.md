# Credit Score Analysis Project

## Overview
This project implements a machine learning solution for analyzing credit scores using clustering techniques. It processes raw credit score data, applies feature engineering, and uses unsupervised learning methods to identify patterns in credit profiles.

## Features
- Data preprocessing and feature engineering
- Anomaly detection using Isolation Forest
- Dimensionality reduction with PCA
- Clustering analysis using K-means
- Model performance evaluation using silhouette score

## Project Structure
```
├── credit_score_dataset.csv    # Raw input data
├── preprocessed_data.csv       # Processed dataset
├── templates.py               # Main processing and model scripts
└── README.md                  # Project documentation
```

## Technical Details

### Data Preprocessing
- Log transformation of years of experience
- One-hot encoding of categorical variables
- Feature scaling using StandardScaler
- Custom economic score calculation based on multiple factors

### Model Pipeline
1. Anomaly Detection: Uses IsolationForest to remove outliers
2. Dimensionality Reduction: PCA for reducing features to 2 dimensions
3. Clustering: K-means algorithm with 5 clusters
4. Performance Evaluation: Silhouette score calculation

## Dependencies
- pandas
- numpy
- scikit-learn

## Usage
1. Place your credit score dataset in the project directory
2. Run the preprocessing step:
```python
data = preprocess("./credit_score_dataset.csv")
```
3. Train and evaluate the model:
```python
results = train_model('./preprocessed_data.csv')
```

## Features Used
- Years of Experience
- Place of Residence
- House Ownership Status
- Car Ownership
- Number of Children
- Custom Economic Score

## Model Parameters
- K-means clusters: 5
- Isolation Forest contamination: 0.3
- PCA components: 2 