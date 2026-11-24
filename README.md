# Manufacturing Defect Prediction Using Machine Learning

## Project Overview

This project develops a machine learning-based approach for predicting manufacturing defects using simulated manufacturing process data.

The project combines exploratory data analysis, statistical investigation, machine learning, model optimization, and classification threshold analysis to investigate the relationship between manufacturing parameters and product defects.

**Project Domain:** Mechanical Engineering | Manufacturing Analytics | Machine Learning

**Primary Model:** Optimized XGBoost Classifier

**Dataset:** Synthetic manufacturing dataset (500 observations)

## Business Problem

Manufacturing defects can increase production costs, material waste, inspection requirements, and rework.

The objective of this project is to develop a predictive classification model that identifies potentially defective manufactured parts based on process parameters and product measurements.

The model is designed as an experimental demonstration of predictive quality analytics. It has not been validated for real industrial deployment.

## Dataset Description

The dataset contains 500 simulated manufacturing observations and 10 original columns.

### Input Features

| Feature | Description |
|---|---|
| Machine_Temperature_C | Machine operating temperature |
| Injection_Pressure_bar | Injection pressure |
| Cycle_Time_sec | Manufacturing cycle time |
| Cooling_Time_sec | Cooling duration |
| Material_Moisture_pct | Material moisture percentage |
| Operator_Experience_years | Operator experience |
| Part_Weight_g | Manufactured part weight |
| Dimensional_Deviation_mm | Dimensional deviation |
| Surface_Roughness_um | Surface roughness |

### Target Variable

- `Defect`: No or Yes

Class distribution:

- No Defect: 372 observations (74.4%)
- Defect: 128 observations (25.6%)

An additional `Deviation_Category` feature was created for exploratory analysis and stored separately in the processed dataset.

**Important:** The dataset is synthetic and does not represent measurements collected from an actual manufacturing facility.

## Exploratory Data Analysis

The exploratory analysis included:

- Manufacturing parameter distributions
- Defect class distribution
- Correlation heatmap
- Dimensional deviation distribution
- Dimensional deviation categories
- Deviation comparison by defect status

A total of 11 visualization figures were generated and saved in `reports/figures/`.

## Machine Learning Methodology

The following classification algorithms were investigated:

1. Logistic Regression
2. Balanced Logistic Regression
3. Random Forest Classifier
4. Optimized Random Forest
5. XGBoost Classifier
6. Optimized XGBoost Classifier

### Model Optimization

- Stratified train-test split (80:20)
- Feature scaling for Logistic Regression
- Class imbalance handling
- GridSearchCV hyperparameter optimization
- Five-fold cross-validation during model tuning
- Out-of-fold prediction-based classification threshold analysis
- Confusion matrix and classification metric evaluation
- Feature importance and permutation importance analysis

The selected XGBoost model uses a classification threshold of 0.35.

## Final Model Performance

**Optimized XGBoost Classifier**

| Metric | Result |
|---|---:|
| Accuracy | 48.00% |
| Precision | 30.88% |
| Recall | 80.77% |
| F1-Score | 44.68% |
| Classification Threshold | 0.35 |

### Confusion Matrix

| | Predicted: No | Predicted: Yes |
|---|---:|---:|
| Actual: No | 27 | 47 |
| Actual: Yes | 5 | 21 |

The model identified 21 out of 26 defective observations in the test set.

The selected threshold prioritizes defect detection (recall), but results in a relatively high number of false positives.

These results are experimental and should not be interpreted as evidence of industrial production performance.

## Feature Importance

Feature importance was investigated using:

- XGBoost built-in feature importance
- Permutation importance
- Threshold-aware permutation importance

These analyses provide model-specific insights into the contribution of input features. They do not establish causal relationships between manufacturing parameters and defects.

## Project Structure

```text
Manufacturing-Defect-Prediction-Using-Machine-Learning/
│
├── data/
│   ├── raw/
│   │   └── manufacturing_data_synthetic.csv
│   │
│   └── processed/
│       └── manufacturing_data_with_deviation_category.csv
│
├── notebooks/
│   └── manufacturing_defect_prediction.ipynb
│
├── reports/
│   └── figures/
│       ├── 01_manufacturing_parameter_distributions.png
│       ├── 02_manufacturing_defect_distribution.png
│       ├── 03_manufacturing_correlation_heatmap.png
│       ├── 04_dimensional_deviation_distribution.png
│       ├── 05_dimensional_deviation_categories.png
│       ├── 06_deviation_by_defect_status.png
│       ├── 07_optimized_xgboost_confusion_matrix.png
│       ├── 08_ml_model_performance_comparison.png
│       ├── 09_xgboost_threshold_analysis.png
│       ├── 10_xgboost_feature_importance.png
│       └── 11_threshold_aware_permutation_importance.png
│
├── models/
│   ├── xgboost_defect_prediction_bundle.joblib
│   └── model_metadata.json
│
├── requirements.txt
└── README.md
```

## Technologies and Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Jupyter Notebook

## Installation and Execution

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project directory

```bash
cd Manufacturing-Defect-Prediction-Using-Machine-Learning
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

Open Jupyter Notebook and execute:

```text
notebooks/manufacturing_defect_prediction.ipynb
```

Execute the notebook cells in their intended order.

## Saved Model

The trained XGBoost model and its inference configuration are stored in:

```text
models/xgboost_defect_prediction_bundle.joblib
```

The model bundle includes:

- Trained XGBoost classifier
- Classification threshold (0.35)
- Required feature names
- Positive and negative class labels

Model metadata is available in `models/model_metadata.json`.

## Limitations and Future Improvements

- The current dataset is synthetic and relatively small.
- Model performance requires validation on authorized real-world manufacturing data.
- The classification threshold should be evaluated against actual inspection and defect-related costs.
- Further improvement may involve additional process variables, larger datasets, independent validation, and more extensive model calibration.
- Future development could include an interactive machine learning prediction application.

## Disclaimer

This project is an educational and experimental machine learning project using synthetic data. Its results do not establish real-world manufacturing accuracy, reliability, or industrial readiness.

## Author

**Chaudhary Shavaiz Ahmed Warraich**

Mechanical Engineering | Machine Learning | Python | Engineering Optimization | Manufacturing Analytics

Research interests include additive manufacturing, predictive modelling, AI-based engineering optimization, and intelligent manufacturing.