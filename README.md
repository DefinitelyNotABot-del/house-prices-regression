# House Price Regression

A comprehensive machine learning project for predicting house prices using the Ames Housing dataset. This project implements multiple regression models including OLS, Robust Linear Models, Regularized models (Lasso, Ridge, ElasticNet), and tree-based ensemble methods (RandomForest, XGBoost, LightGBM).

## Project Overview

This project analyzes house prices using 79 features from the Ames Housing dataset and implements various regression techniques to predict sale prices with high accuracy.

### Key Features

- **Comprehensive EDA**: Visualizations including correlation heatmaps, scatter plots, box plots, and distribution analysis
- **Multiple Model Types**: 
  - Statistical models (OLS, RLM, GLM)
  - Regularized models (Lasso, Ridge, ElasticNet)
  - Tree-based ensembles (RandomForest, XGBoost, LightGBM)
- **Feature Engineering**: Log transformation of target variable, dummy encoding, feature scaling
- **Model Evaluation**: R², RMSE, MAE metrics on both log and original price scales
- **Visualization Suite**: 15+ saved figures for model diagnostics and performance comparison

## Project Structure

```
├── train.csv                  # Training dataset
├── test.csv                   # Test dataset
├── data_description.txt       # Feature descriptions
├── sample_submission.csv      # Submission format
├── train_save_models.py       # Model training pipeline
├── predict_test_set.py        # Prediction and submission generation
├── graph03.py                 # Comprehensive EDA and visualization
├── delete_temp_files.py       # Utility script
├── saved_models/              # Trained model artifacts (gitignored)
├── submissions/               # Model predictions
└── figures/                   # Generated visualizations
```

## Installation

### Prerequisites

```bash
Python 3.8+
```

### Dependencies

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn xgboost lightgbm joblib
```

## Usage

### 1. Train Models

Train all models and save artifacts:

```bash
python train_save_models.py
```

This will:
- Clean data and remove outliers
- Train 9 different models
- Save models, scalers, and feature names to `saved_models/`
- Print model parameters and feature importances

### 2. Generate Predictions

Generate predictions on test set:

```bash
python predict_test_set.py
```

This will:
- Load trained models
- Generate predictions for test data
- Save submission CSVs to `submissions/`

### 3. Run Analysis & Visualization

Generate comprehensive analysis and visualizations:

```bash
python graph03.py
```

This will:
- Perform exploratory data analysis
- Train models and compute metrics
- Generate 15+ figures saved to `figures/`

## Model Performance

Performance on training data (after outlier removal):

| Model | R² (log) | RMSE (log) | RMSE ($) |
|-------|----------|------------|----------|
| XGBoost | 0.998 | 0.018 | $3,299 |
| LightGBM | 0.997 | 0.023 | $3,632 |
| RandomForest | 0.984 | 0.051 | $9,937 |
| OLS | 0.948 | 0.090 | $16,126 |
| GLM | 0.948 | 0.090 | $16,126 |
| RLM | 0.945 | 0.093 | $16,211 |
| Ridge | 0.942 | 0.095 | $16,776 |
| ElasticNet | 0.933 | 0.102 | $18,311 |
| Lasso | 0.933 | 0.103 | $18,355 |

## Key Findings

1. **Top Correlated Features**: OverallQual (0.791), GrLivArea (0.709), GarageCars (0.640)
2. **Feature Selection**: Lasso selected 111/259 features, ElasticNet selected 123/259
3. **Best Performers**: Tree-based models (XGBoost, LightGBM) significantly outperform linear models
4. **Outlier Impact**: Removing 4 GrLivArea outliers improved model performance

## Visualizations

The `figures/` directory contains:
- EDA: Distribution plots, correlation heatmaps, scatter plots
- Model Comparison: R² scores, RMSE comparison
- Diagnostics: Actual vs predicted, residual distributions
- Feature Importance: Top features from Lasso and tree models
- Submission Analysis: Distribution comparisons across models

## Technical Details

- **Target Transform**: Log1p transformation for normalized distribution
- **Feature Engineering**: 259 features after one-hot encoding
- **Scaling**: StandardScaler for regularized models
- **Cross-Validation**: 5-fold CV for Lasso, Ridge, ElasticNet
- **Outlier Removal**: GrLivArea > 4000 sq ft (4 observations)

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

This project is open source and available for educational purposes.

## Acknowledgments

- Ames Housing Dataset from Kaggle
- scikit-learn, statsmodels, XGBoost, and LightGBM communities
