import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math
import os
from glob import glob

FIGURES_DIR = 'figures'


def save_figure(filename: str) -> None:
    """Persist the current Matplotlib figure under figures/ for later review."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight')
    print(f"Saved figure -> {filepath}")

# --- 0. Setup ---
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
# Set global plot style
sns.set_style("whitegrid")
pd.set_option('display.max_rows', 10)


# =============================================================================
# --- 1. Loading Data & Outlier Removal ---
# =============================================================================
print("--- 1. Loading Full Data (1460 rows) ---")
try:
    df_orig = pd.read_csv('train.csv')
    target_col = 'SalePrice'
    print(f"Original dataset shape: {df_orig.shape}")
except FileNotFoundError:
    print("ERROR: 'train.csv' not found. Please ensure it is in the same directory.")
    exit()

# --- 1.5 Explicit Outlier Removal ---
# Based on common practice and visual inspection for Ames dataset:
initial_rows = df_orig.shape[0]
df = df_orig[df_orig['GrLivArea'] < 4000].copy() # Use .copy() to avoid SettingWithCopyWarning
rows_removed = initial_rows - df.shape[0]
print(f"Removed {rows_removed} outliers based on GrLivArea > 4000.")
print(f"New dataset shape: {df.shape}")

# Define the clean log-transformed target for plotting
Y_clean_log = np.log1p(df[target_col])


# =============================================================================
# --- 2. Comprehensive EDA (on Cleaned Data) ---
# =============================================================================
print("\n--- 2. Performing Comprehensive EDA & Visualization (on Cleaned Data) ---")

# --- 2.1. Histogram: Target Variable (SalePrice) Distribution ---
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
sns.histplot(df[target_col], kde=True, bins=50)
plt.title('Distribution of SalePrice (Outliers Removed)')
plt.xlabel('Sale Price ($)')

plt.subplot(1, 2, 2)
sns.histplot(Y_clean_log, kde=True, bins=50, color='green')
plt.title('Distribution of log(SalePrice) (Normalized)')
plt.xlabel('log(Sale Price)')
plt.tight_layout()
save_figure('eda_saleprice_distributions.png')
plt.show()

# --- 2.2. Heatmap: Top Correlated Features (on Cleaned Data) ---
print("Generating Correlation Heatmap...")
corr_matrix = df.corr(numeric_only=True)
top_corr_features = corr_matrix.nlargest(15, 'SalePrice')['SalePrice'].index
top_corr_matrix = df[top_corr_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(top_corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Top Features (Cleaned Data)')
save_figure('correlation_heatmap.png')
plt.show()

# --- 2.3. Scatter Plots: Key Numeric Features vs. log(SalePrice) ---
print("Generating Scatter Plots for key predictors...")
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
sns.regplot(x='GrLivArea', y=Y_clean_log, data=df, 
            scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('GrLivArea vs. log(SalePrice)')
plt.xlabel('Above Ground Living Area (sq ft)')
plt.ylabel('log(Sale Price)')

plt.subplot(2, 2, 2)
sns.regplot(x='TotalBsmtSF', y=Y_clean_log, data=df,
            scatter_kws={'alpha':0.3}, line_kws={'color':'orange'})
plt.title('TotalBsmtSF vs. log(SalePrice)')
plt.xlabel('Total Basement Area (sq ft)')
plt.ylabel('log(Sale Price)')

plt.subplot(2, 2, 3)
sns.regplot(x='GarageArea', y=Y_clean_log, data=df,
            scatter_kws={'alpha':0.3}, line_kws={'color':'purple'})
plt.title('GarageArea vs. log(SalePrice)')
plt.xlabel('Garage Area (sq ft)')
plt.ylabel('log(Sale Price)')

plt.subplot(2, 2, 4)
sns.scatterplot(x='YearBuilt', y=Y_clean_log, data=df, alpha=0.3)
plt.title('YearBuilt vs. log(SalePrice)')
plt.xlabel('Year Built')
plt.ylabel('log(Sale Price)')

plt.tight_layout()
save_figure('scatter_key_predictors.png')
plt.show()

# --- 2.4. Box Plots: Key Categorical Features vs. log(SalePrice) ---
print("Generating Box Plots for key categorical features...")
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x='OverallQual', y=Y_clean_log, data=df)
plt.title('Impact of Overall Quality on log(SalePrice)')
plt.xlabel('Overall Quality (1=Poor, 10=Excellent)')
plt.ylabel('log(Sale Price)')

top_10_neighborhoods = df['Neighborhood'].value_counts().nlargest(10).index
df_top_10_hoods = df[df['Neighborhood'].isin(top_10_neighborhoods)]
plt.subplot(1, 2, 2)
sns.boxplot(x='Neighborhood', y=Y_clean_log, data=df_top_10_hoods)
plt.title('log(SalePrice) by Neighborhood (Top 10)')
plt.xticks(rotation=45)
plt.ylabel('log(Sale Price)')

plt.tight_layout()
save_figure('categorical_boxplots.png')
plt.show()

# --- 2.5. Line Plot & Bar Plot ---
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
year_trend = df.groupby('YrSold')[target_col].median().reset_index()
sns.lineplot(x='YrSold', y=target_col, data=year_trend, marker='o')
plt.title('Median SalePrice by Year Sold')
plt.xlabel('Year Sold')
plt.ylabel('Median Sale Price ($)')
plt.xticks(year_trend['YrSold'])

plt.subplot(1, 2, 2)
sns.countplot(y='HouseStyle', data=df, order=df['HouseStyle'].value_counts().index)
plt.title('Count of Houses by House Style')
plt.xlabel('Number of Houses')
plt.ylabel('House Style')

plt.tight_layout()
save_figure('year_trend_house_style.png')
plt.show()


# =============================================================================
# --- 3. Data Preprocessing (Full 79 Features) ---
# =============================================================================
print("\n--- 3. Data Preprocessing (Full 79 Features) ---")
# Y (target) is already defined from the cleaned 'df' for modeling
Y = np.log1p(df[target_col])

# Drop target and Id from X (using the cleaned 'df')
X = df.drop([target_col, 'Id'], axis=1)

print(f"Original feature count: {X.shape[1]}")
numeric_cols = X.select_dtypes(include=np.number).columns
object_cols = X.select_dtypes(include='object').columns

print(f"Found {len(numeric_cols)} numeric and {len(object_cols)} object columns.")

# Fill NAs
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
X[object_cols] = X[object_cols].fillna('Missing')

# Create Dummies
X = pd.get_dummies(X, columns=object_cols, drop_first=True)
print(f"New feature count after get_dummies: {X.shape[1]}")

# Save feature names for later
feature_names = X.columns.tolist()


# =============================================================================
# --- 4. Create Datasets for Models ---
# =============================================================================
print("\n--- 4. Creating Datasets for Models ---")
# 4.1 For OLS, RLM, GLM (statsmodels)
X_with_const = sm.add_constant(X.astype(float), prepend=False)
X_for_trees = X.astype(float)

# 4.2 For LassoCV, RidgeCV, ElasticNetCV (sklearn)
print("Scaling data for Regularized Models...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_names)


# =============================================================================
# --- 5. Model Building (On Outlier-Cleaned Dataset) ---
# =============================================================================
print("\n--- 5. Model Building (On Outlier-Cleaned Dataset) ---")

model_log_predictions = {}

# --- OLS Model ---
print("\n--- OLS Model Summary (Baseline on Cleaned Data) ---")
model_ols = sm.OLS(Y, X_with_const).fit()
r2_ols = model_ols.rsquared
print(model_ols.summary())
model_log_predictions['OLS'] = model_ols.predict(X_with_const)

# --- RLM Model ---
print("\n--- RLM Model (Robust to any remaining Outliers) ---")
model_rlm = sm.RLM(Y, X_with_const).fit()
ss_tot_log = np.sum((Y - Y.mean())**2)
ss_res_rlm_log = np.sum(model_rlm.resid**2)
r2_rlm = 1 - (ss_res_rlm_log / ss_tot_log)
print("RLM model fitted.")
model_log_predictions['RLM'] = model_rlm.predict(X_with_const)

# --- GLM Model ---
print("\n--- GLM Model Summary (Similar to OLS on Cleaned Data) ---")
model_glm = sm.GLM(Y, X_with_const, family=sm.families.Gaussian()).fit()
ss_res_glm_log = np.sum((Y - model_glm.predict(X_with_const))**2)
r2_glm = 1 - (ss_res_glm_log / ss_tot_log)
print(model_glm.summary())
model_log_predictions['GLM'] = model_glm.predict(X_with_const)

# --- LassoCV Model ---
print("\n--- LassoCV Model (Robust + Feature Selection - L1) ---")
model_lasso = LassoCV(cv=5, random_state=0, max_iter=10000).fit(X_scaled, Y)
r2_lasso = model_lasso.score(X_scaled, Y)
n_features_lasso = np.sum(model_lasso.coef_ != 0)
print("LassoCV model fitted.")
model_log_predictions['Lasso'] = model_lasso.predict(X_scaled)

# --- RidgeCV Model ---
print("\n--- RidgeCV Model (Robust + Handles Multicollinearity - L2) ---")
alphas_ridge = np.logspace(-6, 6, 13)
model_ridge = RidgeCV(alphas=alphas_ridge, store_cv_results=True).fit(X_scaled, Y)
r2_ridge = model_ridge.score(X_scaled, Y)
n_features_ridge = X_scaled.shape[1]
print("RidgeCV model fitted.")
model_log_predictions['Ridge'] = model_ridge.predict(X_scaled)

# --- ElasticNetCV Model ---
print("\n--- ElasticNetCV Model (Robust + Feature Selection/Multicollinearity - L1 & L2) ---")
model_elastic = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, random_state=0, max_iter=10000).fit(X_scaled, Y)
r2_elastic = model_elastic.score(X_scaled, Y)
n_features_elastic = np.sum(model_elastic.coef_ != 0)
best_l1_ratio = model_elastic.l1_ratio_
print("ElasticNetCV model fitted.")
model_log_predictions['ElasticNet'] = model_elastic.predict(X_scaled)

# --- Tree-based Models ---
print("\n--- Tree-based Models (RandomForest, XGBoost, LightGBM) ---")
tree_feature_importances = {}

try:
    from sklearn.ensemble import RandomForestRegressor
    print("Training RandomForestRegressor...")
    rf_model = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    rf_model.fit(X_for_trees, Y)
    model_log_predictions['RandomForest'] = rf_model.predict(X_for_trees)
    tree_feature_importances['RandomForest'] = pd.Series(rf_model.feature_importances_, index=feature_names)
except Exception as e:
    print(f"Skipping RandomForestRegressor: {e}")

try:
    import xgboost as xgb
    print("Training XGBoostRegressor...")
    xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=0, verbosity=0)
    xgb_model.fit(X_for_trees, Y)
    model_log_predictions['XGBoost'] = xgb_model.predict(X_for_trees)
    tree_feature_importances['XGBoost'] = pd.Series(xgb_model.feature_importances_, index=feature_names)
except Exception as e:
    print(f"Skipping XGBoostRegressor: {e}")

try:
    import lightgbm as lgb
    print("Training LGBMRegressor...")
    lgbm_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=0)
    lgbm_model.fit(X_for_trees, Y)
    model_log_predictions['LightGBM'] = lgbm_model.predict(X_for_trees)
    tree_feature_importances['LightGBM'] = pd.Series(lgbm_model.feature_importances_, index=feature_names)
except Exception as e:
    print(f"Skipping LGBMRegressor: {e}")

print("\n--- 5.5 Model Performance Summary (Training Data, log(SalePrice)) ---")
metrics_rows = []
model_predictions_exp = {}
residuals_exp = {}
actual_log = Y
actual_price = np.expm1(actual_log)

for name, log_preds in model_log_predictions.items():
    log_preds = np.array(log_preds)
    r2_log = r2_score(actual_log, log_preds)
    rmse_log = math.sqrt(mean_squared_error(actual_log, log_preds))
    mae_log = mean_absolute_error(actual_log, log_preds)
    preds_price = np.expm1(log_preds)
    model_predictions_exp[name] = preds_price
    residuals_exp[name] = actual_price - preds_price
    rmse_price = math.sqrt(mean_squared_error(actual_price, preds_price))
    mae_price = mean_absolute_error(actual_price, preds_price)
    metrics_rows.append({
        'Model': name,
        'R2_log': r2_log,
        'RMSE_log': rmse_log,
        'MAE_log': mae_log,
        'RMSE_price': rmse_price,
        'MAE_price': mae_price
    })

performance_df = pd.DataFrame(metrics_rows).sort_values(by='R2_log', ascending=False)
print(performance_df)

print("\n--- 5.6 Visualizing Model Performance Comparisons ---")
if not performance_df.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=performance_df, x='R2_log', y='Model', palette='viridis')
    plt.title('Model R-squared on Training Data (log(SalePrice))')
    plt.xlabel('R-squared (log scale)')
    plt.ylabel('Model')
    plt.xlim(0, 1)
    plt.tight_layout()
    save_figure('model_r2_scores.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=performance_df, x='RMSE_price', y='Model', palette='mako')
    plt.title('Model RMSE on Training Data (SalePrice)')
    plt.xlabel('RMSE ($)')
    plt.ylabel('Model')
    plt.tight_layout()
    save_figure('model_rmse_scores.png')
    plt.show()

if model_predictions_exp:
    print("Generating Actual vs Predicted scatter plots...")
    num_models = len(model_predictions_exp)
    cols = 3
    rows = math.ceil(num_models / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()
    for ax, (name, preds_price) in zip(axes, model_predictions_exp.items()):
        ax.scatter(actual_price, preds_price, alpha=0.3, s=20)
        min_val = min(actual_price.min(), preds_price.min())
        max_val = max(actual_price.max(), preds_price.max())
        ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1)
        ax.set_title(name)
        ax.set_xlabel('Actual SalePrice')
        ax.set_ylabel('Predicted SalePrice')
    for ax in axes[len(model_predictions_exp):]:
        ax.set_visible(False)
    plt.tight_layout()
    save_figure('actual_vs_predicted_comparison.png')
    plt.show()

    print("Generating residual distribution plots...")
    residuals_df = pd.DataFrame({name: residuals for name, residuals in residuals_exp.items()})
    residuals_melted = residuals_df.melt(var_name='Model', value_name='Residual ($)')
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=residuals_melted, x='Model', y='Residual ($)', palette='Set3')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title('Residual Distributions (Actual - Predicted SalePrice)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_figure('residual_distributions.png')
    plt.show()

# Visualize tree-based feature importances if available
if tree_feature_importances:
    print("Plotting tree-based feature importances...")
    for model_name, importance_series in tree_feature_importances.items():
        if importance_series is None or importance_series.empty:
            continue
        top_features = importance_series.sort_values(ascending=False).head(15)
        plt.figure(figsize=(8, 6))
        top_features.sort_values().plot(kind='barh', color='teal')
        plt.title(f'Top 15 Feature Importances ({model_name})')
        plt.xlabel('Importance')
        plt.tight_layout()
        save_figure(f'tree_feature_importances_{model_name.lower()}.png')
        plt.show()


# =============================================================================
# --- 6. Model Diagnostics (for OLS) ---
# =============================================================================
print("\n--- 6. Generating Model Diagnostic Plots (for OLS) ---")

residuals_ols = model_ols.resid
fitted_ols = model_ols.fittedvalues

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=fitted_ols, y=residuals_ols, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('OLS: Residuals vs. Fitted Values')
plt.xlabel('Fitted Values (log(SalePrice))')
plt.ylabel('Residuals')

plt.subplot(1, 2, 2)
sm.qqplot(residuals_ols, line='45', fit=True)
plt.title('OLS: Normal Q-Q Plot of Residuals')
plt.tight_layout()
save_figure('ols_diagnostics.png')
plt.show()

# --- Plotting Lasso Feature Importances ---
print("\n--- 6.5 Plotting Top 20 Feature Importances (from Lasso) ---")
try:
    lasso_coeffs = pd.Series(model_lasso.coef_, index=feature_names)
    top_20_coeffs = lasso_coeffs.abs().nlargest(20)
    top_20_features = lasso_coeffs.loc[top_20_coeffs.index]

    plt.figure(figsize=(10, 8))
    top_20_features.sort_values().plot(kind='barh')
    plt.title('Top 20 Feature Importances (Lasso Model)')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    save_figure('lasso_top20_features.png')
    plt.show()
except Exception as e:
    print(f"Could not plot feature importances: {e}")

print("\n--- 6.6 Submission Prediction Distribution Comparisons ---")
submission_predictions = {}
submissions_folder = 'submissions'
if os.path.isdir(submissions_folder):
    submission_files = sorted(glob(os.path.join(submissions_folder, 'submission_*.csv')))
    if not submission_files:
        print("No submission CSVs found in submissions/; skipping submission plots.")
    for submission_path in submission_files:
        try:
            submission_df = pd.read_csv(submission_path)
            if 'SalePrice' not in submission_df.columns:
                continue
            model_label = os.path.splitext(os.path.basename(submission_path))[0].replace('submission_', '').upper()
            submission_predictions[model_label] = submission_df['SalePrice'].values
        except Exception as e:
            print(f"Skipping {submission_path}: {e}")
else:
    print("submissions/ folder not found; skipping submission plots.")

if submission_predictions:
    plt.figure(figsize=(12, 6))
    for label, values in submission_predictions.items():
        sns.kdeplot(values, label=label, linewidth=1.5)
    plt.title('Submission SalePrice Distributions by Model')
    plt.xlabel('Predicted SalePrice ($)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    save_figure('submission_distributions_kde.png')
    plt.show()

    submissions_df = pd.DataFrame(dict(sorted(submission_predictions.items())))
    submissions_melt = submissions_df.melt(var_name='Model', value_name='SalePrice')
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=submissions_melt, x='Model', y='SalePrice', palette='pastel')
    plt.title('Submission SalePrice Distribution (Boxplot)')
    plt.xlabel('Model')
    plt.ylabel('SalePrice ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_figure('submission_distributions_box.png')
    plt.show()
else:
    print("No submission predictions available for plotting.")

# =============================================================================
# --- 7. SPECIFIC CONCLUSIONS (DELIVERABLES) ---
# =============================================================================
print("\n\n" + "="*60)
print("--- 7. SPECIFIC CONCLUSIONS (DELIVERABLES) ---")
print("="*60 + "\n")

print("1. Top Correlated Features (from original numeric data before outlier removal):")
# Your code correctly reads the original file for this, which is good practice
correlations_orig = pd.read_csv('train.csv').corr(numeric_only=True)['SalePrice'].abs().sort_values(ascending=False)
print(correlations_orig.head(6)[1:])

print("\n2. Quality Impact (OverallQual):")
ols_qual_coeff = model_ols.params['OverallQual']
print(f"OLS Coefficient for log(SalePrice) on OverallQual (cleaned data): {ols_qual_coeff:.4f}")
print(f"Interpretation: For each 1-point increase in OverallQual, the log(SalePrice) is expected to increase by {ols_qual_coeff:.4f}, holding other factors constant.")

print("\n3. Model Comparison (R-squared on log(SalePrice), Cleaned Data):")
print("Performance after removing extreme GrLivArea outliers.")
print(f"OLS R-squared:        {r2_ols:.4f}")
print(f"GLM R-squared:        {r2_glm:.4f} (Similar to OLS)")
print(f"RLM R-squared:        {r2_rlm:.4f} (Robust)")
print(f"Ridge R-squared:      {r2_ridge:.4f} (Robust, handles multicollinearity)")
print(f"Lasso R-squared:      {r2_lasso:.4f} (Robust, {n_features_lasso} features)")
print(f"ElasticNet R-squared: {r2_elastic:.4f} (Robust, {n_features_elastic} features, L1 ratio={best_l1_ratio:.2f})")

print("\n4. Living Area Impact (GrLivArea on PricePerSqFt, Cleaned Data):")
df_calc = df.copy() # df is already cleaned
df_calc['TotalBsmtSF'] = df_calc['TotalBsmtSF'].fillna(0)
df_calc['TotalSqFt'] = df_calc['TotalBsmtSF'] + df_calc['GrLivArea']
df_calc = df_calc[df_calc['TotalSqFt'] > 0]
df_calc['PricePerSqFt'] = df_calc[target_col] / df_calc['TotalSqFt']
livarea_impact_corr = df_calc[['GrLivArea', 'PricePerSqFt']].corr(numeric_only=True).iloc[0, 1]
print(f"Correlation between GrLivArea and PricePerSqFt (cleaned data): {livarea_impact_corr:.4f}")
print("Interpretation: A positive correlation suggests that larger living areas also tend to have a higher price per square foot (e.g., higher quality finishes).")

print(f"\n5. Best Performer & Feature Selection (Cleaned Data):")
# Determine best R-squared
models_r2 = {'OLS': r2_ols, 'RLM': r2_rlm, 'GLM': r2_glm, 
             'Lasso': r2_lasso, 'Ridge': r2_ridge, 'ElasticNet': r2_elastic}
best_model_name = max(models_r2, key=models_r2.get)
best_r2 = models_r2[best_model_name]

print(f"Best performing model based on R-squared is {best_model_name} with R2 = {best_r2:.4f}.")
print("Justification: The regularized models (Lasso, Ridge, ElasticNet) often provide the most reliable and generalizable results, as they penalize complexity and select important features.")
print(f"Lasso selected {n_features_lasso} features (eliminated {X.shape[1] - n_features_lasso}).")
print(f"ElasticNet (L1 ratio={best_l1_ratio:.2f}) selected {n_features_elastic} features (eliminated {X.shape[1] - n_features_elastic}).")
print("\n--- Analysis Complete ---")