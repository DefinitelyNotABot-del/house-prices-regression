# train_save_models.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib
import warnings
import os # Import os module

# --- Create Folders ---
models_folder = 'saved_models'
if not os.path.exists(models_folder):
    os.makedirs(models_folder)
    print(f"Created folder: {models_folder}")
# ---

# Suppress warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', '{:.6f}'.format)

print("--- 1. Loading Full Data (1460 rows) ---")
df = pd.read_csv('train.csv')
target_col = 'SalePrice'
print(f"Original dataset shape: {df.shape}")

print("\n--- 1.5 Explicit Outlier Removal ---")
initial_rows = df.shape[0]
df = df[df['GrLivArea'] < 4000]
rows_removed = initial_rows - df.shape[0]
print(f"Removed {rows_removed} outliers based on GrLivArea > 4000.")
print(f"New dataset shape: {df.shape}")

print("\n--- 2. Data Preprocessing (Full 79 Features) ---")
Y = np.log1p(df[target_col])
X = df.drop([target_col, 'Id'], axis=1)

print(f"Original feature count: {X.shape[1]}")
numeric_cols = X.select_dtypes(include=np.number).columns
object_cols = X.select_dtypes(include='object').columns
print(f"Found {len(numeric_cols)} numeric and {len(object_cols)} object columns.")

# --- Save Medians ---
train_medians = X[numeric_cols].median()
joblib.dump(train_medians, os.path.join(models_folder, 'train_medians.joblib')) # Save in folder
print(f"Saved training data medians to {models_folder}/")

X[numeric_cols] = X[numeric_cols].fillna(train_medians)
X[object_cols] = X[object_cols].fillna('Missing')

X = pd.get_dummies(X, columns=object_cols, drop_first=True)
print(f"New feature count after get_dummies: {X.shape[1]}")
feature_names = X.columns.tolist()

# --- Save Column List ---
joblib.dump(feature_names, os.path.join(models_folder, 'feature_names.joblib')) # Save in folder
print(f"Saved feature names to {models_folder}/")

# --- 3. Create Datasets for Models ---
X_with_const = sm.add_constant(X.astype(float), prepend=False)

print("Scaling data for Regularized Models...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

# --- Save Scaler ---
joblib.dump(scaler, os.path.join(models_folder, 'scaler.joblib')) # Save in folder
print(f"Saved the fitted StandardScaler to {models_folder}/")

print("\n--- 4. Model Building & Saving (On Outlier-Cleaned Dataset) ---")

# --- OLS Model ---
print("\n--- OLS Model Summary (Baseline on Cleaned Data) ---")
model_ols = sm.OLS(Y, X_with_const).fit()
r2_ols = model_ols.rsquared
print(model_ols.summary())

# --- RLM Model ---
print("\n--- RLM Model (Robust to any remaining Outliers) ---")
model_rlm = sm.RLM(Y, X_with_const).fit()
ss_tot_log = np.sum((Y - Y.mean())**2)
ss_res_rlm_log = np.sum(model_rlm.resid**2)
r2_rlm = 1 - (ss_res_rlm_log / ss_tot_log)
print("RLM model fitted.")

# --- GLM Model ---
print("\n--- GLM Model Summary (Similar to OLS on Cleaned Data) ---")
model_glm = sm.GLM(Y, X_with_const, family=sm.families.Gaussian()).fit()
ss_res_glm_log = np.sum((Y - model_glm.predict(X_with_const))**2)
r2_glm = 1 - (ss_res_glm_log / ss_tot_log)
print(model_glm.summary())

# --- LassoCV Model ---
print("\n--- LassoCV Model Details (Robust + Feature Selection - L1) ---")
model_lasso = LassoCV(cv=5, random_state=0, max_iter=10000).fit(X_scaled, Y)
r2_lasso = model_lasso.score(X_scaled, Y)
best_alpha_lasso = model_lasso.alpha_
lasso_coefs = pd.Series(model_lasso.coef_, index=feature_names)
n_features_lasso = np.sum(lasso_coefs != 0)
joblib.dump(model_lasso, os.path.join(models_folder, 'lasso_model.joblib')) # Save in folder
print(f"LassoCV Best Alpha: {best_alpha_lasso:.6f}")
print(f"LassoCV R-squared: {r2_lasso:.4f}")
print(f"LassoCV Features Selected: {n_features_lasso}")
print("\nTop 10 Lasso Coefficients (Absolute Value):")
print(lasso_coefs.abs().sort_values(ascending=False).head(10))
print(f"LassoCV model fitted and saved to {models_folder}/")

# --- RidgeCV Model ---
print("\n--- RidgeCV Model Details (Robust + Handles Multicollinearity - L2) ---")
alphas_ridge = np.logspace(-3, 4, 100) 
model_ridge = RidgeCV(alphas=alphas_ridge, store_cv_results=True).fit(X_scaled, Y)
r2_ridge = model_ridge.score(X_scaled, Y)
best_alpha_ridge = model_ridge.alpha_  
ridge_coefs = pd.Series(model_ridge.coef_, index=feature_names)
n_features_ridge = X_scaled.shape[1]
joblib.dump(model_ridge, os.path.join(models_folder, 'ridge_model.joblib')) # Save in folder
print(f"RidgeCV Best Alpha: {best_alpha_ridge:.6f}")
print(f"RidgeCV R-squared: {r2_ridge:.4f}")
print(f"RidgeCV Features Selected: {n_features_ridge} (All are kept, coefficients shrunk)")
print("\nTop 10 Ridge Coefficients (Absolute Value):")
print(ridge_coefs.abs().sort_values(ascending=False).head(10))
print(f"RidgeCV model fitted and saved to {models_folder}/")

# --- ElasticNetCV Model ---
print("\n--- ElasticNetCV Model Details (Robust + Feature Selection/Multicollinearity - L1 & L2) ---")
model_elastic = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, random_state=0, max_iter=10000).fit(X_scaled, Y)
r2_elastic = model_elastic.score(X_scaled, Y)
best_alpha_elastic = model_elastic.alpha_
best_l1_ratio_elastic = model_elastic.l1_ratio_
elastic_coefs = pd.Series(model_elastic.coef_, index=feature_names)
n_features_elastic = np.sum(elastic_coefs != 0)
joblib.dump(model_elastic, os.path.join(models_folder, 'elasticnet_model.joblib')) # Save in folder
print(f"ElasticNetCV Best Alpha: {best_alpha_elastic:.6f}")
print(f"ElasticNetCV Best L1 Ratio: {best_l1_ratio_elastic:.2f}")
print(f"ElasticNetCV R-squared: {r2_elastic:.4f}")
print(f"ElasticNetCV Features Selected: {n_features_elastic}")
print("\nTop 10 ElasticNet Coefficients (Absolute Value):")
print(elastic_coefs.abs().sort_values(ascending=False).head(10))
print(f"ElasticNetCV model fitted and saved to {models_folder}/")

print("\n\n" + "="*60)

# --- Tree-based / Gradient Boosting Models: RandomForest, XGBoost, LightGBM ---
print("\n--- RandomForest, XGBoost, LightGBM (Tree-based Models) ---")
from pprint import pprint

# Use unscaled X for tree-based models (they don't require scaling)
X_for_trees = X.astype(float)

try:
    from sklearn.ensemble import RandomForestRegressor
    print("Training RandomForestRegressor...")
    rf = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
    rf.fit(X_for_trees, Y)
    joblib.dump(rf, os.path.join(models_folder, 'rf_model.joblib'))
    print("RandomForest trained and saved to {}/rf_model.joblib".format(models_folder))
    print("RandomForest parameters:")
    pprint(rf.get_params())
    try:
        importances = pd.Series(rf.feature_importances_, index=feature_names)
        print("Top 20 RandomForest feature importances:")
        print(importances.abs().sort_values(ascending=False).head(20))
    except Exception:
        print("Could not compute RandomForest feature importances.")
except Exception as e:
    print(f"Skipping RandomForest: {e}")

try:
    import xgboost as xgb
    print("Training XGBoost (XGBRegressor)...")
    xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=0, verbosity=0)
    xgb_model.fit(X_for_trees, Y)
    joblib.dump(xgb_model, os.path.join(models_folder, 'xgb_model.joblib'))
    print("XGBoost trained and saved to {}/xgb_model.joblib".format(models_folder))
    print("XGBoost parameters:")
    pprint(xgb_model.get_params())
    try:
        importances = pd.Series(xgb_model.feature_importances_, index=feature_names)
        print("Top 20 XGBoost feature importances:")
        print(importances.abs().sort_values(ascending=False).head(20))
    except Exception:
        print("Could not compute XGBoost feature importances.")
except Exception as e:
    print(f"Skipping XGBoost: {e}")

try:
    import lightgbm as lgb
    print("Training LightGBM (LGBMRegressor)...")
    lgbm_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=0)
    lgbm_model.fit(X_for_trees, Y)
    joblib.dump(lgbm_model, os.path.join(models_folder, 'lgbm_model.joblib'))
    print("LightGBM trained and saved to {}/lgbm_model.joblib".format(models_folder))
    print("LightGBM parameters:")
    pprint(lgbm_model.get_params())
    try:
        importances = pd.Series(lgbm_model.feature_importances_, index=feature_names)
        print("Top 20 LightGBM feature importances:")
        print(importances.abs().sort_values(ascending=False).head(20))
    except Exception:
        print("Could not compute LightGBM feature importances.")
except Exception as e:
    print(f"Skipping LightGBM: {e}")

print("\n" + "="*60)
print("--- 5. SPECIFIC CONCLUSIONS (DELIVERABLES) ---")
print("\n--- Model Training and Saving Complete ---")