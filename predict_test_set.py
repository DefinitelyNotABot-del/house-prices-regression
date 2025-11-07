# predict_test_set.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
import joblib
import warnings
import os # Import os module

# --- Define Folders ---
models_folder = 'saved_models'
submissions_folder = 'submissions'
if not os.path.exists(submissions_folder):
    os.makedirs(submissions_folder)
    print(f"Created folder: {submissions_folder}")
# ---

# Suppress warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.6f}'.format)
target_col = 'SalePrice'

print("--- 1. Loading Test Data and Saved Objects ---")
try:
    df_test = pd.read_csv('test.csv')
    test_ids = df_test['Id']
    print(f"Test data loaded. Shape: {df_test.shape}")
except FileNotFoundError:
    print("Error: test.csv not found in the current directory.")
    exit()

# Load preprocessing objects from saved_models folder
try:
    train_medians = joblib.load(os.path.join(models_folder, 'train_medians.joblib'))
    feature_names = joblib.load(os.path.join(models_folder, 'feature_names.joblib'))
    scaler = joblib.load(os.path.join(models_folder, 'scaler.joblib'))
    print(f"Loaded medians, feature names, and scaler from {models_folder}/")
except FileNotFoundError:
    print(f"Error: Could not find preprocessing files in '{models_folder}'.")
    print("Please run the training script first.")
    exit()

# Load trained sklearn models from saved_models folder
try:
    model_lasso = joblib.load(os.path.join(models_folder, 'lasso_model.joblib'))
    model_ridge = joblib.load(os.path.join(models_folder, 'ridge_model.joblib'))
    model_elastic = joblib.load(os.path.join(models_folder, 'elasticnet_model.joblib'))
    print(f"Loaded Lasso, Ridge, and ElasticNet models from {models_folder}/")
except FileNotFoundError:
    print(f"Error: Could not find model files in '{models_folder}'.")
    print("Please run the training script first.")
    exit()

# Attempt to load tree-based models (may be absent if training skipped or package missing)
model_rf = None
model_xgb = None
model_lgbm = None
try:
    model_rf = joblib.load(os.path.join(models_folder, 'rf_model.joblib'))
    print("Loaded RandomForest model from rf_model.joblib")
    try:
        from pprint import pprint
        print("RandomForest parameters:")
        pprint(model_rf.get_params())
    except Exception:
        pass
except FileNotFoundError:
    print("RandomForest model file not found; skipping RF predictions.")

try:
    model_xgb = joblib.load(os.path.join(models_folder, 'xgb_model.joblib'))
    print("Loaded XGBoost model from xgb_model.joblib")
    try:
        from pprint import pprint
        print("XGBoost parameters:")
        pprint(model_xgb.get_params())
    except Exception:
        pass
except FileNotFoundError:
    print("XGBoost model file not found; skipping XGBoost predictions.")

try:
    model_lgbm = joblib.load(os.path.join(models_folder, 'lgbm_model.joblib'))
    print("Loaded LightGBM model from lgbm_model.joblib")
    try:
        from pprint import pprint
        print("LightGBM parameters:")
        pprint(model_lgbm.get_params())
    except Exception:
        pass
except FileNotFoundError:
    print("LightGBM model file not found; skipping LightGBM predictions.")

print("\n--- 2. Preprocessing Test Data (Applying Training Steps) ---")
X_test = df_test.drop(['Id'], axis=1)

numeric_cols_test = X_test.select_dtypes(include=np.number).columns
object_cols_test = X_test.select_dtypes(include='object').columns

common_numeric_cols = list(set(numeric_cols_test) & set(train_medians.index))
X_test[common_numeric_cols] = X_test[common_numeric_cols].fillna(train_medians[common_numeric_cols])

numeric_na_remaining = X_test[numeric_cols_test].isnull().sum()
if numeric_na_remaining.sum() > 0:
    print(f"Warning: Found NaNs in numeric test columns after median imputation: \n{numeric_na_remaining[numeric_na_remaining > 0]}")
    print("Filling remaining numeric NaNs with 0.")
    X_test[numeric_cols_test] = X_test[numeric_cols_test].fillna(0)

X_test[object_cols_test] = X_test[object_cols_test].fillna('Missing')
X_test = pd.get_dummies(X_test, columns=object_cols_test, drop_first=True)
print(f"Test data feature count after get_dummies: {X_test.shape[1]}")

# Align columns
missing_cols = set(feature_names) - set(X_test.columns)
print(f"Adding {len(missing_cols)} columns to test set.")
for c in missing_cols:
    X_test[c] = 0

extra_cols = set(X_test.columns) - set(feature_names)
if extra_cols:
    print(f"Warning: Removing {len(extra_cols)} columns from test set.")
    X_test = X_test.drop(columns=list(extra_cols))

X_test = X_test[feature_names]
print(f"Test data columns aligned. Shape: {X_test.shape}")

# --- 3. Create Datasets for Predictions ---
X_test_with_const = sm.add_constant(X_test.astype(float), prepend=False, has_constant='skip')
if 'const' not in X_test_with_const.columns: X_test_with_const['const'] = 1.0
else: X_test_with_const['const'] = 1.0

print("Scaling test data using loaded scaler...")
X_test_scaled = scaler.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

print("\n--- 4. Making Predictions on Test Data ---")

# --- Refit statsmodels models ---
print("Reloading necessary training components to refit statsmodels...")
try:
    df_train_orig = pd.read_csv('train.csv')
    df_train_cleaned = df_train_orig[df_train_orig['GrLivArea'] < 4000]
    Y_train = np.log1p(df_train_cleaned[target_col])

    X_train_orig = df_train_cleaned.drop([target_col, 'Id'], axis=1)
    numeric_cols_train = X_train_orig.select_dtypes(include=np.number).columns
    object_cols_train = X_train_orig.select_dtypes(include='object').columns

    X_train_orig[numeric_cols_train] = X_train_orig[numeric_cols_train].fillna(train_medians)
    X_train_orig[object_cols_train] = X_train_orig[object_cols_train].fillna('Missing')

    X_train_orig = pd.get_dummies(X_train_orig, columns=object_cols_train, drop_first=True)
    X_train_orig = X_train_orig.reindex(columns=feature_names, fill_value=0)
    X_train_const_refit = sm.add_constant(X_train_orig.astype(float), prepend=False)
    print("Training data reloaded and preprocessed for refitting.")
except FileNotFoundError:
    print("Error: train.csv not found. Cannot refit statsmodels.")
    exit()
except Exception as e:
    print(f"An error occurred during statsmodels refitting setup: {e}")
    exit()

# --- Predictions (OLS, RLM, GLM refitted; others loaded) ---
print("Refitting OLS and predicting...")
ols_refit = sm.OLS(Y_train, X_train_const_refit).fit()
ols_preds_log = ols_refit.predict(X_test_with_const)
ols_preds = np.expm1(ols_preds_log)
print("OLS predictions generated.")

print("Refitting RLM and predicting...")
rlm_refit = sm.RLM(Y_train, X_train_const_refit).fit()
rlm_preds_log = rlm_refit.predict(X_test_with_const)
rlm_preds = np.expm1(rlm_preds_log)
print("RLM predictions generated.")

print("Refitting GLM and predicting...")
glm_refit = sm.GLM(Y_train, X_train_const_refit, family=sm.families.Gaussian()).fit()
glm_preds_log = glm_refit.predict(X_test_with_const)
glm_preds = np.expm1(glm_preds_log)
print("GLM predictions generated.")

print("Predicting with loaded Lasso model...")
lasso_preds_log = model_lasso.predict(X_test_scaled)
lasso_preds = np.expm1(lasso_preds_log)
print("Lasso predictions generated.")

print("Predicting with loaded Ridge model...")
ridge_preds_log = model_ridge.predict(X_test_scaled)
ridge_preds = np.expm1(ridge_preds_log)
print("Ridge predictions generated.")

print("Predicting with loaded ElasticNet model...")
elastic_preds_log = model_elastic.predict(X_test_scaled)
elastic_preds = np.expm1(elastic_preds_log)
print("ElasticNet predictions generated.")

# Predict with tree-based models if they were loaded
rf_preds = None
xgb_preds = None
lgbm_preds = None

if model_rf is not None:
    try:
        print("Predicting with RandomForest...")
        rf_preds_log = model_rf.predict(X_test)
        rf_preds = np.expm1(rf_preds_log)
        print("RandomForest predictions generated.")
        try:
            importances = pd.Series(model_rf.feature_importances_, index=feature_names)
            print("Top 20 RandomForest importances:")
            print(importances.abs().sort_values(ascending=False).head(20))
        except Exception:
            pass
    except Exception as e:
        print(f"RandomForest prediction failed: {e}")

if model_xgb is not None:
    try:
        print("Predicting with XGBoost...")
        xgb_preds_log = model_xgb.predict(X_test)
        xgb_preds = np.expm1(xgb_preds_log)
        print("XGBoost predictions generated.")
        try:
            importances = pd.Series(model_xgb.feature_importances_, index=feature_names)
            print("Top 20 XGBoost importances:")
            print(importances.abs().sort_values(ascending=False).head(20))
        except Exception:
            pass
    except Exception as e:
        print(f"XGBoost prediction failed: {e}")

if model_lgbm is not None:
    try:
        print("Predicting with LightGBM...")
        lgbm_preds_log = model_lgbm.predict(X_test)
        lgbm_preds = np.expm1(lgbm_preds_log)
        print("LightGBM predictions generated.")
        try:
            importances = pd.Series(model_lgbm.feature_importances_, index=feature_names)
            print("Top 20 LightGBM importances:")
            print(importances.abs().sort_values(ascending=False).head(20))
        except Exception:
            pass
    except Exception as e:
        print(f"LightGBM prediction failed: {e}")

print("\n--- 5. Creating Submission Files ---")

# Function to create submission CSV in the 'submissions' folder
def create_submission(ids, predictions, model_name):
    filename = f"submission_{model_name}.csv"
    file_path = os.path.join(submissions_folder, filename) # Save in folder
    submission_df = pd.DataFrame({'Id': ids, 'SalePrice': predictions})
    submission_df['SalePrice'] = submission_df['SalePrice'].clip(lower=0)
    submission_df.to_csv(file_path, index=False)
    print(f"Created submission file: {file_path}")

# Create submission files
create_submission(test_ids, ols_preds, 'ols')
create_submission(test_ids, rlm_preds, 'rlm')
create_submission(test_ids, glm_preds, 'glm')
create_submission(test_ids, lasso_preds, 'lasso')
create_submission(test_ids, ridge_preds, 'ridge')
create_submission(test_ids, elastic_preds, 'elasticnet')
if rf_preds is not None:
    create_submission(test_ids, rf_preds, 'rf')
else:
    print("RandomForest predictions unavailable; submission skipped.")
if xgb_preds is not None:
    create_submission(test_ids, xgb_preds, 'xgb')
else:
    print("XGBoost predictions unavailable; submission skipped.")
if lgbm_preds is not None:
    create_submission(test_ids, lgbm_preds, 'lgbm')
else:
    print("LightGBM predictions unavailable; submission skipped.")
print(f"\n--- Testing and Submission File Generation Complete (Files in '{submissions_folder}') ---")