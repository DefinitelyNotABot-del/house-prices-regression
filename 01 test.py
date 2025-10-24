import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Set pandas options to avoid flooding the console
pd.set_option('display.max_rows', 10)

print("--- 1. Loading Data ---")
df = pd.read_csv('train.csv')
target_col = 'SalePrice'

# --- Task 1: Exploratory Data Analysis (EDA) ---
print("--- 1. Unclean Data Visualization (EDA) ---")
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(df[target_col], kde=True, bins=40)
plt.title('Unclean SalePrice Distribution (Skewed)')

plt.subplot(1, 2, 2)
sns.scatterplot(x=df['GrLivArea'], y=df[target_col], alpha=0.5)
plt.title('Unclean GrLivArea vs SalePrice')
plt.tight_layout()
plt.show()

# --- Task 2: Data Preprocessing (Professional) ---
print("\n--- 2. Data Preprocessing (Full 79) ---")

# Professional Step: Log-transform the target variable for a better model fit
# This is our new 'Y'
Y = np.log1p(df[target_col])
# X is everything else except the target and ID
X = df.drop([target_col, 'Id'], axis=1)

print(f"Original feature count: {X.shape[1]}")

# Separate columns by type
numeric_cols = X.select_dtypes(include=np.number).columns
object_cols = X.select_dtypes(include='object').columns

print(f"Found {len(numeric_cols)} numeric and {len(object_cols)} object columns.")

# Impute (fill) numeric columns with median (more robust to outliers)
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
# Impute (fill) object columns (treating 'NaN' as a separate category 'Missing')
X[object_cols] = X[object_cols].fillna('Missing')

# One-hot encode all object columns (e.g., 'Neighborhood' -> 'Neighborhood_NAmes', etc.)
X = pd.get_dummies(X, columns=object_cols, drop_first=True)

print(f"New feature count after get_dummies: {X.shape[1]}")

# --- THE CRITICAL FIX ---
# Convert all columns (int, float, and bool) to float64
print("Converting all columns to float64 to fix 'object' error...")
X = X.astype(float) 

# Add constant for statsmodels
X = sm.add_constant(X, prepend=False)

print("--- Preprocessing Complete. All columns are float64. ---")


print("\n--- 3. Clean Data Visualization (Processed) ---")
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(Y, kde=True, bins=30)
plt.title('Clean log(SalePrice) Distribution (Normalized)')

plt.subplot(1, 2, 2)
sns.scatterplot(x=X['GrLivArea'], y=Y, alpha=0.5)
plt.title('GrLivArea vs log(SalePrice)')
plt.tight_layout()
plt.show()

# --- Task 3: Model Building ---
print("\n--- 4. Model Building ---")

print("\n--- OLS Model Summary ---")
model_ols = sm.OLS(Y, X).fit()
print(model_ols.summary())

print("\n--- RLM Model ---")
# RLM is very slow with this many features, but it will run.
model_rlm = sm.RLM(Y, X).fit()
print("RLM model fitted.")

print("\n--- GLM Model Summary ---")
model_glm = sm.GLM(Y, X, family=sm.families.Gaussian()).fit()
print(model_glm.summary())


# --- Task 4 & Deliverables ---
print("\n\n" + "="*60)
print("--- 5. SPECIFIC CONCLUSIONS (DELIVERABLES) ---")
print("="*60 + "\n")

print("1. Top Correlated Features (from original numeric data):")
# We use the original 'df' for this calculation as required
correlations = df.corr(numeric_only=True)['SalePrice'].abs().sort_values(ascending=False)
print(correlations.head(6)[1:]) # [1:] to skip SalePrice itself

print("\n2. Quality Impact (OverallQual):")
# We get this from the OLS model summary we just printed
ols_qual_coeff = model_ols.params['OverallQual']
print(f"OLS Coefficient for log(SalePrice) on OverallQual: {ols_qual_coeff:.4f}")

# --- R-squared Calculation for all models ---
r2_ols = model_ols.rsquared

# Manual R-squared for RLM (since it doesn't have an .rsquared attribute)
ss_tot_log = np.sum((Y - Y.mean())**2)
ss_res_rlm_log = np.sum(model_rlm.resid**2)
r2_rlm = 1 - (ss_res_rlm_log / ss_tot_log)

# Manual R-squared for GLM (for a consistent comparison)
ss_res_glm_log = np.sum((Y - model_glm.predict(X))**2)
r2_glm = 1 - (ss_res_glm_log / ss_tot_log)

print("\n3. Model Comparison (R-squared on log(SalePrice)):")
print(f"OLS R-squared:   {r2_ols:.4f}")
print(f"RLM R-squared:   {r2_rlm:.4f}")
print(f"GLM R-squared:   {r2_glm:.4f}")

print("\n4. Living Area Impact (GrLivArea on PricePerSqFt):")
# We use the original 'df' for this as well, filling NaNs just for this calculation
df_calc = df.copy()
df_calc['TotalBsmtSF'] = df_calc['TotalBsmtSF'].fillna(0)
df_calc['TotalSqFt'] = df_calc['TotalBsmtSF'] + df_calc['GrLivArea']
# Avoid division by zero if TotalSqFt is 0
df_calc = df_calc[df_calc['TotalSqFt'] > 0]
df_calc['PricePerSqFt'] = df_calc[target_col] / df_calc['TotalSqFt']
livarea_impact_corr = df_calc[['GrLivArea', 'PricePerSqFt']].corr(numeric_only=True).iloc[0, 1]
print(f"Correlation between GrLivArea and PricePerSqFt: {livarea_impact_corr:.4f}")

print("\n5. Best Performer:")
models_r2 = {'OLS': r2_ols, 'RLM': r2_rlm, 'GLM': r2_glm}
best_model_name = max(models_r2, key=models_r2.get)
best_r2 = models_r2[best_model_name]
print(f"Best performing model based on R-squared is {best_model_name} with R2 = {best_r2:.4f}.")
