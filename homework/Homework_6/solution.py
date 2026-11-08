"""
Homework 6: Decision Trees and Ensemble Learning
ML Zoomcamp 2026

Predicting car fuel efficiency (fuel_efficiency_mpg) using
Decision Trees, Random Forests, and XGBoost.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# -- Load dataset --------------------------------------------------------------
df = pd.read_csv('car_fuel_efficiency_2026_homework6.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# -- Preparing the dataset ----------------------------------------------------
# Fill missing values with zeros
df = df.fillna(0)

# Split
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=1
)

print(f"\nTrain size: {len(df_train)}")
print(f"Validation size: {len(df_val)}")
print(f"Test size: {len(df_test)}")

# Separate target
y_train = df_train['fuel_efficiency_mpg'].values
y_val = df_val['fuel_efficiency_mpg'].values
y_test = df_test['fuel_efficiency_mpg'].values

# Remove target from features
del df_train['fuel_efficiency_mpg']
del df_val['fuel_efficiency_mpg']
del df_test['fuel_efficiency_mpg']

# Use DictVectorizer
dv = DictVectorizer(sparse=True)

train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')
test_dicts = df_test.to_dict(orient='records')

X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)
X_test = dv.transform(test_dicts)

feature_names = dv.get_feature_names_out()
print(f"\nNumber of features after DictVectorizer: {len(feature_names)}")

# ==============================================================================
# Question 1: Decision Tree with max_depth=1 - which feature is used?
# ==============================================================================
print("\n" + "=" * 70)
print("QUESTION 1: Decision Tree (max_depth=1) - splitting feature")
print("=" * 70)

dt = DecisionTreeRegressor(max_depth=1)
dt.fit(X_train, y_train)

# The feature used for the split at the root
split_feature_idx = dt.tree_.feature[0]
split_feature_name = feature_names[split_feature_idx]
print(f"Feature used for splitting: {split_feature_name}")

# Map one-hot feature back to original column name
if '=' in split_feature_name:
    original_feature = split_feature_name.split('=')[0]
else:
    original_feature = split_feature_name
print(f"Original feature name: {original_feature}")

# ==============================================================================
# Question 2: Random Forest (n_estimators=10) - RMSE on validation
# ==============================================================================
print("\n" + "=" * 70)
print("QUESTION 2: Random Forest (n_estimators=10) - validation RMSE")
print("=" * 70)

rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_val = rf.predict(X_val)
rmse_q2 = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"RMSE on validation: {rmse_q2:.4f}")

# ==============================================================================
# Question 3: Experiment with n_estimators - which gives lowest RMSE?
# ==============================================================================
print("\n" + "=" * 70)
print("QUESTION 3: Best n_estimators from [10, 50, 100, 150]")
print("=" * 70)

n_estimators_list = [10, 50, 100, 150]
best_rmse_q3 = float('inf')
best_n = None

for n in n_estimators_list:
    rf_n = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    rf_n.fit(X_train, y_train)
    y_pred = rf_n.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"  n_estimators={n:>3d} -> RMSE = {rmse:.4f}")
    if rmse < best_rmse_q3:
        best_rmse_q3 = rmse
        best_n = n

print(f"\nBest n_estimators: {best_n} (RMSE = {best_rmse_q3:.4f})")

# ==============================================================================
# Question 4: Best max_depth using mean RMSE
# ==============================================================================
print("\n" + "=" * 70)
print("QUESTION 4: Best max_depth from [10, 15, 20, 25] (mean RMSE)")
print("=" * 70)

max_depth_list = [10, 15, 20, 25]
best_mean_rmse = float('inf')
best_depth = None

for depth in max_depth_list:
    rmse_values = []
    for n in n_estimators_list:
        rf_d = RandomForestRegressor(
            n_estimators=n, max_depth=depth, random_state=1, n_jobs=-1
        )
        rf_d.fit(X_train, y_train)
        y_pred = rf_d.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_values.append(rmse)
    mean_rmse = np.mean(rmse_values)
    print(f"  max_depth={depth:>2d} -> mean RMSE = {mean_rmse:.4f}  "
          f"(individual: {[f'{r:.4f}' for r in rmse_values]})")
    if mean_rmse < best_mean_rmse:
        best_mean_rmse = mean_rmse
        best_depth = depth

print(f"\nBest max_depth: {best_depth} (mean RMSE = {best_mean_rmse:.4f})")

# ==============================================================================
# Question 5: Feature importance
# ==============================================================================
print("\n" + "=" * 70)
print("QUESTION 5: Most important feature")
print("=" * 70)

rf_imp = RandomForestRegressor(
    n_estimators=10, max_depth=20, random_state=1, n_jobs=-1
)
rf_imp.fit(X_train, y_train)

importances = rf_imp.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 10 features by importance:")
print(feature_importance_df.head(10).to_string(index=False))

# Check the 4 candidate features
candidates = ['vehicle_weight', 'horsepower', 'acceleration', 'engine_displacement']
print("\nImportance of candidate features:")
for c in candidates:
    imp = feature_importance_df[feature_importance_df['feature'] == c]['importance'].values
    if len(imp) > 0:
        print(f"  {c}: {imp[0]:.6f}")
    else:
        print(f"  {c}: not found directly")

# ==============================================================================
# Question 6: XGBoost - eta=0.3 vs eta=0.1
# ==============================================================================
print("\n" + "=" * 70)
print("QUESTION 6: XGBoost - eta=0.3 vs eta=0.1")
print("=" * 70)

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(feature_names))
dval = xgb.DMatrix(X_val, label=y_val, feature_names=list(feature_names))

watchlist = [(dtrain, 'train'), (dval, 'val')]

# eta = 0.3
xgb_params_03 = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

print("\n--- Training XGBoost with eta=0.3 ---")
evals_result_03 = {}
model_03 = xgb.train(
    xgb_params_03, dtrain, num_boost_round=100,
    evals=watchlist, evals_result=evals_result_03, verbose_eval=10
)

rmse_03 = evals_result_03['val']['rmse'][-1]
print(f"\nFinal validation RMSE (eta=0.3): {rmse_03:.4f}")

# eta = 0.1
xgb_params_01 = {
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

print("\n--- Training XGBoost with eta=0.1 ---")
evals_result_01 = {}
model_01 = xgb.train(
    xgb_params_01, dtrain, num_boost_round=100,
    evals=watchlist, evals_result=evals_result_01, verbose_eval=10
)

rmse_01 = evals_result_01['val']['rmse'][-1]
print(f"\nFinal validation RMSE (eta=0.1): {rmse_01:.4f}")

# Compare
print("\n--- Comparison ---")
print(f"  eta=0.3 -> val RMSE = {rmse_03:.4f}")
print(f"  eta=0.1 -> val RMSE = {rmse_01:.4f}")

if rmse_03 < rmse_01:
    print("  -> eta=0.3 is better")
elif rmse_01 < rmse_03:
    print("  -> eta=0.1 is better")
else:
    print("  -> Both give equal value")

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "=" * 70)
print("SUMMARY OF ANSWERS")
print("=" * 70)
print(f"Q1: Splitting feature = {original_feature}")
print(f"Q2: RMSE (RF, n=10) = {rmse_q2:.4f}")
print(f"Q3: Best n_estimators = {best_n}")
print(f"Q4: Best max_depth = {best_depth}")

# Find most important among candidates
best_candidate = None
best_candidate_imp = -1
for c in candidates:
    imp = feature_importance_df[feature_importance_df['feature'] == c]['importance'].values
    if len(imp) > 0 and imp[0] > best_candidate_imp:
        best_candidate_imp = imp[0]
        best_candidate = c
print(f"Q5: Most important feature = {best_candidate}")
print(f"Q6: Best eta = {'0.3' if rmse_03 <= rmse_01 else '0.1'}")
