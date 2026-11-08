# Machine Learning Zoomcamp 2026 - Homework 6 Solutions
## Decision Trees and Ensemble Learning

This document provides the complete, reproducible solutions and explanations for **Homework 6: Decision Trees and Ensemble Learning** from DataTalksClub's Machine Learning Zoomcamp (2026 cohort).

---

### Dataset Overview & Preparation

* **Dataset source**: `car_fuel_efficiency_2026.csv` (2026 cohort release)
* **Goal**: Predict car fuel efficiency (column: `fuel_efficiency_mpg`)
* **Total records**: 10,000 rows, 11 columns
* **Target variable**: `fuel_efficiency_mpg` (continuous numerical target)

#### Dataset Columns:
1. `model_year`
2. `origin`
3. `fuel_type`
4. `drivetrain`
5. `num_doors`
6. `engine_displacement`
7. `num_cylinders`
8. `horsepower` (contains 877 missing values)
9. `vehicle_weight`
10. `acceleration` (contains 264 missing values)
11. `fuel_efficiency_mpg` (Target)

#### Preparation Steps:
1. **Fill missing values with zeros**:
   ```python
   df = df.fillna(0)
   ```
2. **Train / Validation / Test split**:
   ```python
   df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
   df_train, df_val = train_test_split(
       df_full_train, test_size=0.25, random_state=1
   )
   ```
   - Full Train: 8,000 samples
   - Train (60%): 6,000 samples
   - Validation (20%): 2,000 samples
   - Test (20%): 2,000 samples

3. **Vectorization**:
   Transform dataframes to sparse matrices using `DictVectorizer(sparse=True)`:
   ```python
   dv = DictVectorizer(sparse=True)
   train_dicts = df_train.to_dict(orient='records')
   val_dicts = df_val.to_dict(orient='records')
   test_dicts = df_test.to_dict(orient='records')

   X_train = dv.fit_transform(train_dicts)
   X_val = dv.transform(val_dicts)
   X_test = dv.transform(test_dicts)
   ```
   Total one-hot encoded features: 16.

---

### Question 1: Decision Tree Regressor Splitting Feature

**Question**: Train a decision tree regressor to predict `fuel_efficiency_mpg` with `max_depth=1`. Which original feature is used for splitting the data?

- `'vehicle_weight'`
- **`'model_year'`** *(Selected Answer)*
- `'origin'`
- `'fuel_type'`

#### Calculation & Explanation:
```python
dt = DecisionTreeRegressor(max_depth=1)
dt.fit(X_train, y_train)

split_feature_idx = dt.tree_.feature[0]
split_feature_name = feature_names[split_feature_idx]
```
The root node splits on feature `model_year` (feature index corresponding to `model_year`).

**Answer**: **`'model_year'`**

---

### Question 2: Random Forest Regressor Validation RMSE

**Question**: Train a random forest regressor with:
- `n_estimators=10`
- `random_state=1`
- `n_jobs=-1`

What's the RMSE of this model on the validation data?

- 0.1837
- **1.837** *(Selected Answer)*
- 18.37
- 183.7

#### Calculation & Explanation:
```python
rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_val = rf.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(rmse)  # Output: 1.8332
```
The validation RMSE is `1.8332`, which corresponds to option **`1.837`**.

**Answer**: **`1.837`**

---

### Question 3: Tuning `n_estimators`

**Question**: Experiment with `n_estimators` in `[10, 50, 100, 150]` with `random_state=1`. Which value gives the lowest validation RMSE? (Consider 3 decimal places).

- 10
- 50
- 100
- **150** *(Selected Answer)*

#### Calculation & Explanation:
```python
for n in [10, 50, 100, 150]:
    rf = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"n_estimators={n} -> RMSE: {rmse:.4f}")
```

Results:
- `n_estimators = 10`  -> RMSE: `1.8332`
- `n_estimators = 50`  -> RMSE: `1.7740`
- `n_estimators = 100` -> RMSE: `1.7698` (rounds to `1.770`)
- `n_estimators = 150` -> RMSE: `1.7692` (rounds to `1.769`)

The validation RMSE steadily improves as `n_estimators` increases, reaching the lowest RMSE at `150`.

**Answer**: **`150`**

---

### Question 4: Tuning `max_depth`

**Question**: Select the best `max_depth` from `[10, 15, 20, 25]`:
- For each depth, evaluate `n_estimators` in `[10, 50, 100, 150]`
- Calculate the mean RMSE
- Fix `random_state=1`

What's the best `max_depth`, using the mean RMSE?

- **10** *(Selected Answer)*
- 15
- 20
- 25

#### Calculation & Explanation:
```python
for depth in [10, 15, 20, 25]:
    scores = []
    for n in [10, 50, 100, 150]:
        rf = RandomForestRegressor(n_estimators=n, max_depth=depth, random_state=1, n_jobs=-1)
        rf.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_val, rf.predict(X_val)))
        scores.append(rmse)
    print(f"max_depth={depth} -> Mean RMSE: {np.mean(scores):.4f}")
```

Results:
- `max_depth = 10` -> Mean RMSE: **`1.7544`** (individual: 1.7823, 1.7475, 1.7440, 1.7437)
- `max_depth = 15` -> Mean RMSE: `1.7842` (individual: 1.8314, 1.7709, 1.7679, 1.7668)
- `max_depth = 20` -> Mean RMSE: `1.7868` (individual: 1.8373, 1.7724, 1.7689, 1.7685)
- `max_depth = 25` -> Mean RMSE: `1.7863` (individual: 1.8369, 1.7721, 1.7683, 1.7679)

`max_depth=10` prevents overfitting best and achieves the lowest mean RMSE by a significant margin.

**Answer**: **`10`**

---

### Question 5: Feature Importance

**Question**: Train a Random Forest with `n_estimators=10`, `max_depth=20`, `random_state=1`, `n_jobs=-1`. What's the most important feature among these 4?

- **`vehicle_weight`** *(Selected Answer)*
- `horsepower`
- `acceleration`
- `engine_displacement`

#### Calculation & Explanation:
```python
rf = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
```

Feature importances of the 4 candidates:
1. **`vehicle_weight`**: **`0.203313`**
2. `horsepower`: `0.078271`
3. `engine_displacement`: `0.061281`
4. `acceleration`: `0.054998`

Among the given four features, `vehicle_weight` has by far the highest importance score (~0.203).

**Answer**: **`vehicle_weight`**

---

### Question 6: XGBoost Tuning (`eta=0.3` vs `eta=0.1`)

**Question**: Train an XGBoost model for 100 rounds with `eta=0.3` and `eta=0.1`:
Which `eta` leads to the best RMSE score on the validation dataset?

- 0.3
- **0.1** *(Selected Answer)*
- Both give equal value

#### Calculation & Explanation:
```python
xgb_params_03 = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

xgb_params_01 = {
    'eta': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}
```

Validation RMSE after 100 rounds:
- `eta = 0.3`: final validation RMSE = **`1.8265`** (lowest across rounds: `1.7321` at round 10)
- `eta = 0.1`: final validation RMSE = **`1.7248`** (lowest across rounds: `1.7129` at round 40)

`eta=0.1` achieves a substantially lower (better) validation RMSE (`1.7248` vs `1.8265`).

**Answer**: **`0.1`**

---

### Summary Table of Answers

| Question | Topic | Selected Answer |
| :--- | :--- | :--- |
| **Q1** | Splitting feature (Decision Tree `max_depth=1`) | **`'model_year'`** |
| **Q2** | Random Forest validation RMSE (`n=10`) | **`1.837`** |
| **Q3** | Best `n_estimators` | **`150`** |
| **Q4** | Best `max_depth` (mean RMSE) | **`10`** |
| **Q5** | Most important feature among candidates | **`vehicle_weight`** |
| **Q6** | Best learning rate `eta` in XGBoost | **`0.1`** |
