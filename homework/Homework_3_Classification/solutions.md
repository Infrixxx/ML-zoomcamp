# Machine Learning Zoomcamp 2026 — Homework 3 Solutions
## Machine Learning for Classification

This document provides the complete, reproducible solutions and explanations for **Homework 3: Machine Learning for Classification** from DataTalksClub's Machine Learning Zoomcamp (2026 cohort).

---

### Dataset Overview & Preparation

* **Dataset source**: `https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv`
* **Target variable**: `converted` (binary classification: `0` or `1`)
* **Categorical features**: `['lead_source', 'industry', 'employment_status', 'location']`
* **Numerical features**: `['annual_income', 'number_of_courses_viewed', 'interaction_count', 'lead_score']`

#### Missing Values Handling:
* Missing values in categorical features are replaced with `'NA'`.
* Missing values in numerical features are replaced with `0.0`.

---

### Question 1: Mode for Industry

**Question**: What is the most frequent observation (mode) for the column `industry`?

- `NA`
- **`technology`**  *(Selected Answer)*
- `healthcare`
- `retail`

#### Calculation & Explanation:
Counting occurrences in the `industry` column:
- `technology`: 1,173
- `retail`: 925
- `healthcare`: 840
- `finance`: 720
- `education`: 639
- `manufacturing`: 463
- Missing / `NA`: 240

The most frequent category (mode) is **`technology`** with 1,173 records.

---

### Question 2: Biggest Correlation

**Question**: Create the correlation matrix for the numerical features of your dataset. What are the two features that have the biggest correlation?

- **`interaction_count` and `lead_score`**  *(Selected Answer)*
- `number_of_courses_viewed` and `lead_score`
- `number_of_courses_viewed` and `interaction_count`
- `annual_income` and `interaction_count`

#### Calculation & Explanation:
Pearson correlation coefficients for the candidate pairs:
- `interaction_count` and `lead_score`: **`0.91575`** (or `0.92871` without imputing)
- `number_of_courses_viewed` and `lead_score`: `0.75720` (or `0.76935`)
- `number_of_courses_viewed` and `interaction_count`: `0.72161`
- `annual_income` and `interaction_count`: `0.12284` (or `0.10637`)

The pair with the highest correlation is **`interaction_count` and `lead_score`** (~0.92).

---

### Data Splitting

Data split performed according to the exact instructions:
```python
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train['converted'].values
y_val = df_val['converted'].values
y_test = df_test['converted'].values

del df_train['converted']
del df_val['converted']
del df_test['converted']
```
Sizes:
- Full dataset: 5,000 samples
- Train (60%): 3,000 samples
- Validation (20%): 1,000 samples
- Test (20%): 1,000 samples

---

### Question 3: Mutual Information

**Question**: Calculate the mutual information score between `converted` and other categorical variables in the dataset. Use the training set only. Round the scores to 2 decimals using `round(score, 2)`. Which of these variables has the biggest mutual information score?

- `industry`
- `location`
- **`lead_source`**  *(Selected Answer)*
- `employment_status`

#### Calculation & Explanation:
Using `sklearn.metrics.mutual_info_score` on the training set:
- `lead_source`: raw = `0.034346`, `round(score, 2)` = **`0.03`**
- `employment_status`: raw = `0.018966`, `round(score, 2)` = `0.02`
- `industry`: raw = `0.001995`, `round(score, 2)` = `0.00`
- `location`: raw = `0.001043`, `round(score, 2)` = `0.00`

The categorical variable with the highest mutual information score is **`lead_source`**.

---

### Question 4: Logistic Regression Accuracy

**Question**: Train a logistic regression with one-hot encoding on training set (`solver='liblinear', C=1.0, max_iter=1000, random_state=42`). Calculate the accuracy on the validation dataset and round it to 2 decimal digits. What accuracy did you get?

- `0.55`
- **`0.65`**  *(Selected Answer)*
- `0.75`
- `0.85`

#### Calculation & Explanation:
Using `DictVectorizer(sparse=False)`:
```python
dv = DictVectorizer(sparse=False)
train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
```
- Raw validation accuracy: **`0.645000`**
- `round(0.645, 2)` = **`0.65`**

---

### Question 5: Feature Elimination

**Question**: Let's find the least useful feature using the feature elimination technique. Train a model without each feature and calculate the difference between the original accuracy and the accuracy without the feature. Which of following feature has the smallest difference?

- `'lead_source'`
- **`'number_of_courses_viewed'`**  *(Selected Answer)*
- `'interaction_count'`

#### Calculation & Explanation:
- Original model validation accuracy: `0.645000`

Exclusion results:
- Excluding `interaction_count`: accuracy = `0.601000`, difference = `0.645 - 0.601` = `0.044000`
- Excluding `lead_source`: accuracy = `0.642000`, difference = `0.645 - 0.642` = `0.003000`
- Excluding `number_of_courses_viewed`: accuracy = `0.643000`, difference = `0.645 - 0.643` = **`0.002000`**

Among the candidate options, **`number_of_courses_viewed`** has the smallest difference (`0.002000`).

---

### Question 6: Parameter Tuning

**Question**: Try values of parameter `C`: `[0.000001, 0.00001, 0.0001, 0.001]`. Calculate the accuracy on the validation dataset and round it to 3 decimal digits. Which of these `C` leads to the best accuracy on the validation set?

- `0.000001`
- `0.00001`
- `0.0001`
- **`0.001`**  *(Selected Answer)*

#### Calculation & Explanation:
Tuning parameter `C` using `LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=42)`:
- `C = 0.000001`: accuracy = `0.598000`, round(3) = `0.598`
- `C = 0.00001`: accuracy = `0.598000`, round(3) = `0.598`
- `C = 0.0001`: accuracy = `0.613000`, round(3) = `0.613`
- `C = 0.001`: accuracy = **`0.645000`**, round(3) = **`0.645`**

The best validation accuracy is `0.645`, achieved at **`0.001`**.

---

### Reflection Question (from `homework.yaml`)

**Prompt**: *Describe one practical idea you took from this module.*

**Sample Answer**:
> In this module, a key practical takeaway was understanding how to quantify the relationship between categorical features and binary targets using **Mutual Information**. Unlike correlation coefficients which only measure linear relationships between numeric variables, Mutual Information allows us to evaluate the predictive power of categorical variables regardless of their relationship structure. Additionally, the **feature elimination** technique demonstrated how to rigorously assess the importance of individual features by measuring the change in model evaluation metrics when each feature is withheld.

---

### Submission Summary Table

| # | Question | Answer |
|---|---|---|
| **1** | Mode for `industry` | **`technology`** |
| **2** | Two features with biggest correlation | **`interaction_count` and `lead_score`** |
| **3** | Biggest mutual information score | **`lead_source`** |
| **4** | Logistic regression validation accuracy | **`0.65`** |
| **5** | Feature with smallest accuracy difference | **`number_of_courses_viewed`** |
| **6** | Best regularisation parameter `C` | **`0.001`** |
