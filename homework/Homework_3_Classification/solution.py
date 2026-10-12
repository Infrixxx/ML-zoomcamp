"""
Machine Learning Zoomcamp 2026 - Homework 3: Classification
Complete Solutions Script
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score, accuracy_score

def main():
    print("=" * 60)
    print("Machine Learning Zoomcamp 2026 - Homework 3 Solutions")
    print("=" * 60)

    # -------------------------------------------------------------
    # 0. Load Dataset
    # -------------------------------------------------------------
    url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv"
    try:
        df = pd.read_csv('course_lead_scoring_2026.csv')
    except Exception:
        df = pd.read_csv(url)

    print(f"Dataset shape: {df.shape}")

    categorical = ['lead_source', 'industry', 'employment_status', 'location']
    numerical = ['annual_income', 'number_of_courses_viewed', 'interaction_count', 'lead_score']

    # -------------------------------------------------------------
    # Question 1: Mode for industry
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("QUESTION 1: Most frequent observation (mode) for 'industry'")
    print("=" * 60)
    industry_mode_raw = df['industry'].mode()[0]
    print(f"Value counts of 'industry' (before preparation):\n{df['industry'].value_counts(dropna=False)}")
    print(f"\nMode (most frequent): {industry_mode_raw}")

    # Data preparation as specified in homework
    # - For categorical features, replace missing values with 'NA'
    # - For numerical features, replace missing values with 0.0
    for col in categorical:
        df[col] = df[col].fillna('NA')

    for col in numerical:
        df[col] = df[col].fillna(0.0)

    industry_mode_prep = df['industry'].mode()[0]
    print(f"Mode (after preparation): {industry_mode_prep}")
    print(f"--> Question 1 Answer: {industry_mode_prep}")

    # -------------------------------------------------------------
    # Question 2: Biggest correlation
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("QUESTION 2: Biggest correlation between numerical features")
    print("=" * 60)
    corr_matrix = df[numerical].corr()
    print("Correlation matrix:")
    print(corr_matrix)

    q2_pairs = [
        ('interaction_count', 'lead_score'),
        ('number_of_courses_viewed', 'lead_score'),
        ('number_of_courses_viewed', 'interaction_count'),
        ('annual_income', 'interaction_count')
    ]

    print("\nPair correlations:")
    corr_results = {}
    for f1, f2 in q2_pairs:
        corr_val = corr_matrix.loc[f1, f2]
        corr_results[f"{f1} and {f2}"] = corr_val
        print(f"  {f1} and {f2}: {corr_val:.5f}")

    best_pair = max(corr_results, key=corr_results.get)
    print(f"--> Question 2 Answer: {best_pair}")

    # -------------------------------------------------------------
    # Split the data
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Splitting the data (60% train / 20% val / 20% test)")
    print("=" * 60)
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

    print(f"Train size: {len(df_train)}, Val size: {len(df_val)}, Test size: {len(df_test)}")

    # -------------------------------------------------------------
    # Question 3: Biggest Mutual Information
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("QUESTION 3: Mutual information score with 'converted'")
    print("=" * 60)
    mi_scores = {}
    for col in categorical:
        mi = mutual_info_score(df_train[col], y_train)
        mi_scores[col] = mi
        print(f"  {col}: raw = {mi:.6f}, round(2) = {round(mi, 2)}")

    best_mi_var = max(mi_scores, key=mi_scores.get)
    print(f"--> Question 3 Answer: {best_mi_var}")

    # -------------------------------------------------------------
    # Question 4: Logistic Regression Accuracy
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("QUESTION 4: Logistic Regression Accuracy")
    print("=" * 60)
    dv = DictVectorizer(sparse=False)
    train_dict = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)

    val_dict = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dict)

    model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc_orig = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: raw = {acc_orig:.6f}, round(2) = {round(acc_orig, 2)}")
    print(f"--> Question 4 Answer: {round(acc_orig, 2)}")

    # -------------------------------------------------------------
    # Question 5: Feature Elimination
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("QUESTION 5: Feature elimination (smallest difference)")
    print("=" * 60)
    all_features = categorical + numerical
    diffs = {}
    q5_candidates = ['lead_source', 'number_of_courses_viewed', 'interaction_count']

    for feature in all_features:
        subset = [f for f in all_features if f != feature]
        dv_sub = DictVectorizer(sparse=False)
        X_train_sub = dv_sub.fit_transform(df_train[subset].to_dict(orient='records'))
        X_val_sub = dv_sub.transform(df_val[subset].to_dict(orient='records'))

        model_sub = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
        model_sub.fit(X_train_sub, y_train)
        acc_sub = accuracy_score(y_val, model_sub.predict(X_val_sub))
        diff = acc_orig - acc_sub
        diffs[feature] = diff
        print(f"  Excluding '{feature}': accuracy = {acc_sub:.6f}, diff = {diff:.6f}")

    print("\nCandidate features comparison:")
    for c in q5_candidates:
        print(f"  '{c}': difference = {diffs[c]:.6f}")

    best_q5 = min(q5_candidates, key=lambda c: diffs[c])
    print(f"--> Question 5 Answer: {best_q5}")

    # -------------------------------------------------------------
    # Question 6: Regularized Logistic Regression (Tuning C)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("QUESTION 6: Parameter tuning (best C)")
    print("=" * 60)
    c_values = [0.000001, 0.00001, 0.0001, 0.001]
    c_accs = {}
    for c_val in c_values:
        m_c = LogisticRegression(solver='liblinear', C=c_val, max_iter=1000, random_state=42)
        m_c.fit(X_train, y_train)
        acc_c = accuracy_score(y_val, m_c.predict(X_val))
        rounded_acc = round(acc_c, 3)
        c_accs[c_val] = (acc_c, rounded_acc)
        print(f"  C = {c_val}: accuracy = {acc_c:.6f}, round(3) = {rounded_acc}")

    # Find C with highest rounded accuracy (if tie, smallest C)
    max_acc = max(c_accs[c][1] for c in c_values)
    best_c_candidates = [c for c in c_values if c_accs[c][1] == max_acc]
    best_c = min(best_c_candidates)
    print(f"\nBest accuracy is {max_acc} at C = {best_c}")
    print(f"--> Question 6 Answer: {best_c}")

    print("\n" + "=" * 60)
    print("SUMMARY OF ALL ANSWERS")
    print("=" * 60)
    print(f"Question 1: {industry_mode_prep}")
    print(f"Question 2: {best_pair}")
    print(f"Question 3: {best_mi_var}")
    print(f"Question 4: {round(acc_orig, 2)}")
    print(f"Question 5: {best_q5}")
    print(f"Question 6: {best_c}")
    print("=" * 60)

if __name__ == '__main__':
    main()
