# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 20:44:48 2024

@author: ahmed
"""

import numpy as np  # doğrusal cebir
import pandas as pd  # veri işleme, CSV dosyası girdi/çıktı (örneğin: pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, cohen_kappa_score, f1_score, confusion_matrix, roc_auc_score

import xgboost as xgb

# Read dataset
df = pd.read_csv("heart.csv")

features_to_be_scaled = [feature for feature in df.columns if len(df[feature].unique()) < 10]
features_to_be_scaled.remove('target')
print('No of features to be scaled: {}'.format(len(features_to_be_scaled)))

temp_df = pd.get_dummies(df, columns=features_to_be_scaled)
std = StandardScaler()
temp_df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']] = std.fit_transform(temp_df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])

y = temp_df.target
X = temp_df.drop(["target"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

params = {
    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
}

classifier = xgb.XGBClassifier()
random_search = RandomizedSearchCV(classifier, param_distributions=params, n_iter=5, scoring='roc_auc', n_jobs=-1, cv=5, verbose=3)

random_search.fit(X_train, y_train)
classifier = random_search.best_estimator_

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions
predictions = classifier.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
precision = precision_score(y_test, predictions)
cohen_kappa = cohen_kappa_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

# Confusion Matrix for calculating Sensitivity and Specificity
conf_matrix = confusion_matrix(y_test, predictions)
tn, fp, fn, tp = conf_matrix.ravel()

sensitivity = tp / (tp + fn)  # True Positive Rate
specificity = tn / (tn + fp)  # True Negative Rate

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_test, classifier.predict_proba(X_test)[:, 1])

# Print metrics
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Recall (Sensitivity): %.2f%%" % (recall * 100.0))
print("Precision: %.2f%%" % (precision * 100.0))
print("Cohen Kappa: %.2f" % cohen_kappa)
print("F1 Score: %.2f" % f1)
print("Sensitivity: %.2f%%" % (sensitivity * 100.0))
print("Specificity: %.2f%%" % (specificity * 100.0))
print("ROC AUC: %.2f" % roc_auc)
