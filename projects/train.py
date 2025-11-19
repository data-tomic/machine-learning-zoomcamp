#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Parameters
C_PARAM = 1.0
OUTPUT_FILE = 'model.bin'
DATA_FILE = 'anemia.csv'

print(f"Loading data from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

# Data Splitting
# We use 80% for training (Full Train) and 20% for final Testing
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train.Result.values
y_test = df_test.Result.values

del df_full_train['Result']
del df_test['Result']

# Training
print("Training the final model...")

dv = DictVectorizer(sparse=False)
train_dicts = df_full_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

# Logistic Regression was chosen as the best model
model = LogisticRegression(solver='liblinear', C=C_PARAM, random_state=1)
model.fit(X_train, y_full_train)

# Validation on Test Set
test_dicts = df_test.to_dict(orient='records')
X_test = dv.transform(test_dicts)

y_pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)

print(f"Final ROC-AUC on Test Set: {auc:.4f}")

# Exporting
print(f"Saving the model to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print("Done!")
