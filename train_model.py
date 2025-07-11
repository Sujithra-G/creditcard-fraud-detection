# train_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import pickle

# Load dataset
df = pd.read_csv('creditcard.csv')

# Preprocess
df['Amount'] = StandardScaler().fit_transform(df[['Amount']])
df.drop(['Time'], axis=1, inplace=True)

# Balance dataset
fraud = df[df.Class == 1]
non_fraud = df[df.Class == 0].sample(n=len(fraud), random_state=42)
df_balanced = pd.concat([fraud, non_fraud]).sample(frac=1, random_state=42)

# Split
X = df_balanced.drop('Class', axis=1)
y = df_balanced['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Train
model = XGBClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
with open("xgb_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model saved as xgb_model.pkl")
