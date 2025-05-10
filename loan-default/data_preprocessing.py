import numpy as np
import pandas as pd

df = pd.read_csv("dataset/credit_risk_dataset.csv")

# Handle missing values
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].mean())
df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].mean())


# Handle categorical columns
home_ownership_mapping = {'MORTGAGE': 0, 'RENT': 1, 'OWN': 2, 'OTHER': 3}
loan_grade_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
default_on_file_mapping = {'N': 0, 'Y': 1}

df['person_home_ownership'] = df['person_home_ownership'].map(home_ownership_mapping)
df['loan_grade'] = df['loan_grade'].map(loan_grade_mapping)
df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map(default_on_file_mapping)

# Label Encoder
from sklearn.preprocessing import LabelEncoder

loan_intent_encoder = LabelEncoder()

loan_intent_encoder.fit(df['loan_intent'])

df['loan_intent'] = loan_intent_encoder.transform(df['loan_intent'])


# Train-test split
from sklearn.model_selection import train_test_split

X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

