import numpy as np
import pandas as pd

# df = pd.read_csv("dataset/credit_risk_dataset.csv")


############ DVC related code START ###################

import os


import dvc.api
import pandas as pd

# Repo:  'https://<github-username>:<github-token>@github.com/yograjm/credit-risk-data'

repo_name = "loan-data-repo"     # Change as per your GitHub repository name
repo_url = 'https://' + os.environ['GH_USERNAME'] + ':' + os.environ['GH_ACCESS_TOKEN'] + '@github.com/' + os.environ['GH_USERNAME'] + '/' + repo_name


data_revision = 'v1.1'

# Configurations to access remote storage
remote_config = {
    'access_key_id': os.environ["AWS_ACCESS_KEY_ID"],
    'secret_access_key': os.environ["AWS_SECRET_ACCESS_KEY"],
}

with dvc.api.open('data/credit_risk_dataset.csv', repo=repo_url, rev=data_revision, remote_config=remote_config) as file:
    df = pd.read_csv(file)

#df.tail()

############ DVC related code END ###################


# Handle missing values
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].mean())
df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].mean())


# Handle categorical columns
from custom_utils import loan_grade_mapping, home_ownership_mapping, default_on_file_mapping

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

