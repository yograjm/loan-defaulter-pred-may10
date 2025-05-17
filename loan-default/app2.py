# Creating REST endpoints using FastAPI

from fastapi import FastAPI

app = FastAPI()

# GET endpoint

@app.get("/")    # /
def func1():
    return {"message": "Hello, all!"}

# Get info from user, then respond
# 1. Query parameter

@app.get("/hi")      # /hi?name=Yograj
def func2(name):
    return {"message": "Hi " + name}


# 2. Path parameter

@app.get("/hello/{name}")    # /hello/Yograj
def func3(name):
    return {"message": "Hello " + name}



# POST endpoint

@app.post("/hey")
def func4(name):
    return {"message": "Hey " + name}


# cURL

# SmartBear --> Swagger Documentation v1   "/docs"

# --> OpenAPI Specification v3


# Request Body

from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    income: int


@app.post("/person")
def func5(person: Person):

    return {"message": "Your name is " + person.name + ". Your age is " + str(person.age) + ". Your income is " + str(person.income)}


# Prediction

class Features(BaseModel):
    person_age: int
    person_income: int
    person_home_ownership: str
    person_emp_length: int
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int


from app import predict_loan_status

@app.post("/predict")
def func6(f: Features):

    label = predict_loan_status(f.person_age, 
                        f.person_income, 
                        f.person_home_ownership, 
                        f.person_emp_length, 
                        f.loan_intent, 
                        f.loan_grade, 
                        f.loan_amnt, 
                        f.loan_int_rate, 
                        f.loan_percent_income, 
                        f.cb_person_default_on_file, 
                        f.cb_person_cred_hist_length)

    return {"prediction": label}



# Prometheus metrics objects
import prometheus_client as prom

acc_metric = prom.Gauge('loan_defaulter_pred_accuracy_score', "Accuracy score for Loan Defaulter prediction application")
f1_metric = prom.Gauge('loan_defaulter_pred_f1_score', "F1 score for Loan Defaulter prediction application")
precision_metric = prom.Gauge('loan_defaulter_pred_precision_score', "Precision score for Loan Defaulter prediction application")
recall_metric = prom.Gauge('loan_defaulter_pred_recall_score', "Recall score for Loan Defaulter prediction application")


from custom_utils import loan_grade_mapping, home_ownership_mapping, default_on_file_mapping
from predict import loan_intent_encoder

def update_metrics():
    # Get test data
    # from data_preprocessing import X_test, y_test
    # from sklearn.model_selection import train_test_split

    # _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=100)

    import pandas as pd
    import psycopg2
    from psycopg2 import sql

    # Database connection parameters
    db_params = {
        'dbname': 'storedb',
        'user': 'postgres',
        'password': 'mypassword',
        'host': '3.111.57.45',  # EC2 public IP # or your database host
        'port': '5432'  #'5432'        # default PostgreSQL port
    }

    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()

        # Read existing data from the table
        query = "SELECT * FROM loans_data;"  # SQL query to select all data from the table
        cursor.execute(query)

        # Fetch all results
        rows = cursor.fetchall()

        # Get column names from the cursor
        column_names = [desc[0] for desc in cursor.description]

        # Create a DataFrame from the fetched data
        df = pd.DataFrame(rows, columns=column_names)
        print(f"Existing rows in db: {len(df)}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Close the cursor and connection
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    
    # Handle missing values
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].mean())
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].mean())

    # Handle categorical columns
    df['person_home_ownership'] = df['person_home_ownership'].map(home_ownership_mapping)
    df['loan_grade'] = df['loan_grade'].map(loan_grade_mapping)
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map(default_on_file_mapping)

    df['loan_intent'] = loan_intent_encoder.transform(df['loan_intent'])

    # Fetures
    X_test = df.drop('loan_status', axis=1)
    # Target
    y_test = df['loan_status']


    # Calc. metrics - F1
    from predict import rf_model
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    y_pred = rf_model.predict(X_test)
    acc = round(accuracy_score(y_pred, y_test), 3)
    f1 = round(f1_score(y_pred, y_test), 3)
    prec = round(precision_score(y_pred, y_test), 3)
    recall = round(recall_score(y_pred, y_test), 3)

    # return it in a prometheus supported format
    acc_metric.set(acc)
    f1_metric.set(f1)
    precision_metric.set(prec)
    recall_metric.set(recall)



from fastapi import Response

@app.get("/metrics")
def get_metrics():
    update_metrics()
    return Response(media_type="text/plain", content= prom.generate_latest())



import uvicorn

uvicorn.run(app, host="0.0.0.0", port=8000)

