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



import uvicorn

uvicorn.run(app, host="0.0.0.0", port=8000)

