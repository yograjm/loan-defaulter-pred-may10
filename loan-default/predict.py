
import pandas as pd
from data_preprocessing import loan_grade_mapping, home_ownership_mapping, default_on_file_mapping

# To load model
import joblib
rf_model = joblib.load("trained_model/rf_model_loan_default_pred.pkl")
loan_intent_encoder = joblib.load("trained_model/loan_intent_encoder.pkl")

# model.predict(sample_input_df)

# Inference
sample_input = {'person_age': 22,
                'person_income': 50000,
                'person_home_ownership': home_ownership_mapping['RENT'],
                'person_emp_length': 6.0,
                'loan_intent': loan_intent_encoder.transform(['DEBTCONSOLIDATION'])[0],
                'loan_grade': loan_grade_mapping['B'],
                'loan_amnt': 6000,
                'loan_int_rate': 11.89,
                'loan_percent_income': 0.12,
                'cb_person_default_on_file': default_on_file_mapping['N'],
                'cb_person_cred_hist_length': 2}

sample_input_df = pd.DataFrame(sample_input, index=[0])


def make_prediction(sample_input_df):
    prediction = rf_model.predict(sample_input_df)
    label = "Likely to default" if prediction[0] == 1 else "Less likely to default"
    return label


# python predict.py   __name__ = "__main__"
# python app.py       __name__ = "predict"

if __name__ == "__main__":
    pred = make_prediction(sample_input_df)
    print(pred)
