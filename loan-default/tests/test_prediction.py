
import sys
from pathlib import Path
filepath = Path(__file__)
sys.path.append(str(filepath.parents[1]))


from data_preprocessing import X_test, y_test
from predict import rf_model

from sklearn.metrics import accuracy_score

def test_model_accuracy():
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    assert acc > 0.8, "Model accuracy not greater than 0.8"


from predict import make_prediction, sample_input_df

def test_make_prediction():
    pred = make_prediction(sample_input_df)
    assert pred in ["Likely to default",  "Less likely to default"], "Mismatch in prediction string"

