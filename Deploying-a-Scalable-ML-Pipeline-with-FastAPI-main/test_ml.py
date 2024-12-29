import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data
import pandas as pd
from sklearn.model_selection import train_test_split

# Sample data to use in tests
sample_data = pd.DataFrame({
    "age": [39, 50, 38],
    "workclass": ["State-gov", "Self-emp-not-inc", "Private"],
    "education": ["Bachelors", "HS-grad", "Masters"],
    "marital-status": ["Never-married", "Married-civ-spouse", "Divorced"],
    "occupation": ["Adm-clerical", "Exec-managerial", "Prof-specialty"],
    "relationship": ["Not-in-family", "Husband", "Unmarried"],
    "race": ["White", "White", "Black"],
    "sex": ["Male", "Male", "Female"],
    "hours-per-week": [40, 13, 40],
    "native-country": ["United-States", "United-States", "United-States"],
    "salary": ["<=50K", ">50K", ">50K"],
})

# Process the sample data
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

train, test = train_test_split(sample_data, test_size=0.33, random_state=42)
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Test 1: Check if train_model returns the expected model type
def test_train_model_type():
    """
    Test that train_model returns a RandomForestClassifier.
    """
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier"
    print("test_ml.py::test_train_model_type PASSED")

# Test 2: Verify inference outputs the correct shape
def test_inference_output_shape():
    """
    Test that inference outputs predictions with the correct shape.
    """
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert preds.shape[0] == X_test.shape[0], "Inference output shape is incorrect"
    print("test_ml.py::test_inference_output_shape PASSED")

# Test 3: Verify compute_model_metrics produces expected metrics
def test_compute_model_metrics():
    """
    Test compute_model_metrics produces the expected precision, recall, and F1.
    """
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, preds)

    assert 0 <= precision <= 1, "Precision is out of bounds"
    assert 0 <= recall <= 1, "Recall is out of bounds"
    assert 0 <= f1 <= 1, "F1 score is out of bounds"
    print("test_ml.py::test_compute_model_metrics PASSED")