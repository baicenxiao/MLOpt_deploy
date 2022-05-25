import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from .ml.model import compute_model_metrics, train_model, inference


@pytest.fixture()
def dummy_predictions_actual():
    """ Generate synthetic predicions and actual values """
    actual = pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    prediction = pd.Series([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1])

    return actual, prediction


@pytest.fixture()
def dummy_model_data():
    """ Generate a random n-class classification problem."""
    X, y = make_classification()
    return X, y


def test_compute_model_metrics(dummy_predictions_actual):
    """ Tests to see if we get the correct metric values given
    a static prediction and actual array """

    actual, predic = dummy_predictions_actual
    prec, recall, fbeta = compute_model_metrics(actual, predic)
    assert round(prec, 2) == 0.57
    assert round(recall, 2) == 0.67
    assert round(fbeta, 2) == 0.62


def test_model(dummy_model_data):
    """Tests if the train model step works by passing in synthetic data and checking the outputs.
    We should see a reasonable AUC (>.60)
    """
    X, y = dummy_model_data

    train_X, test_X, train_y, test_y = train_test_split(X, y)
    model = train_model(train_X, train_y)
    preds = model.predict_proba(test_X)[:, 1]
    auc = roc_auc_score(test_y, preds)

    assert auc > 0.6


def test_inference(dummy_model_data):
    """ Tests that the inference step produces values between 0 and 1
    """
    X, y = dummy_model_data

    train_X, test_X, train_y, test_y = train_test_split(X, y)
    model = train_model(train_X, train_y)
    preds = inference(model, test_X)

    assert preds.max() <= 1 and preds.min() >= 0