import numpy as np
import pytest

from src.compare_experiments import get_best_run, get_all_runs
from src.evaluate import load_model_from_run, compute_metrics
from src.train import load_config, prepare_data, target_col

@pytest.fixture(scope='module')
def best_model_and_data():
    config= load_config()
    train_df, test_df, feature_sets= prepare_data(config)

    runs= get_all_runs(config)
    best_run= get_best_run(runs)
    model= load_model_from_run(best_run['run_id'])

    feature_set_key= 'full feature set' if best_run['params.feature_set']== 'full' else 'Rates Only columns'
    feature_cols= feature_sets[feature_set_key]

    x_test= test_df[feature_cols]
    y_test= test_df[target_col]

    return model, x_test, y_test

def test_prediction_shape_and_type(best_model_and_data):
    model, x_test, y_test= best_model_and_data
    preds= model.predict(x_test)

    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == len(x_test)
    assert preds.dtype.kind == 'f'

def test_predictions_plausible_ba(best_model_and_data):
    model, x_test, y_test= best_model_and_data
    preds= model.predict(x_test)

    assert preds.min() >-0.05
    assert preds.max() < 0.6

def test_model_meets_min(best_model_and_data):
    model, x_test, y_test= best_model_and_data
    preds = model.predict(x_test)
    metrics= compute_metrics(y_test, preds)

    assert metrics['mae'] < 0.05