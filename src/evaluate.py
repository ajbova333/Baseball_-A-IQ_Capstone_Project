import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred):
    mae= mean_absolute_error(y_true, y_pred)
    rmse= mean_squared_error(y_true, y_pred)** 0.5
    r2= r2_score(y_true, y_pred)
    return {'mae': mae, 'rmse': rmse, 'r2': r2}

def load_model_from_run(run_id):
    model_uri= f'runs:/{run_id}/model'
    return mlflow.sklearn.load_model(model_uri)

def prediction_interval(prediction, mae):
    """This is the range of uncertainty."""
    return (prediction- mae, prediction+ mae)

