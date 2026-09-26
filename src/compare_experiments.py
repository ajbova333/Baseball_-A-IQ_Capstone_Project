import mlflow
from src.train import load_config


def get_all_runs(config):
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    return mlflow.search_runs(experiment_names=[config['mlflow']['experiment_name']])

def summarize_runs(runs):
    cols = ['tags.mlflow.runName', 'metrics.mae', 'metrics.rmse', 'metrics.r2', 'params.feature_set']
    cols = [c for c in cols if c in runs.columns]
    return runs[cols].sort_values('metrics.mae')


def get_best_run(runs):
    """Lower MAE is better."""
    return runs.sort_values('metrics.mae', ascending=True).iloc[0]


def main():
    config = load_config()
    runs = get_all_runs(config)

    print(f"Found {len(runs)} runs in experiment '{config['mlflow']['experiment_name']}'\n")
    print(summarize_runs(runs).to_string(index=False))

    best_run= get_best_run(runs)
    print(f'\nBest run: {best_run['tags.mlflow.runName']}')
    print(f' MAE: {best_run['metrics.mae']:.4f}')
    print(f' RMSE:{best_run['metrics.rmse']: .4f}')
    print(f' R2: {best_run['metrics.r2']: .4f}')
    print(f' Run ID: {best_run['run_id']}')

if __name__== '__main__':
    main()
    