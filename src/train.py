import yaml
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from src.preprocessing import preprocess, get_numeric_feature_cols, fit_scaler, apply_scaler
from src.evaluate import compute_metrics
from sklearn.model_selection import GridSearchCV

rate_stat_cols= ['AVG', 'OBP', 'SLG', 'BB%', 'K%', 'BABIP', 'PA/AB', 'Age']
target_col= 'AVG_next'
scaler_path= 'data/processed/scaler.joblib'

def load_config(path= 'configs/train_config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def chrono_split(df, test_fraction):
    """Sort by year, then take the last test_fraction of rows (count) as
    test set."""
    df_sorted= df.sort_values('yearID').reset_index(drop=True)
    split_idx= int(len(df_sorted) * (1 - test_fraction))
    train_df= df_sorted.iloc[:split_idx].copy()
    test_df= df_sorted.iloc[split_idx:].copy()
    return train_df, test_df

def get_feature_sets(df):
    """Return the full and rates_only feature column lists."""
    position_cols = [c for c in df.columns if c.startswith('Position_')]
    full_cols= get_numeric_feature_cols(df) + position_cols
    rates_only_cols= [c for c in rate_stat_cols if c in df.columns] + position_cols
    return {'full feature set': full_cols, 'Rates Only columns': rates_only_cols}

def prepare_data(config):
    panel = pd.read_csv(config['data']['data_path'])
    panel = preprocess(panel)

    train_df, test_df= chrono_split(panel, config['data']['test_fraction'])

    feature_sets= get_feature_sets(panel)
    scale_cols= get_numeric_feature_cols(panel)

    scaler = fit_scaler(train_df, scale_cols)
    train_df= apply_scaler(train_df, scaler, scale_cols)
    test_df= apply_scaler(test_df, scaler, scale_cols)

    joblib.dump(scaler, scaler_path)

    return train_df, test_df, feature_sets

model_registry= {
    'linear_regression': LinearRegression,
    'random_forest': RandomForestRegressor,
    'gradient_boosting': GradientBoostingRegressor
}

def train_and_log(run_name, model, X_train, y_train, X_test, y_test, params, feature_set_name, data_description):
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        y_pred= model.predict(X_test)

        metrics = compute_metrics(y_test, y_pred)

        mlflow.log_params(params)
        mlflow.log_param('feature_set', feature_set_name)
        mlflow.log_param('data_description', data_description)

        mlflow.log_metric('mae', metrics['mae'])
        mlflow.log_metric('rmse', metrics['rmse'])
        mlflow.log_metric('r2', metrics['r2'])

        mlflow.sklearn.log_model(model, name='model', skops_trusted_types=['sklearn.tree._tree.Tree'])

        return {**metrics, 'model': model}


def main():
    config = load_config()
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    train_df, test_df, feature_sets = prepare_data(config)

    y_train = train_df[target_col]
    y_test = test_df[target_col]

    data_description = f"lahman_panel_{train_df['yearID'].min()}-{test_df['yearID'].max()}_min_ab130"

    # Runs 1-3 are baseline models on the full feature set
    full_col = feature_sets['full feature set']
    X_train_full = train_df[full_col]
    X_test_full = test_df[full_col]

    baseline_results = {}
    for model_name in ['linear_regression', 'random_forest', 'gradient_boosting']:
        params = config['models'][model_name]['params']
        model = model_registry[model_name](**params)
        result = train_and_log(
            run_name=f'{model_name}_full',
            model=model,
            X_train=X_train_full, y_train=y_train,
            X_test=X_test_full, y_test=y_test,
            params=params,
            feature_set_name='full',
            data_description=data_description,
        )
        baseline_results[model_name] = result

    best_baseline_name = min(baseline_results, key=lambda name: baseline_results[name]['mae'])
    print(f"Best of the 3 Baseline models: {best_baseline_name} (MAE={baseline_results[best_baseline_name]['mae']:.4f})")

    # 4th run is the best baseline, rates only feature set
    rates_cols = feature_sets['Rates Only columns']
    X_train_rates = train_df[rates_cols]
    X_test_rates = test_df[rates_cols]

    rates_params = config['models'][best_baseline_name]['params']
    rates_model = model_registry[best_baseline_name](**rates_params)
    rates_result = train_and_log(
        run_name=f'{best_baseline_name}_rates_only',
        model=rates_model,
        X_train=X_train_rates, y_train=y_train,
        X_test=X_test_rates, y_test=y_test,
        params=rates_params,
        feature_set_name='rates_only',
        data_description=data_description,
    )

    # To determine the best model across the previous 4 runs
    all_results = {
        **{f'{name}_full': res for name, res in baseline_results.items()},
        f'{best_baseline_name}_rates_only': rates_result,
    }

    overall_winner_key = min(all_results, key=lambda k: all_results[k]['mae'])
    print(f"Overall Winner after 4 runs: {overall_winner_key} (MAE={all_results[overall_winner_key]['mae']:.4f})")

    if overall_winner_key.endswith('_rates_only'):
        winning_run = best_baseline_name
        winning_feature_set = 'rates_only'
        X_train_winner, X_test_winner = X_train_rates, X_test_rates
    else:
        winning_run = overall_winner_key.replace('_full', '')
        winning_feature_set = 'full'
        X_train_winner, X_test_winner = X_train_full, X_test_full

    # Final run: No. 5 GridSearchCV-tuned variant of the overall winner
    param_grid = config['tuning'][winning_run]
    grid_search = GridSearchCV(
        model_registry[winning_run](), param_grid, scoring='neg_mean_absolute_error', cv=5
    )

    grid_search.fit(X_train_winner, y_train)

    tuned_result = train_and_log(
        run_name=f'{winning_run}_tuned_{winning_feature_set}',
        model=grid_search.best_estimator_,
        X_train=X_train_winner, y_train=y_train,
        X_test=X_test_winner, y_test=y_test,
        params=grid_search.best_params_,
        feature_set_name=winning_feature_set,
        data_description=data_description,
    )
    print(f"Tuned Model MAE: {tuned_result['mae']:.4f}")


if __name__ == '__main__':
    main()
