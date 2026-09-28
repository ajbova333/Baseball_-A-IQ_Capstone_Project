import pandas as pd
import joblib
import pytest

from src.app import normalize_stats, build_feature_row, load_app_config, get_client, extract_features
from src.player_lookup import load_people, build_name_lookup, find_player
from src.preprocessing import preprocess
from src.train import get_feature_sets


@pytest.fixture(scope='module')
def resources():
    config = load_app_config()
    scaler = joblib.load(config['data']['scaler_path'])
    panel = pd.read_csv(config['data']['panel_path'])
    processed_panel = preprocess(panel)
    feature_sets = get_feature_sets(processed_panel)
    feature_cols = feature_sets['full feature set']

    people = load_people(config['data']['people_path'])
    name_lookup = build_name_lookup(people)

    return {
        'panel': panel,
        'processed_panel': processed_panel,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'name_lookup': name_lookup,
    }


def test_normalize_stats_conv_to_dec():
    stats = {'K%': 15, 'AGE': 32}
    result = normalize_stats(stats)
    assert result['K%'] == 0.15


def test_normalize_match_stats_to_position():
    stats = {'Position': 'left fielder'}
    result = normalize_stats(stats)
    assert result['Position'] == 'LF'


def test_find_exact_match(resources):
    match = find_player('mike trout', resources['name_lookup'])
    assert match['status'] == 'found'
    assert match['player_id'] == 'troutmi01'


def test_find_player_close_mispelling(resources):
    match = find_player('mike trot', resources['name_lookup'])
    assert match['status'] == 'suggestion'
    assert match['suggested_name'] == 'mike trout'


def test_find_player_not_found(resources):
    match = find_player('zzqxnotrealplayer', resources['name_lookup'])
    assert match['status'] == 'not_found'


def test_build_feature_row_handles_sparse_input(resources):
    stats = {'Age': 32, 'Position': 'LF', 'K%': 0.15}
    x, caveats = build_feature_row(stats, resources)

    assert x.shape[0] == 1
    assert list(x.columns) == resources['feature_cols']
    assert 'AVG' in caveats
    assert 'Age' not in caveats

def test_extract_features_hypothetical():
    config= load_app_config()
    client= get_client(config)

    result= extract_features(
        client, config, "hypothetical",
        "I'm a 45 year old first baseman with a .310 batting average and 500 at bats."
    )

    result = normalize_stats(result)

    assert result['Age'] == 45
    assert result['AB'] == 500
    assert abs(result['AVG'] - 0.310) < 0.01
    assert result['Position'] == '1B'


def test_extract_features_real_player_name_from_natural_language():
    config = load_app_config()
    client = get_client(config)

    result = extract_features(client, config, 'real_player', "What was Mike Trout's batting average last year?")

    assert result['player_name'] is not None
    assert 'trout' in result['player_name'].lower()
