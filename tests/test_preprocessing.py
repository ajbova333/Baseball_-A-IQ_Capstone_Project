import pandas as pd
import pytest

from src.preprocessing import(
    fill_missing_pos,
    encode_position,
    preprocess,
    get_numeric_feature_cols,
    fit_scaler,
    apply_scaler,
    pos_cat,
)

@pytest.fixture
def sample_panel():
    return pd.DataFrame({
        'playerID': ['a01', 'b01', 'c01'],
        'yearID': [2018, 2019, 2020],
        'AB': [500, 450, 600],
        'H': [150, 120, 180],
        'AVG': [0.300, 0.267, 0.405],
        'Position': ['SS', None, 'OF'],
        'AVG_next': [0.280, 0.290, 0.310]
    })

def test_fill_missing_pos(sample_panel):
    result= fill_missing_pos(sample_panel)
    assert result['Position'].isna().sum() == 0
    assert result.loc[1, 'Position'] =='Unknown'

def test_encode_position(sample_panel):
    filled= fill_missing_pos(sample_panel)
    encoded= encode_position(filled)
    for cat in pos_cat:
        assert f'Position_{cat}' in encoded.columns
    assert encoded.loc[0, 'Position_SS'] == 1
    assert encoded.loc[1, 'Position_Unknown']== 1

def test_scaling_production(sample_panel):
    processed= preprocess(sample_panel)
    feature_cols= get_numeric_feature_cols(processed)
    scaler= fit_scaler(processed, feature_cols)
    scaled= apply_scaler(processed, scaler, feature_cols)
    means= scaled[feature_cols].mean()
    assert (means.abs() < 1e-8).all()

def test_preprocess_does_not_modify_original(sample_panel):
    original= sample_panel.copy(deep=True)
    _= preprocess(sample_panel)
    pd.testing.assert_frame_equal(sample_panel, original)