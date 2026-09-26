import pandas as pd
from sklearn.preprocessing import StandardScaler

pos_cat= ['P', 'C', '1B', '2B', '3B', 'SS', 'LF', 'RF', 'CF', 'OF', 'DH', 'Unknown']


def fill_missing_pos(df):
    """Fill in missing positions with new 'Unknown' position
    rather than dropping the rows."""
    df= df.copy()
    df['Position']= df['Position'].fillna('Unknown')
    return df

def encode_position(df):
    """one-Hot Encode Position using a fixed category list, 
    so train/test always get the same columns."""
    df= df.copy()
    df['Position']= pd.Categorical(df['Position'], categories= pos_cat)
    dummies= pd.get_dummies(df['Position'], prefix='Position')
    df= pd.concat([df.drop(columns=['Position']), dummies], axis=1)
    return df

non_scaled_cols= ['playerID', 'yearID', 'AVG_next']

def get_numeric_feature_cols(df):
    """Every Column that should be scaled: numeric features, exluding IDs,
    target and one-hot columns."""
    return[
        col for col in df.columns
        if col not in non_scaled_cols and not col.startswith('Position_')
    ]

def fit_scaler(train_df, feature_cols):
    """Fit a StandardScaler on the training split only."""
    scaler= StandardScaler()
    scaler.fit(train_df[feature_cols])
    return scaler

def apply_scaler(df, scaler, feature_cols):
    """Apply an already fit scaler to any split"""
    df= df.copy()
    df[feature_cols]= scaler.transform(df[feature_cols])
    return df

def preprocess(df):
    """Apply the safe-before-split cleaning/encoding steps: fill missing
    position, then one-hot encode it."""
    df= fill_missing_pos(df)
    df=encode_position(df)
    return df
