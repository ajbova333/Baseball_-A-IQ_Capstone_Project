import pandas as pd

data_dir= 'data/raw'

data_path= "data/processed/panel.csv"

start_year= 1962
min_ab= 130

batting_count_columns= [
    "G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS",
    "BB", "SO", "IBB", "HBP", "SH", "SF", "GIDP",
]


def load_raw_data(data_dir=data_dir):
    batting= pd.read_csv(f'{data_dir}/Batting.csv')
    people = pd.read_csv(f'{data_dir}/People.csv')
    fielding= pd.read_csv(f'{data_dir}/Fielding.csv')
    return batting, people, fielding

def aggregate_batting_stints(batting):
    "Control Mid-Season trades by combining(stints) into one row per year"
    return(
        batting.groupby(["playerID", "yearID"], as_index=False)[batting_count_columns].sum()
    )

def aggregate_fielding_stints(fielding):
    "Sum up all games at each position in each stint before assigning a primary position."
    return(
        fielding.groupby(["playerID", "yearID", "POS"], as_index=False)['G'].sum()
    )

def player_age(batting_agg, people):
    "Simple computation to find player age per stat year."
    merged= batting_agg.merge(
        people[["playerID", "birthYear"]], on="playerID", how="left"
    )
    merged["Age"]= merged["yearID"]- merged["birthYear"]
    return merged.drop(columns=["birthYear"])

def get_primary_pos(fielding_agg):
    "To pick the position for each player that had most time played."
    idx= fielding_agg.groupby(["playerID", "yearID"])["G"].idxmax()
    primary= fielding_agg.loc[idx, ["playerID", "yearID", "POS"]]
    return primary.rename(columns={"POS": "Position"})

def compute_rate_stats(df):
    "Adding AVG, OBP, SLG, BB%, K%, BABIP, PA, PA/AB to a players season row"
    df= df.copy()

    df["PA"]= df["AB"] + df["BB"] + df["HBP"] + df["SH"] + df["SF"]
    df["AVG"]= df["H"] / df["AB"]
    df["OBP"]= (df["H"] + df["BB"] + df["HBP"])/ (df["AB"] + df["BB"] + df["HBP"] + df["SF"])

    singles = df["H"] - df["2B"] - df["3B"] - df["HR"]
    total_bases = singles + (df["2B"] * 2) + (df["3B"] * 3) + (df["HR"] * 4)

    df['SLG']= total_bases / df['AB']

    df['BB%']= df['BB'] / df['PA']
    df['K%']= df['SO']/ df['PA']
    df['BABIP']= (df["H"] - df["HR"]) / (df["AB"] - df["SO"] - df["HR"] + df["SF"])
    df["PA/AB"] = df["PA"] / df["AB"]

    return df


def filter_qualified(df, min_ab= min_ab, start_year= start_year):
    "Filtering out seasons prior to the common era of baseball while also removing small at bat totals that can skew the dataset"
    return df[(df["yearID"] >= start_year) & (df["AB"] >= min_ab)].copy()


def build_lag_panel(qualified):
    """Pair year N's features with Year N+1 AVG. inner Join a mair only survives if BOTH years are in 'qualified' since
    'qualified is already has a AB/Year filter prior to this running."""

    targets= qualified[['playerID', 'yearID', 'AVG']].copy()
    targets= targets.rename(columns={'AVG': 'AVG_next'})
    targets['yearID'] = targets['yearID'] - 1 #shift back so it aligns

    panel= qualified.merge(
        targets, on=['playerID', 'yearID'], how= 'inner'
    )

    return panel


import os

def main():
    batting, people, fielding= load_raw_data()

    batting_agg= aggregate_batting_stints(batting)
    fielding_agg= aggregate_fielding_stints(fielding)

    batting_agg= player_age(batting_agg, people)

    primary_position= get_primary_pos(fielding_agg)
    batting_agg= batting_agg.merge(
        primary_position, on=['playerID', 'yearID'], how = 'left'
    )

    batting_agg= compute_rate_stats(batting_agg)

    qualified= filter_qualified(batting_agg)
    panel = build_lag_panel(qualified)

    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    panel.to_csv(data_path, index=False)

    print(f"panel shape: {panel.shape}")
    print(
        f"Players: {panel['playerID'].nunique()}, "
        f"Years: {panel['yearID'].min()}-{panel['yearID'].max()}"
    )


if __name__ == "__main__":
    main()


