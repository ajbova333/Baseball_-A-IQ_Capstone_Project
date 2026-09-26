import pandas as pd
from difflib import get_close_matches


people_path= 'data/raw/People.csv'
panel_path= 'data/processed/panel.csv'

def load_people(path= people_path):
    return pd.read_csv(path)


def build_name_lookup(people):
    """This is to map lowercase 'first last' to playerID. If a name is
    shared by more than one player, keep whomever had their finalGame most recently."""
    people= people.dropna(subset=['nameFirst', 'nameLast']).copy()
    people['full_name']= (people['nameFirst']+ ' ' + people['nameLast']).str.strip()
    people['full_name_lower'] = people['full_name'].str.lower()
    people= people.sort_values('finalGame').drop_duplicates('full_name_lower', keep= 'last')
    return dict(zip(people['full_name_lower'], people['playerID']))


def find_player(name, name_lookup):
    """Match user-entered name against the lookup table."""
    normalized= name.strip().lower()

    if normalized in name_lookup:
        return {'status': 'found', 'player_id': name_lookup[normalized], 'matched_name': normalized}

    close= get_close_matches(normalized, name_lookup.keys(), n=1, cutoff= 0.6)
    if close:
            return {'status': 'suggestion', 'suggested_name': close[0]}
    return{'status': 'not_found'}

def get_latest_season(player_id, panel):
    """Return Most Recent feature row in panel for player."""
    player_rows= panel[panel['playerID'] == player_id]
    if player_rows.empty:
        return None
    return player_rows.sort_values('yearID', ascending=False).iloc[0]
    