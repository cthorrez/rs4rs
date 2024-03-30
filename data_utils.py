import numpy as np
import pandas as pd
from riix.utils import MatchupDataset, split_matchup_dataset

def load_dataset(
    name,
    max_rows=None
):
    if name == 'nba':
        df = pd.read_csv('data/nba/processed.csv')
        if max_rows is not None:
            df = df.head(max_rows)
        dataset = MatchupDataset(
            df,
            competitor_cols=['team_1', 'team_2'],
            datetime_col='date',
            outcome_col='outcome',
            rating_period='365D'
        )
    elif name == 'chess':
        time_steps, matchups, outcomes = np.load('data/chess/processed.npz').values()
        df = pd.DataFrame({
            'time_step' : time_steps,
            'player_1' : matchups[:,0],
            'player_2' : matchups[:,1],
            'outcome' : outcomes
        })
        if max_rows is not None:
            df = df.head(max_rows)
        dataset = MatchupDataset(
            df,
            competitor_cols=['player_1', 'player_2'],
            time_step_col='time_step',
            outcome_col='outcome',
        )
    elif name == 'lol':
        df = pd.read_csv('data/lol/league_of_legends.csv')
        if max_rows is not None:
            df = df.head(max_rows)
        dataset = MatchupDataset(
            df,
            competitor_cols=['team_1', 'team_2'],
            datetime_col='date',
            outcome_col='outcome',
            rating_period='7D'
        )
    else:
        print(f"I don't got that data!!! ({name})")

    return dataset


def load_datasets(
    name,
    test_fraction=0.2,
    max_rows=None
):
    dataset = load_dataset(name, max_rows=max_rows)
    train_dataset, test_dataset = split_matchup_dataset(dataset, test_fraction)
    return train_dataset, test_dataset