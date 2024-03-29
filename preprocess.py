import glob
import pandas as pd
import numpy as np

def preprocess_chess():
    print('preprocessing chess data')
    dfs = []
    # for file in glob.glob('data/chess/raw/*.csv'): # this is too much data it's killing my computer
    for file in ['data/chess/raw/primary_training_part1.csv']:

        df = pd.read_csv(file)
        if 'primary' in file:
            df['time_step'] = df['MonthID'] - 1
            df = df.drop(columns=['MonthID'])
            df = df[['time_step', 'WhitePlayer', 'BlackPlayer', 'WhiteScore']]
            df = df.set_axis(['time_step', 'player_1', 'player_2', 'outcome'], axis=1)
        else:
            df['time_step'] = df['MonthID'] - 1
            df = df.drop(columns=['MonthID'])
            df = df[['time_step', 'Player1', 'Player2', 'Player1Score']]
            df = df.set_axis(['time_step', 'player_1', 'player_2', 'outcome'], axis=1)

        dfs.append(df)
    df = pd.concat(dfs)
    df = df.sort_values(by='time_step')

    df['player_1'] = df['player_1'].astype(np.int32)
    df['player_2'] = df['player_2'].astype(np.int32)
    df['outcome'] = df['outcome'].astype(np.float32)

    time_steps = df['time_step'].values
    matchups = df[['player_1', 'player_2']].values
    outcomes = df['outcome'].values
    np.savez_compressed('data/chess/processed.npz', time_steps, matchups, outcomes)

def preprocess_nba():
    print('preprocessing nba data')
    df = pd.read_csv('data/nba/raw/nbaallelo_csv.csv')
    df['outcome'] = np.where(df['game_result'] == 'w', 1.0, np.where(df['game_result'] == 'l', 0.0, 0.5))
    df = df[['date_game', 'team_id', 'opp_id', 'outcome']]
    df = df.set_axis(['date', 'team_1', 'team_2', 'outcome'], axis=1)
    df.to_csv('data/nba/processed.csv', index=False)


if __name__ == '__main__':
    preprocess_nba()
    preprocess_chess()
