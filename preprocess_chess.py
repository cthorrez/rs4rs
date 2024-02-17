import glob
import pandas as pd
import numpy as np

def main():
    dfs = []
    for file in glob.glob('data/chess/*'):
        print(file)
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
    print(df.shape)

    df['player_1'] = df['player_1'].astype(np.int32)
    df['player_2'] = df['player_2'].astype(np.int32)
    df['outcome'] = df['outcome'].astype(np.float32)
    df.to_csv('data/chess/all_matchups_processed.csv', index=False)

if __name__ == '__main__':
    main()