import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from riix.utils import MatchupDataset
from riix.models.elo import Elo
from riix.models.glicko import Glicko
from riix.metrics import binary_metrics_suite

def nba():
    nba_df = pd.read_csv('data/nbaallelo_csv.csv')
    nba_df['outcome'] = np.where(nba_df['game_result'] == 'w', 1.0, np.where(nba_df['game_result'] == 'l', 0.0, 0.5))
    nba_dataset = MatchupDataset(
        nba_df,
        competitor_cols=['team_id', 'opp_id'],
        outcome_col='outcome',
        datetime_col='date_game',
        rating_period='30D'
    )
    nba_elo = Elo(competitors=nba_dataset.competitors)
    nba_elo_probs = nba_elo.fit_dataset(nba_dataset, return_pre_match_probs=True)
    nba_elo_metrics = binary_metrics_suite(nba_elo_probs, nba_dataset.outcomes)
    print(nba_elo_metrics)

    nba_glicko = Glicko(competitors=nba_dataset.competitors)
    nba_glicko_probs = nba_glicko.fit_dataset(nba_dataset, return_pre_match_probs=True)
    nba_glicko_metrics = binary_metrics_suite(nba_glicko_probs, nba_dataset.outcomes)
    print(nba_glicko_metrics)

def chess():
    df = pd.read_csv('data/chess/primary_training_part1.csv')
    df['timestamp'] = df['MonthID'].astype(np.float64) * (30 * 24 * 60 * 60)
    chess_dataset = MatchupDataset(
        df,
        competitor_cols=['WhitePlayer', 'BlackPlayer'],
        outcome_col='WhiteScore',
        timestamp_col='timestamp'
    )
    chess_elo = Elo(competitors=chess_dataset.competitors)
    chess_elo_probs = chess_elo.fit_dataset(chess_dataset, return_pre_match_probs=True)
    chess_elo_metrics = binary_metrics_suite(chess_elo_probs, chess_dataset.outcomes)
    print(chess_elo_metrics)

    chess_glicko = Glicko(competitors=chess_dataset.competitors)
    chess_glicko_probs = chess_glicko.fit_dataset(chess_dataset, return_pre_match_probs=True)
    chess_glicko_metrics = binary_metrics_suite(chess_glicko_probs, chess_dataset.outcomes)
    print(chess_glicko_metrics)



if __name__ == '__main__':
    # nba()
    chess()