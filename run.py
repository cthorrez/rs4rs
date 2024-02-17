import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from riix.utils import MatchupDataset
from riix.models.elo import Elo
from riix.models.glicko import Glicko
from riix.metrics import binary_metrics_suite
from riix.eval import evaluate, train_and_evaluate


def evaluate_dataset(dataset):
    elo = Elo(dataset.competitors)
    elo_metrics = evaluate(elo, dataset)
    print(elo_metrics)

    glicko = Glicko(dataset.competitors)
    glicko_metrics = evaluate(glicko, dataset)
    print(glicko_metrics)

def main():
    nba_df = pd.read_csv('data/nba/processed.csv')
    nba_dataset = MatchupDataset(
        nba_df,
        competitor_cols=['team_1', 'team_2'],
        datetime_col='date',
        outcome_col='outcome',
        rating_period='30D'
    )
    print('evaluating nba')
    evaluate_dataset(nba_dataset)

    chess_dataset = MatchupDataset.load_from_npz('data/chess/processed.npz')
    print('evaluating chess')
    evaluate_dataset(chess_dataset)


if __name__ == '__main__':
    main()
