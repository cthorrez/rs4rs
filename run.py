import random
import itertools
import multiprocessing
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from riix.utils import MatchupDataset
from riix.models.elo import Elo
from riix.models.glicko import Glicko
from riix.metrics import binary_metrics_suite
from riix.eval import evaluate, train_and_evaluate

GLICKO_PARAMS_GRID = {
    'initial_rating_dev': np.linspace(1,100, num=20),
    'c': np.linspace(10,100, num=20),
}

ELO_PARAMS_GRID = {
    'k' : np.linspace(1,128, num=50)
}

def hyperparameter_sweep(model_class, dataset, params_grid, seed=0):
    pool = multiprocessing.Pool(8)
    random.seed(seed)
    keys, values = zip(*params_grid.items())
    params_combs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    random.shuffle(params_combs)
    all_metrics = []
    models = [model_class(dataset.competitors, **params) for params in params_combs]
    all_metrics = pool.map(partial(evaluate, dataset=dataset), models)
    return all_metrics, params_combs
        

def main():
    nba_df = pd.read_csv('data/nba/processed.csv')
    nba_dataset = MatchupDataset(
        nba_df,
        competitor_cols=['team_1', 'team_2'],
        datetime_col='date',
        outcome_col='outcome',
        rating_period='7D'
    )
    chess_dataset = MatchupDataset.load_from_npz('data/chess/processed.npz')
    dataset = chess_dataset

    # glicko_metrics = hyperparameter_sweep(Glicko, dataset, GLICKO_PARAMS_GRID)
    elo_metrics, params_combs = hyperparameter_sweep(Elo, dataset, ELO_PARAMS_GRID)
    x = np.arange(len(elo_metrics))
    x = [params['k'] for params in params_combs]
    y = [metric['accuracy'] for metric in elo_metrics]
    # y = np.maximum.accumulate([metric['accuracy'] for metric in elo_metrics])
    plt.scatter(x,y)
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.show()

if __name__ == '__main__':
    main()
