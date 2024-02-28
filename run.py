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
    'initial_rating_dev': np.linspace(1,100, num=10),
    'c': np.linspace(10,100, num=10),
}

def hyperparameter_sweep(model_class, dataset, params_grid, seed=10):
    pool = multiprocessing.Pool(4)
    random.seed(seed)
    keys, values = zip(*params_grid.items())
    params_combs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    random.shuffle(params_combs)
    all_metrics = []
    models = [model_class(dataset.competitors, **params) for params in params_combs]
    all_metrics = pool.map(partial(evaluate, dataset=dataset), models)
    print(all_metrics)
    # print(f'{params} acc: {metrics["accuracy"]}')
    # all_metrics.append(metrics)
    return all_metrics
        

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
        rating_period='7D'
    )
    print('evaluating nba')
    # evaluate_dataset(nba_dataset)

    chess_dataset = MatchupDataset.load_from_npz('data/chess/processed.npz')
    print('evaluating chess')
    # evaluate_dataset(chess_dataset)

    nba_glicko_metrics = hyperparameter_sweep(Glicko, nba_dataset, GLICKO_PARAMS_GRID)

    x = np.arange(len(nba_glicko_metrics))
    # y = [metric['accuracy'] for metric in nba_glicko_metrics]
    y = np.maximum.accumulate([metric['accuracy'] for metric in nba_glicko_metrics])
    plt.plot(x,y)
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.show()

if __name__ == '__main__':
    main()
