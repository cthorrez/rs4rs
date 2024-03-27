import random
import itertools
import multiprocessing
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from riix.utils import MatchupDataset, split_matchup_dataset
from riix.models.elo import Elo
from riix.models.glicko import Glicko
from riix.models.trueskill import TrueSkill
from riix.metrics import binary_metrics_suite
from riix.eval import grid_search


ELO_PARAM_RANGES = {
    'k' : (1e-6, 128)
}

GLICKO_PARAM_RANGES = {
    'initial_rating_dev': (0.01, 1000),
    'c': (1, 200),
}

TRUESKILL_PARAM_RANGES = {
    'initial_sigma': (1,100),
    'beta':  (1, 50),
    'tau': (0.01, 10.0)
}


def create_uniform_grid(
    param_ranges,
    n_samples,
    seed = 0,
):
    rng = np.random.default_rng(seed)
    n_hyperparams = len(param_ranges)
    low = np.empty(n_hyperparams)
    high = np.empty(n_hyperparams)
    for idx, param_range in enumerate(param_ranges.values()):
        low[idx] = param_range[0]
        high[idx] = param_range[1]
    values = rng.uniform(low=low, high=high, size=(n_samples, n_hyperparams))
    params = []
    for sample_idx, values_row in enumerate(values):
        current_params = {}
        for param_idx, param_name in enumerate(param_ranges.keys()):
            current_params[param_name] = values_row[param_idx]
        params.append(current_params)
    return params


def main():
    nba_df = pd.read_csv('data/nba/processed.csv')
    nba_dataset = MatchupDataset(
        nba_df,
        competitor_cols=['team_1', 'team_2'],
        datetime_col='date',
        outcome_col='outcome',
        rating_period='7D'
    )
    seed = 42
    train_dataset, test_dataset = split_matchup_dataset(nba_dataset, 0.2)

    num_runs = 50
    elo_params = create_uniform_grid(ELO_PARAM_RANGES, num_runs, seed)
    glicko_params = create_uniform_grid(GLICKO_PARAM_RANGES, num_runs, seed)
    trueskill_params = create_uniform_grid(TRUESKILL_PARAM_RANGES, num_runs, seed)

    grid_search_params = {
        'train_dataset' : train_dataset,
        'test_dataset': test_dataset,
        'metric': 'accuracy',
        'minimize_metric': False,
        'num_processes' : 10,
        'return_all_metrics': True
    }
    _, _, elo_metrics = grid_search(
        rating_system_class=Elo,
        inputs=elo_params,
        **grid_search_params
    )

    _, _, glicko_metrics = grid_search(
        rating_system_class=Glicko,
        inputs=glicko_params,
        **grid_search_params
    )

    _, _, trueskill_metrics = grid_search(
        rating_system_class=TrueSkill,
        inputs=trueskill_params,
        **grid_search_params
    )

    x = np.arange(len(elo_metrics))
    elo_y = np.maximum.accumulate([metrics['accuracy'] for metrics in elo_metrics])
    glicko_y = np.maximum.accumulate([metrics['accuracy'] for metrics in glicko_metrics])
    trueskill_y = np.maximum.accumulate([metrics['accuracy'] for metrics in trueskill_metrics])

    plt.plot(x, elo_y, label='elo')
    plt.plot(x, glicko_y, label='glicko')
    plt.plot(x, trueskill_y, label='trueskill')
    plt.legend()
    plt.show()
    




if __name__ == '__main__':
    main()