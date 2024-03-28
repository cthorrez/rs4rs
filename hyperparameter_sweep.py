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
from riix.models.skf import VSKF
from riix.metrics import binary_metrics_suite
from riix.eval import grid_search


ELO_PARAM_RANGES = {
    'k' : (0.32, 320)
}

GLICKO_PARAM_RANGES = {
    'initial_rating_dev': (35, 3500),
    'c': (6.3, 630.0),
}

TRUESKILL_PARAM_RANGES = {
    'initial_sigma': (0.833, 83.33),
    'beta':  (0.4166, 41.66),
    'tau': (0.00833, 0.833)
}

VSKF_PARAM_RANGES = {
    'v_0' : (0.1, 10.0),
    'beta': (0.0, 1.0),
    's' :  (0.1, 10.0),
    'epsilon': (0.0002, 0.02),
}


def create_uniform_grid(
    param_ranges,
    num_samples,
    seed = 0,
):
    rng = np.random.default_rng(seed)
    num_hyperparams = len(param_ranges)
    low = np.empty(num_hyperparams)
    high = np.empty(num_hyperparams)
    for idx, param_range in enumerate(param_ranges.values()):
        low[idx] = param_range[0]
        high[idx] = param_range[1]
    values = rng.uniform(low=low, high=high, size=(num_samples, num_hyperparams))
    params = []
    for sample_idx, values_row in enumerate(values):
        current_params = {}
        for param_idx, param_name in enumerate(param_ranges.keys()):
            current_params[param_name] = values_row[param_idx]
        params.append(current_params)
    return params


def run_grid_searches(
    rating_system_class,
    param_ranges,
    train_dataset,
    test_dataset,
    num_samples=20,
    num_repetitions=5,
):
    results = np.empty((num_samples, num_repetitions))
    for seed in np.arange(num_repetitions):
        params = create_uniform_grid(
            param_ranges=param_ranges,
            num_samples=num_samples,
            seed=seed
        )
        _, _, all_metrics = grid_search(
            rating_system_class=rating_system_class,
            inputs=params,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            metric='accuracy',
            minimize_metric=False,
            return_all_metrics=True,
            num_processes=16
        )
        accs = np.maximum.accumulate([metric['accuracy'] for metric in all_metrics])
        results[:,seed] = accs
    return results.mean(axis=1)


def main():
    nba_df = pd.read_csv('data/nba/processed.csv')
    nba_dataset = MatchupDataset(
        nba_df,
        competitor_cols=['team_1', 'team_2'],
        datetime_col='date',
        outcome_col='outcome',
        rating_period='7D'

    )
    dataset = nba_dataset
    # chess_dataset = MatchupDataset.load_from_npz('data/chess/processed.npz')
    # dataset = chess_dataset
    train_dataset, test_dataset = split_matchup_dataset(dataset, 0.2)

    num_repetitions = 10
    num_samples = 50

    common_grid_search_args = {
        'train_dataset':  train_dataset,
        'test_dataset': test_dataset,
        'num_samples': num_samples,
        'num_repetitions': num_repetitions
    }

    elo_accs = run_grid_searches(Elo, ELO_PARAM_RANGES, **common_grid_search_args)
    glicko_accs = run_grid_searches(Glicko, GLICKO_PARAM_RANGES, **common_grid_search_args)
    trueskill_accs = run_grid_searches(TrueSkill, TRUESKILL_PARAM_RANGES, **common_grid_search_args)
    vsfk_accs = run_grid_searches(VSKF, VSKF_PARAM_RANGES, **common_grid_search_args)


    x = np.arange(num_samples)
    plt.plot(x, elo_accs, label='elo')
    plt.plot(x, glicko_accs, label='glicko')
    plt.plot(x, trueskill_accs, label='trueskill')
    plt.plot(x, vsfk_accs, label='vskf')

    plt.legend()
    plt.show()
    




if __name__ == '__main__':
    main()