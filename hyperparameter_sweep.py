import argparse
import random
import itertools
import multiprocessing
from functools import partial
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from riix.models.elo import Elo
from riix.models.glicko import Glicko
from riix.models.trueskill import TrueSkill
from riix.models.weng_lin_thurstone_mosteller import WengLinThurstoneMosteller
from riix.metrics import binary_metrics_suite
from riix.eval import grid_search_train_test, grid_search

from data_utils import load_datasets

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=CB_color_cycle)
np.seterr(divide='ignore')
import warnings
# Suppress overflow warnings globally
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

WL_TM_PARAM_RANGES = {
    'initial_sigma': (0.833, 83.33),
    'beta':  (0.4166, 41.66),
    'kappa': (0.00001, 0.001),
    'tau': (0.00833, 0.833),
}

MODEL_CONFIGS = {
    'Elo' : {
        'class' : Elo,
        'param_ranges' : ELO_PARAM_RANGES,
    },
    'Glicko' : {
        'class' : Glicko,
        'param_ranges' : GLICKO_PARAM_RANGES,
    },
    'TrueSkill' : {
        'class' : TrueSkill,
        'param_ranges' : TRUESKILL_PARAM_RANGES,
    },
    'Weng-Lin Thurstone-Mosteller' : {
        'class' : WengLinThurstoneMosteller,
        'param_ranges' : WL_TM_PARAM_RANGES,
    }
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
        best_params, _, all_metrics = grid_search_train_test(
            rating_system_class=rating_system_class,
            inputs=params,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            metric='accuracy',
            minimize_metric=False,
            return_all_metrics=True,
            num_processes=12
        )
        print('best params:')
        print(best_params)
        accs = np.maximum.accumulate([metric['accuracy'] for metric in all_metrics])
        results[:,seed] = accs
    return best_params, results

def run_grid_search(
    rating_system_class,
    param_ranges,
    dataset,
    num_samples=1000,
    seed=0
):
    params = create_uniform_grid(
        param_ranges=param_ranges,
        num_samples=num_samples,
        seed=seed
    )
    best_params, _, all_metrics = grid_search(
        rating_system_class=rating_system_class,
        inputs=params,
        dataset=dataset,
        metric='log_loss',
        minimize_metric=True,
        return_all_metrics=True,
        num_processes=12
    )
    print('best params:')
    print(best_params)

    return best_params

def main(
    game,
    max_rows=None
):

    train_dataset, test_dataset = load_datasets(game, max_rows=max_rows)
    num_repetitions = 5
    num_samples = 100

    x = np.arange(num_samples)
    all_results = {}
    for model in MODEL_CONFIGS.keys():
        print(f'running sweep on {model}')
        _, model_results = run_grid_searches(
            rating_system_class=MODEL_CONFIGS[model]['class'],
            param_ranges=MODEL_CONFIGS[model]['param_ranges'],
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            num_samples=num_samples,
            num_repetitions=num_repetitions,
        )
        all_results[model] = model_results
        plt.plot(x, model_results.mean(axis=1), label=model, lw=2.5)

    plt.title(f'{game}')
    plt.xlabel('hyperparameter sweep iteration')
    plt.ylabel('best accuracy')
    plt.legend()
    plt.savefig(f'figs/{game}_{num_samples}_{num_repetitions}.png')
    plt.show()
    np.savez(file=f'data/{game}_{num_samples}_{num_repetitions}_sweep_results.npz', **all_results)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--game', type=str)
    parser.add_argument('-mr', '--max_rows', type=int, required=False)
    args = vars(parser.parse_args())
    main(**args)