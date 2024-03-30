import json
import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from riix.models.elo import Elo
from riix.models.glicko import Glicko
from riix.models.trueskill import TrueSkill
from riix.models.weng_lin_thurstone_mosteller import WengLinThurstoneMosteller
from riix.metrics import binary_metrics_suite
from riix.eval import grid_search
from riix.utils import MatchupDataset
from hyperparameter_sweep import MODEL_CONFIGS, create_uniform_grid
from data_utils import load_dataset

GAMES = ['chess', 'nba', 'lol']
TIME_STEP_MAP = {
    'chess' : 'months',
    'nba' : 'years',
    'lol' : 'weeks',

}

# these are determined during the sweep
BEST_PARAMS = {
    'chess' : {
        'Elo' : {'k' : 48.36},
        'Glicko' : {'initial_rating_dev' : 834.17, 'c' : 38.74},
        'TrueSkill' : {'initial_sigma' : 74.28, 'beta' : 9.786, 'tau' : 0.52},
        'Weng-Lin Thurstone-Mosteller' : {'initial_sigma' : 78.69, 'beta' : 5.64, 'tau' : 0.057, 'kappa' : 8.66e-4},
    },
    'nba' : {
        'Elo' : {'k' : 12.98},
        'Glicko' : {'initial_rating_dev' : 2895.32, 'c' : 108.9},
        'TrueSkill' : {'initial_sigma' : 74.64, 'beta' : 17.85, 'tau' : 0.49},
        'Weng-Lin Thurstone-Mosteller' : {'initial_sigma' : 55.33, 'beta' : 38.83, 'tau' : 0.53, 'kappa' : 2.15e-4},
    },
    'lol' : {
        'Elo' : {'k' : 124.65},
        'Glicko' : {'initial_rating_dev' : 834.17, 'c' : 38.74},
        'TrueSkill' : {'initial_sigma' : 6.90, 'beta' : 3.32, 'tau' : 0.72},
        'Weng-Lin Thurstone-Mosteller' : {'initial_sigma' : 26.89, 'beta' : 7.95, 'tau' : 0.67, 'kappa' : 8.81e-4},
    },
}

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

def get_results():
    results = defaultdict(lambda : defaultdict(list))
    for game in GAMES:
        max_rows = 200000 if game == 'chess' else None
        dataset = load_dataset(game, max_rows=max_rows)
        for model_name in MODEL_CONFIGS.keys():
            print(f'fitting {model_name} on {game}')
            model = MODEL_CONFIGS[model_name]['class'](
                competitors = dataset.competitors,
                **BEST_PARAMS[game][model_name]
            )
            for matchups, outcomes, time_step in dataset:
                probs = model.predict(matchups, time_step=time_step)
                model.update(matchups, outcomes, time_step=time_step)
                acc = ((probs >= 0.5) == outcomes).mean()
                results[game][model_name].append(acc)

    json.dump(results, open('data/best_param_time_step_results.json', 'w'))
    return results


def construct_dataset_from_results(game):
    results = json.load(open('data/best_param_time_step_results.json'))[game]
    time_steps = []
    matchups = []
    outcomes = []
    models = list(results.keys())
    combs = list(itertools.combinations(models, 2))
    for time_step in range(len(results['Elo'])):
        for rs_1, rs_2 in combs:
            acc_1 = results[rs_1][time_step]
            acc_2 = results[rs_2][time_step]
            outcome = float(acc_1 > acc_2)
            if acc_1 == acc_2:
                outcome = 0.5
            time_steps.append(time_step)
            matchups.append((rs_1, rs_2))
            outcomes.append(outcome)
            
    matchups = np.array(matchups)
    df = pd.DataFrame({
        'time_step' : time_steps,
        'model_1' : matchups[:,0],
        'model_2' : matchups[:,1],
        'outcome' : outcomes
    })
    return df


def grid_search_system_2(dataset):
    all_results = {}
    for model in MODEL_CONFIGS.keys():
        print(f'running sweep on {model}')
        best_params = run_grid_search(
            rating_system_class=MODEL_CONFIGS[model]['class'],
            param_ranges=MODEL_CONFIGS[model]['param_ranges'],
            dataset=dataset,
            num_samples=1000,
        )
        all_results[model] = best_params
    return all_results

    
    

def main():
    # results = get_results()
    for game in GAMES:
        print(f'hyperparam sweeping meta results for {game}')
        df = construct_dataset_from_results(game)
        dataset = MatchupDataset(
            df,
            competitor_cols=['model_1', 'model_2'],
            time_step_col='time_step',
            outcome_col='outcome'
        )
        all_best_params = grid_search_system_2(dataset)


        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        for ax_id, (eval_model_name, ax) in enumerate(zip(MODEL_CONFIGS.keys(), axes.flat)):
            model = MODEL_CONFIGS[eval_model_name]['class'](
                competitors=dataset.competitors,
                **all_best_params[eval_model_name]
            )

            num_time_steps = dataset.unique_time_steps.shape[0]
            ratings = np.empty((num_time_steps, len(dataset.competitors)))
            for matchups, outcomes, time_step in dataset:
                ratings[time_step,:] = model.ratings
                model.update(matchups, outcomes, time_step=time_step)

            x = np.arange(num_time_steps)
            for comp_model_name in MODEL_CONFIGS.keys():
                model_idx = dataset.competitor_to_idx[comp_model_name]
                ax.plot(x, ratings[:,model_idx], label=comp_model_name, lw=2.5)
            ax.set_title(f'{eval_model_name}')
            ax.set_xlabel(f'time_step ({TIME_STEP_MAP[game]})')
            ax.set_ylabel('rating')
        plt.legend()
            
        fig.suptitle(f'Ratings by/of Different Rating Systems for {game}')
        plt.tight_layout()
        plt.savefig(f'figs/system_2_{game}.png')
        plt.show()



if __name__ == '__main__':
    main()