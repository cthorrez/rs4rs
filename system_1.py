import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from riix.models.elo import Elo
from riix.models.glicko import Glicko
from riix.models.trueskill import TrueSkill
from riix.models.weng_lin_thurstone_mosteller import WengLinThurstoneMosteller
from riix.metrics import binary_metrics_suite
from riix.utils import MatchupDataset
from hyperparameter_sweep import MODEL_CONFIGS, run_grid_search

GAMES = ['chess', 'nba', 'lol']

def construct_dataset_from_sweep_results(file_name):
    # num timesteps = number hyper param configurations sweeped
    # num rows = (num_timesteps) * (num_repetitions) * (num_models choose 2)
    time_steps = []
    matchups = []
    outcomes = []
    results = np.load(file_name)
    models = list(results.keys())
    combs = list(itertools.combinations(models, 2))
    for rs_1, rs_2 in combs:
        for time_step, (accs_1, accs_2) in enumerate(zip(results[rs_1], results[rs_2])):
            for acc_1, acc_2 in zip(accs_1, accs_2):
                p = acc_1 / (acc_1 + acc_2)
                outcome = float(acc_1 > acc_2)
                matchups.append((rs_1, rs_2))
                time_steps.append(time_step)
                outcomes.append(outcome)

    matchups = np.array(matchups)
    df = pd.DataFrame({
        'time_step' : time_steps,
        'model_1' : matchups[:,0],
        'model_2' : matchups[:,1],
        'outcome' : outcomes
    })
    return df

def fit_models():
    df = pd.read_csv(f'data/sweep/meta_dataset.csv')
    dataset = MatchupDataset(
        df,
        competitor_cols=['model_1', 'model_2'],
        time_step_col='time_step',
        outcome_col='outcome'
    )

    all_results = {}
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    for ax, model_name in zip(axes.flat, MODEL_CONFIGS.keys()):
        print(f'running sweep on {model_name}')
        best_params = run_grid_search(
            rating_system_class=MODEL_CONFIGS[model_name]['class'],
            param_ranges=MODEL_CONFIGS[model_name]['param_ranges'],
            dataset=dataset,
            num_samples=10,
        )
        

        model = MODEL_CONFIGS[model_name]['class'](
            competitors=dataset.competitors,
            **best_params
        )

        num_time_steps = dataset.unique_time_steps.shape[0]
        ratings = np.empty((num_time_steps, len(dataset.competitors)))
        for matchups, outcomes, time_step in dataset:
            ratings[time_step,:] = model.ratings
            model.update(matchups, outcomes, time_step=time_step)

        x = np.arange(num_time_steps)
        for comp_model in MODEL_CONFIGS.keys():
            model_idx = dataset.competitor_to_idx[comp_model]
            ax.plot(x, ratings[:,model_idx], label=comp_model, lw=2.5)
        ax.set_xlabel('hyperparameter sweep iteration')
        ax.set_ylabel('rating')
        ax.set_title(model_name)
    fig.suptitle('System 1 Rating Evolution over Hyperparameter Sweep')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figs/system_1.png')
    plt.show()


def preprocess():
    files = [
        'data/chess_100_5_sweep_results.npz',
        'data/nba_100_5_sweep_results.npz',
        'data/lol_100_5_sweep_results.npz',
    ]
    dfs = []
    for file in files:
        dfs.append(construct_dataset_from_sweep_results(file))
    df = pd.concat(dfs).sort_values('time_step')
    df.to_csv(f'data/sweep/meta_dataset.csv', index=False)




def main():
    # preprocess()
    fit_models()

if __name__ == '__main__':
    main()