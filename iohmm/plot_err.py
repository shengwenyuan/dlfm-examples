import numpy as np
import numpy.random as npr
import random
import ssm
import time
import multiprocessing as mp
import os
import scipy.stats as st
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style='ticks', font_scale=1.5)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = ['sans-serif']
cols_traces = ['#BE1F24', '#2E3192']

# * * * * * * os dirs * * * * * *
root = os.path.dirname(os.path.abspath(__file__))
npy_dir = os.path.join(root, "output", "results_IOHMM")
graph_dir = os.path.join(root, "figs")
os.makedirs(graph_dir, exist_ok=True)

# * * * * * * parameters of the GLM-HMM * * * * * *
num_states = 3        # number of discrete states
obs_dim = 1           # number of observed dimensions
num_categories = 2    # number of categories for output
input_dim = 2         # input dimensions
num_trials = 997
init_trials = 100

# * * * * groundtruth HMM for generation * * * * * *
true_iohmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                   observation_kwargs=dict(C=num_categories), transitions="standard")
gen_weights = np.array([[[6, 1]], [[2, -3]], [[2, 3]]])
gen_log_trans_mat = np.log(np.array([[[0.98, 0.01, 0.01], [0.05, 0.92, 0.03], [0.02, 0.03, 0.94]]]))
true_iohmm.observations.params = gen_weights
true_iohmm.transitions.params = gen_log_trans_mat
gen_trans_mat = np.exp(gen_log_trans_mat)[0]


def plot_rmse_w():
    # TODO: For model mismatch
    true_weights = np.reshape(gen_weights, (num_states, input_dim))

    # Plotting error in weights
    num_repeats = 1
    error_in_weights=[]
    for seed in np.arange(num_repeats):
        error_in_weights_dlfm = np.load(os.path.join(npy_dir, "dlfm", "1001", "dlfm_errorinweights_atseed"+str(seed)+".npy"))
        error_in_weights_dlfm = np.convolve(error_in_weights_dlfm, np.ones(5)/5, mode='valid')
        error_in_weights += error_in_weights_dlfm.tolist()

        error_in_weights_random = np.load(os.path.join(npy_dir, "mcmc", "1001", "random_gibbs_PG_errorinweights_atseed"+str(seed)+"_gibbs_400.npy"))
        error_in_weights_random = np.convolve(error_in_weights_random, np.ones(5)/5, mode='valid')
        error_in_weights += error_in_weights_random.tolist()

    sampling_method = (['dlfm'] * num_trials + ['random'] * num_trials) * num_repeats
    trials = (np.arange(num_trials) + init_trials).tolist() * 2 * num_repeats

    weights_list = {"trial \#": trials, "Method": sampling_method, "RMSE ($\{w_k\}_{k=1}^K$)": error_in_weights}
    df = pd.DataFrame(weights_list, columns=['trial \#', 'Method', 'RMSE ($\{w_k\}_{k=1}^K$)'])
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.lineplot(x="trial \#", y="RMSE ($\{w_k\}_{k=1}^K$)", hue="Method", data=df, ax=ax, palette=cols_traces, linewidth=2, alpha=0.8)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)
    ax.get_legend().remove()
    plt.xticks()
    plt.yticks()
    plt.tight_layout()
    plt.grid()
    # plt.show()
    plt.savefig(os.path.join(graph_dir, f"rmse_weights_{num_trials}.png"), dpi=400)


def plot_rmse_p():
    # TODO: For model mismatch
    true_weights = np.reshape(gen_weights, (num_states, input_dim))

    # Plotting error in weights
    num_repeats = 1
    error_in_weights=[]
    for seed in np.arange(num_repeats):
        error_in_weights_dlfm = np.load(os.path.join(npy_dir, "dlfm", "1001", "dlfm_errorinPs_atseed"+str(seed)+".npy"))
        error_in_weights_dlfm = np.convolve(error_in_weights_dlfm, np.ones(5)/5, mode='valid')
        error_in_weights += error_in_weights_dlfm.tolist()

        error_in_weights_random = np.load(os.path.join(npy_dir, "mcmc", "1001", "random_gibbs_PG_errorinPs_atseed"+str(seed)+"_gibbs_400.npy"))
        error_in_weights_random = np.convolve(error_in_weights_random, np.ones(5)/5, mode='valid')
        error_in_weights += error_in_weights_random.tolist()

    sampling_method = (['dlfm'] * num_trials + ['random'] * num_trials) * num_repeats
    trials = (np.arange(num_trials) + init_trials).tolist() * 2 * num_repeats

    weights_list = {"trial \#": trials, "Method": sampling_method, "RMSE (A)": error_in_weights}
    df = pd.DataFrame(weights_list, columns=['trial \#', 'Method', 'RMSE (A)'])
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.lineplot(x="trial \#", y="RMSE (A)", hue="Method", data=df, ax=ax, palette=cols_traces, linewidth=2, alpha=0.8)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)
    ax.get_legend().remove()
    plt.xticks()
    plt.yticks()
    plt.tight_layout()
    plt.grid()
    # plt.show()
    plt.savefig(os.path.join(graph_dir, f"rmse_Ps_{num_trials}.png"), dpi=400)


if __name__ == "__main__":
    plot_rmse_w()
    plot_rmse_p()