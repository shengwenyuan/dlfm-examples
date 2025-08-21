import numpy as np
import numpy.random as npr
import random
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
npy_dir = os.path.join(root, "output", "results_MGLM")
graph_dir = os.path.join(root, "figs")
os.makedirs(graph_dir, exist_ok=True)

# * * * Set the parameters of the MGLM * * *
num_states = 2        # number of discrete states
obs_dim = 1           # number of observed dimensions
num_categories = 2    # number of categories for output
input_dim = 2         # input dimensions

init_trials = 100
num_trials = 1001 - 4 # consider convolve

num_repeats = 1

# * * * * * * groundtruth MGLM for generation * * * * * *
true_pi0 = np.array([0.6, 0.4])
true_weights = np.array([[3,-6], [3, 6]])
true_weights = np.reshape(true_weights, (num_states,input_dim))


def plot_rmse_w():
    # Plotting error in weights
    error=[]
    for seed in np.arange(num_repeats):
        error_in_weights_dlfm = np.load(os.path.join(npy_dir, "dlfm", "1001", f"dlfm_error_in_weights_atseed{seed}.npy"))
        error_in_weights_dlfm = np.convolve(error_in_weights_dlfm, np.ones(5)/5, mode='valid')
        error += error_in_weights_dlfm.tolist()

        error_in_weights_random = np.load(os.path.join(npy_dir, "mcmc", "1001", f"random_atseed{seed}_error_in_weights.npy"))
        error_in_weights_random = np.convolve(error_in_weights_random, np.ones(5)/5, mode='valid')
        error += error_in_weights_random.tolist()

        
    sampling_method = (['dlfm'] * num_trials + ['random'] * num_trials) * num_repeats
    trials = (np.arange(num_trials) + init_trials).tolist() * 2 * num_repeats
   
    weights_list = {"trial \#": trials, "Method": sampling_method, "RMSE ($\{w_k\}_{k=1}^K$)": error}
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


def plot_rmse_pi():
    # Plotting error in weights
    error=[]
    for seed in np.arange(num_repeats):
        error_in_pis_dlfm = np.load(os.path.join(npy_dir, "dlfm", "1001", f"dlfm_error_in_pis_atseed{seed}.npy"))
        error_in_pis_dlfm = np.convolve(error_in_pis_dlfm, np.ones(5)/5, mode='valid')
        error += error_in_pis_dlfm.tolist()

        error_in_pis_random = np.load(os.path.join(npy_dir, "mcmc", "1001", f"random_atseed{seed}_error_in_pis.npy"))
        error_in_pis_random = np.convolve(error_in_pis_random, np.ones(5)/5, mode='valid')
        error += error_in_pis_random.tolist()

    sampling_method = (['dlfm'] * num_trials + ['random'] * num_trials) * num_repeats
    trials = (np.arange(num_trials) + init_trials).tolist() * 2 * num_repeats
   
    weights_list = {"trial \#": trials, "Method": sampling_method, "RMSE (A)": error}
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


def plot_input_selection():
    seed = 0
    
    selected_inputs_mcmc = np.load(os.path.join(npy_dir, "mcmc", "1001", f"random_atseed{seed}_selectedinputs.npy"))
    selected_inputs_dlfm = np.load(os.path.join(npy_dir, "dlfm", "1001", f"dlfm_selectedinputs_atseed{seed}.npy"))
    selected_inputs_mcmc = np.array(selected_inputs_mcmc)[:, 0]  # First 1000 selected inputs
    selected_inputs_dlfm = selected_inputs_dlfm[init_trials:, 0]  # Skip initial trials, get first column
    
    num_samples = 500
    stim_vals = np.arange(-10, 10, step=0.01)
    xgrid = np.linspace(start=stim_vals[0], stop=stim_vals[-1], num=num_samples)
    samples = np.ones((num_samples, input_dim))
    samples[:, 0] = xgrid
    
    ygrid = np.empty((num_states, num_samples))
    for k in range(num_states):
        logits = samples @ true_weights[k]
        ygrid[k] = 1 / (1 + np.exp(-logits))
    
    fig, ax1 = plt.subplots(figsize=(6, 5), facecolor='white')
    bins = np.arange(-10, 10, 0.5)
    
    ax1.hist(selected_inputs_dlfm, alpha=0.7, color=cols_traces[0], label='DLFM', bins=bins)
    ax1.hist(selected_inputs_mcmc, alpha=0.7, color=cols_traces[1], label='MCMC', bins=bins)
    ax1.set_xlabel("input")
    ax1.set_ylabel("frequency")
    
    ax2 = ax1.twinx()
    for k in range(num_states):
        ax2.plot(xgrid, ygrid[k], linewidth=3, color='gray', alpha=0.3, zorder=1)
    ax2.set_ylabel("$p(y=1|x)$")
    
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    ax1.legend(loc='upper left')
    plt.title("Input Selection: MGLM", fontsize=16)
    plt.tight_layout()
    plt.grid(alpha=0.3)
    # plt.show()
    plt.savefig(os.path.join(graph_dir, f"input_selection_{num_trials}.png"), dpi=400)


def print_total_times():
    print("=" * 50)
    print("TOTAL EXECUTION TIMES (MGLM)")
    print("=" * 50)
    
    
    for seed in range(num_repeats):
        print(f"\nSeed {seed}:")
        mcmc_time = np.load(os.path.join(npy_dir, "mcmc", "1001", f"random_atseed{seed}_total_time.npy"))
        print(f"  MCMC: {mcmc_time:.2f} seconds ({mcmc_time/60:.2f} minutes)")
        dlfm_time = np.load(os.path.join(npy_dir, "dlfm", "1001", f"dlfm_total_time_atseed{seed}.npy"))
        print(f"  DLFM: {dlfm_time:.2f} seconds ({dlfm_time/60:.2f} minutes)")
    
    mcmc_times = []
    dlfm_times = []
    
    for seed in range(num_repeats):
        mcmc_time = np.load(os.path.join(npy_dir, "mcmc", "1001", f"random_atseed{seed}_total_time.npy"))
        mcmc_times.append(mcmc_time)
        dlfm_time = np.load(os.path.join(npy_dir, "dlfm", "1001", f"dlfm_total_time_atseed{seed}.npy"))
        dlfm_times.append(dlfm_time)
    
    avg_mcmc = np.mean(mcmc_times)
    avg_dlfm = np.mean(dlfm_times)
    print(f"\nAverage MCMC time: {avg_mcmc:.2f} seconds ({avg_mcmc/60:.2f} minutes)")
    print(f"Average DLFM time: {avg_dlfm:.2f} seconds ({avg_dlfm/60:.2f} minutes)")
    
    speedup = avg_mcmc / avg_dlfm
    print(f"DLFM speedup: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than MCMC")
    print("=" * 50)


if __name__ == "__main__":
    plot_rmse_w()
    plot_rmse_pi()
    plot_input_selection()
    print_total_times()