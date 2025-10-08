import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_theme(style='ticks', font_scale=1.5)
mpl.rcParams['text.usetex'] = False  # Set to False to avoid LaTeX issues
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = ['sans-serif']
cols_traces = ['#2E3192', '#BE1F24', '#1f77b4', '#ff7f0e', '#2ca02c']

root = os.path.dirname(os.path.abspath(__file__))
ibl_dir = os.path.join(root, "output", "results_IOHMM")
graph_dir = os.path.join(root, "figs")
os.makedirs(graph_dir, exist_ok=True)

num_states = K = 3   # number of discrete states = [engaged, disengaged, right/left-bias]
obs_dim = 1           # number of observed dimensions
num_categories = 2    # number of categories for output = [0, 1(rightward choice=_ibl_trials.choice=1)]
input_dim = 4         # input dimensions = [stimulus = contrastRight - contrastLeft, 
                                            #bias = 1, 
                                            #prev_choice = _ibl_trials.choice, 
                                            #prev_stimulus_side(win-stay, lose-switch) = prev_contrastR/L]
initial_trials = 100
n_folds = 3
mcmc_downsample_step = 1
num_gibbs_samples = 2000
# seed_list = [0, 1, 2, 3]
seed_list = [0, 1, 2]
num_repeats = len(seed_list)

def load_mcmc_data_all_seeds():
    mcmc_dir = os.path.join(ibl_dir, "mcmc", "795")
    null_ll_per_trial = np.log(0.5)
    all_data = []
    
    for seed in seed_list:
        # if seed==1: seed = 2
        for fold in range(n_folds):
            fold_dir = os.path.join(mcmc_dir, f"fold_{fold}")
            ll_file = os.path.join(fold_dir, f"ibl_gibbs_PG_LL_atseed{seed}_gibbs_{num_gibbs_samples}.npy")
            if os.path.exists(ll_file):
                ll = np.load(ll_file)
                n_trials = ll.shape[0]
                ll = np.concatenate([[ll[0]], ll[1::mcmc_downsample_step]])
                relative_ll = (ll / n_trials) - null_ll_per_trial
                
                # Create proper x-axis considering mcmc_downsample_step
                n_points = len(relative_ll)
                trials = np.zeros(n_points)
                trials[0] = initial_trials  # First point
                for i in range(1, n_points):
                    trials[i] = initial_trials + i * mcmc_downsample_step
                
                # Apply smoothing
                relative_ll_smooth = np.convolve(relative_ll, np.ones(5)/5, mode='valid')
                trials_smooth = trials[:len(relative_ll_smooth)]
                
                for trial, rel_ll in zip(trials_smooth, relative_ll_smooth):
                    all_data.append({
                        'trial #': trial,
                        'Method': 'MCMC',
                        'Fold': fold,
                        'Seed': seed,
                        'Relative Log-Likelihood': rel_ll
                    })
    
    return all_data


def load_dlfm_data_all_seeds():
    dlfm_dir = os.path.join(ibl_dir, "dlfm", "795")
    null_ll_per_trial = np.log(0.5)
    all_data = []
    
    for seed in seed_list:
        for fold in range(n_folds):
            fold_dir = os.path.join(dlfm_dir, f"fold_{fold}")
            ll_file = os.path.join(fold_dir, f"ibl_dlfm_lls_atseed{seed}.npy")
            if os.path.exists(ll_file):
                ll = np.load(ll_file)
                n_trials = ll.shape[0]
                relative_ll = (ll / n_trials) - null_ll_per_trial
                
                trials = np.arange(len(relative_ll)) + initial_trials
                # Apply smoothing
                relative_ll_smooth = np.convolve(relative_ll, np.ones(5)/5, mode='valid')
                trials_smooth = trials[:len(relative_ll_smooth)]
                
                # Add to all_data
                for trial, rel_ll in zip(trials_smooth, relative_ll_smooth):
                    all_data.append({
                        'trial #': trial,
                        'Method': 'MCP',
                        'Fold': fold,
                        'Seed': seed,
                        'Relative Log-Likelihood': rel_ll
                    })
    
    return all_data if all_data else None


def plot_mcmc_only():
    """Plot MCMC log-likelihood curves with confidence intervals across seeds"""
    mcmc_data = load_mcmc_data_all_seeds()
    
    if not mcmc_data:
        print("MCMC data not available. Skipping MCMC plot.")
        return
    
    df = pd.DataFrame(mcmc_data)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.lineplot(x="trial #", y="Relative Log-Likelihood", 
                data=df, ax=ax, color=cols_traces[1], 
                linewidth=2, alpha=0.8, label='MCMC')
    
    ax.set_xlabel('Trial #')
    ax.set_ylabel('Relative Log-Likelihood')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, f"mcmc_ll_{n_folds}folds_{num_repeats}seeds.png"), dpi=400)
    plt.show()


def plot_dlfm_only():
    """Plot MCP log-likelihood curves with confidence intervals across seeds"""
    dlfm_data = load_dlfm_data_all_seeds()
    
    if dlfm_data is None:
        print("MCP data not available. Skipping MCP-only plot.")
        return
    
    df = pd.DataFrame(dlfm_data)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.lineplot(x="trial #", y="Relative Log-Likelihood", 
                data=df, ax=ax, color=cols_traces[0], 
                linewidth=2, alpha=0.8, label='MCP')
    
    ax.set_xlabel('Trial #')
    ax.set_ylabel('Relative Log-Likelihood')
    ax.set_title('MCP Log-Likelihood with Confidence Intervals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, f"dlfm_ll_by_folds_{n_folds}folds_{num_repeats}seeds.png"), dpi=400)
    plt.show()


def plot_comprehensive_comparison():
    mcmc_data = load_mcmc_data_all_seeds()
    dlfm_data = load_dlfm_data_all_seeds()
    
    all_data = []
    if mcmc_data:
        all_data.extend(mcmc_data)
    if dlfm_data:
        all_data.extend(dlfm_data)
    if not all_data:
        print("No data available for comparison plot.")
        return
    
    df = pd.DataFrame(all_data)
    fig, ax = plt.subplots(figsize=(10, 6))
    if dlfm_data and mcmc_data:
        sns.lineplot(x="trial #", y="Relative Log-Likelihood", hue="Method", 
                    data=df, ax=ax, palette=cols_traces[:2], linewidth=2, alpha=0.8, errorbar=('ci', 68))
        # ax.set_title('Log-Likelihood Comparison')
    elif mcmc_data:
        sns.lineplot(x="trial #", y="Relative Log-Likelihood", 
                    data=df[df['Method'] == 'MCMC'], ax=ax, 
                    color=cols_traces[1], linewidth=2, alpha=0.8, label='MCMC', errorbar=('ci', 68))
        ax.set_title('MCMC Log-Likelihood')
    elif dlfm_data:
        sns.lineplot(x="trial #", y="Relative Log-Likelihood", 
                    data=df[df['Method'] == 'MCP'], ax=ax, 
                    color=cols_traces[0], linewidth=2, alpha=0.8, label='MCP', errorbar=('ci', 68))
        ax.set_title('MCP Log-Likelihood')
    
    ax.set_xlabel('Trial #')
    ax.set_ylabel('Relative Log-Likelihood')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, f"ll_comparison_{n_folds}folds_{num_repeats}seeds.png"), dpi=400)
    plt.show()


def print_summary_stats():
    mcmc_data = load_mcmc_data_all_seeds()
    dlfm_data = load_dlfm_data_all_seeds()
    
    print("=== Log-Likelihood Summary Statistics Across All Seeds ===")
    
    if mcmc_data:
        mcmc_df = pd.DataFrame(mcmc_data)
        print(f"\nMCMC Results (Seeds: {seed_list}):")
        final_mcmc = mcmc_df.groupby(['Seed', 'Fold'])['Relative Log-Likelihood'].last().reset_index()
        print(f"Mean final Relative LL: {final_mcmc['Relative Log-Likelihood'].mean():.4f}")
        print(f"Std final Relative LL: {final_mcmc['Relative Log-Likelihood'].std():.4f}")
        
        for seed in seed_list:
            seed_data = final_mcmc[final_mcmc['Seed'] == seed]
            print(f"  Seed {seed}: Mean = {seed_data['Relative Log-Likelihood'].mean():.4f}, "
                  f"Std = {seed_data['Relative Log-Likelihood'].std():.4f}")
    
    if dlfm_data:
        dlfm_df = pd.DataFrame(dlfm_data)
        print(f"\nDLFM Results (Seeds: {seed_list}):")
        final_dlfm = dlfm_df.groupby(['Seed', 'Fold'])['Relative Log-Likelihood'].last().reset_index()
        print(f"Mean final Relative LL: {final_dlfm['Relative Log-Likelihood'].mean():.4f}")
        print(f"Std final Relative LL: {final_dlfm['Relative Log-Likelihood'].std():.4f}")
        
        for seed in seed_list:
            seed_data = final_dlfm[final_dlfm['Seed'] == seed]
            print(f"  Seed {seed}: Mean = {seed_data['Relative Log-Likelihood'].mean():.4f}, "
                  f"Std = {seed_data['Relative Log-Likelihood'].std():.4f}")
    else:
        print("\nDLFM Results: Not available")


def print_timing_comparison():
    print("=== Running Time Comparison ===")
    
    mcmc_times = []
    dlfm_times = []
    mcmc_trial_counts = []
    dlfm_trial_counts = []
    
    mcmc_dir = os.path.join(ibl_dir, "mcmc", "795")
    for seed in seed_list:
        seed_mcmc_times = []
        seed_trial_counts = []
        for fold in range(n_folds):
            fold_dir = os.path.join(mcmc_dir, f"fold_{fold}")
            time_file = os.path.join(fold_dir, f"ibl_gibbs_PG_total_time_atseed{seed}_gibbs_{num_gibbs_samples}.npy")
            ll_file = os.path.join(fold_dir, f"ibl_gibbs_PG_LL_atseed{seed}_gibbs_{num_gibbs_samples}.npy")
            if os.path.exists(time_file) and os.path.exists(ll_file):
                time_val = np.load(time_file)
                ll_data = np.load(ll_file)
                n_trials = ll_data.shape[0]
                seed_mcmc_times.append(time_val)
                seed_trial_counts.append(n_trials)
                print(f"MCMC - Seed {seed}, Fold {fold}: {time_val:.2f} seconds ({time_val/60:.2f} minutes), {n_trials} trials")
        
        if seed_mcmc_times:
            mcmc_times.extend(seed_mcmc_times)
            mcmc_trial_counts.extend(seed_trial_counts)
    
    dlfm_dir = os.path.join(ibl_dir, "dlfm", "795")
    for seed in seed_list:
        seed_dlfm_times = []
        seed_dlfm_trial_counts = []
        for fold in range(n_folds):
            fold_dir = os.path.join(dlfm_dir, f"fold_{fold}")
            time_file = os.path.join(fold_dir, f"ibl_dlfm_total_time_atseed{seed}.npy")
            ll_file = os.path.join(fold_dir, f"ibl_dlfm_lls_atseed{seed}.npy")
            if os.path.exists(time_file) and os.path.exists(ll_file):
                time_val = np.load(time_file)
                ll_data = np.load(ll_file)
                n_trials = ll_data.shape[0]
                seed_dlfm_times.append(time_val)
                seed_dlfm_trial_counts.append(n_trials)
                print(f"MCP - Seed {seed}, Fold {fold}: {time_val:.2f} seconds ({time_val/60:.2f} minutes), {n_trials} trials")
        
        if seed_dlfm_times:
            dlfm_times.extend(seed_dlfm_times)
            dlfm_trial_counts.extend(seed_dlfm_trial_counts)
    
    print("\n=== Timing Summary Statistics ===")
    
    if mcmc_times:
        avg_mcmc_time = np.mean(mcmc_times)
        std_mcmc_time = np.std(mcmc_times)
        avg_mcmc_trials = np.mean(mcmc_trial_counts)
        print(f"MCMC (Seeds: {seed_list}):")
        print(f"  Average time: {avg_mcmc_time:.2f} ± {std_mcmc_time:.2f} seconds ({avg_mcmc_time/60:.2f} ± {std_mcmc_time/60:.2f} minutes)")
        print(f"  Average trials: {avg_mcmc_trials:.0f}")
        print(f"  Time per trial: {avg_mcmc_time/avg_mcmc_trials:.4f} seconds/trial")
        
        for seed in seed_list:
            seed_times = [mcmc_times[i] for i in range(len(mcmc_times)) if i // n_folds == seed_list.index(seed)]
            if seed_times:
                print(f"    Seed {seed}: {np.mean(seed_times):.2f} ± {np.std(seed_times):.2f} seconds")
    
    if dlfm_times:
        avg_dlfm_time = np.mean(dlfm_times)
        std_dlfm_time = np.std(dlfm_times)
        avg_dlfm_trials = np.mean(dlfm_trial_counts)
        print(f"\nDLFM (Seeds: {seed_list}):")
        print(f"  Average time: {avg_dlfm_time:.2f} ± {std_dlfm_time:.2f} seconds ({avg_dlfm_time/60:.2f} ± {std_dlfm_time/60:.2f} minutes)")
        print(f"  Average trials: {avg_dlfm_trials:.0f}")
        print(f"  Time per trial: {avg_dlfm_time/avg_dlfm_trials:.4f} seconds/trial")
        
        for seed in seed_list:
            seed_times = [dlfm_times[i] for i in range(len(dlfm_times)) if i // n_folds == seed_list.index(seed)]
            if seed_times:
                print(f"    Seed {seed}: {np.mean(seed_times):.2f} ± {np.std(seed_times):.2f} seconds")
    
    # Speedup comparison
    if mcmc_times and dlfm_times:
        avg_mcmc = np.mean(mcmc_times)
        avg_dlfm = np.mean(dlfm_times)
        speedup = avg_mcmc / avg_dlfm
        print(f"\n=== Speedup Analysis ===")
        print(f"MCP is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than MCMC")
        print(f"MCMC/MCP time ratio: {speedup:.2f}")
        
        # Per-trial speedup
        mcmc_per_trial = avg_mcmc / np.mean(mcmc_trial_counts)
        dlfm_per_trial = avg_dlfm / np.mean(dlfm_trial_counts)
        per_trial_speedup = mcmc_per_trial / dlfm_per_trial
        print(f"Per-trial speedup: {per_trial_speedup:.2f}x")
    
    print("=" * 50)


if __name__ == "__main__":
    print_summary_stats()
    print_timing_comparison()
    # plot_mcmc_only()
    # plot_dlfm_only()
    plot_comprehensive_comparison()
