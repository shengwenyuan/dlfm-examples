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
cols_traces = ['#BE1F24', '#2E3192', '#1f77b4', '#ff7f0e', '#2ca02c']

root = os.path.dirname(os.path.abspath(__file__))
ibl_dir = os.path.join(root, "output", "results_IOHMM/mcmc/795")
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
n_folds = 4
num_gibbs_samples = 5000


def load_mcmc_data():
    """Load MCMC log-likelihood data from all folds"""
    null_ll_per_trial = np.log(0.5)
    ll_data = {}
    
    for fold in range(n_folds):
        fold_dir = os.path.join(ibl_dir, f"fold_{fold}")
        ll = np.load(os.path.join(fold_dir, f"ibl_gibbs_PG_LL_atseed0_gibbs_{num_gibbs_samples}.npy"))
        ps = np.load(os.path.join(fold_dir, f"ibl_gibbs_PG_Ps_atseed0_gibbs_{num_gibbs_samples}.npy"))
        n_trials = ps.shape[0]
        ll_bits = ll / (np.log(2) * n_trials)
        relative_ll = (ll / n_trials) - null_ll_per_trial
        
        ll_data[fold] = {
            'll': ll,
            'll_bits': ll_bits,
            'relative_ll': relative_ll,
            'n_trials': n_trials
        }
    
    return ll_data


def load_dlfm_data():
    """Load DLFM log-likelihood data from all folds (if available)"""
    # Check if DLFM results directory exists
    dlfm_dir = os.path.join(root, "output", "results_IOHMM/dlfm/795")
    if not os.path.exists(dlfm_dir):
        print(f"DLFM directory not found: {dlfm_dir}")
        return None
    
    null_ll_per_trial = np.log(0.5)
    ll_data = {}
    
    for fold in range(n_folds):
        fold_dir = os.path.join(dlfm_dir, f"fold_{fold}")
        if not os.path.exists(fold_dir):
            print(f"DLFM fold directory not found: {fold_dir}")
            continue
            
        # Try to find DLFM log-likelihood files
        ll_file = os.path.join(fold_dir, f"ibl_dlfm_LL_atseed0.npy")
        ps_file = os.path.join(fold_dir, f"ibl_dlfm_Ps_atseed0.npy")
        
        if os.path.exists(ll_file) and os.path.exists(ps_file):
            ll = np.load(ll_file)
            ps = np.load(ps_file)
            n_trials = ps.shape[0]
            ll_bits = ll / (np.log(2) * n_trials)
            relative_ll = (ll / n_trials) - null_ll_per_trial
            
            ll_data[fold] = {
                'll': ll,
                'll_bits': ll_bits,
                'relative_ll': relative_ll,
                'n_trials': n_trials
            }
    
    return ll_data if ll_data else None


def plot_mcmc_only():
    """Plot MCMC log-likelihood curves by folds"""
    mcmc_data = load_mcmc_data()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for fold in range(n_folds):
        data = mcmc_data[fold]
        trials = np.arange(len(data['relative_ll'])) + initial_trials
        # Apply smoothing
        relative_ll_smooth = np.convolve(data['relative_ll'], np.ones(5)/5, mode='valid')
        trials_smooth = trials[:len(relative_ll_smooth)]
        
        ax.plot(trials_smooth, relative_ll_smooth, 
                label=f'Fold {fold}', color=cols_traces[fold % len(cols_traces)], 
                linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Trial #')
    ax.set_ylabel('Relative Log-Likelihood')
    # ax.set_title('MCMC Log-Likelihood by Folds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, f"mcmc_ll_{n_folds}folds.png"), dpi=400)
    plt.show()


def plot_dlfm_only():
    """Plot DLFM log-likelihood curves by folds"""
    dlfm_data = load_dlfm_data()
    
    if dlfm_data is None:
        print("DLFM data not available. Skipping DLFM-only plot.")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for fold in range(n_folds):
        if fold not in dlfm_data:
            continue
        data = dlfm_data[fold]
        trials = np.arange(len(data['relative_ll'])) + initial_trials
        # Apply smoothing
        relative_ll_smooth = np.convolve(data['relative_ll'], np.ones(5)/5, mode='valid')
        trials_smooth = trials[:len(relative_ll_smooth)]
        
        ax.plot(trials_smooth, relative_ll_smooth, 
                label=f'Fold {fold}', color=cols_traces[fold % len(cols_traces)], 
                linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Trial #')
    ax.set_ylabel('Relative Log-Likelihood')
    ax.set_title('DLFM Log-Likelihood by Folds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, f"dlfm_ll_by_folds_{n_folds}folds.png"), dpi=400)
    plt.show()


def plot_comprehensive_comparison():
    """Plot comprehensive comparison of MCMC vs DLFM log-likelihood"""
    mcmc_data = load_mcmc_data()
    dlfm_data = load_dlfm_data()
    
    # Prepare data for seaborn
    all_data = []
    
    # Add MCMC data
    for fold in range(n_folds):
        data = mcmc_data[fold]
        trials = np.arange(len(data['relative_ll'])) + initial_trials
        # Apply smoothing
        relative_ll_smooth = np.convolve(data['relative_ll'], np.ones(5)/5, mode='valid')
        trials_smooth = trials[:len(relative_ll_smooth)]
        
        for trial, rel_ll in zip(trials_smooth, relative_ll_smooth):
            all_data.append({
                'trial #': trial,
                'Method': 'MCMC',
                'Fold': fold,
                'Relative Log-Likelihood': rel_ll
            })
    
    # Add DLFM data (if available)
    if dlfm_data is not None:
        for fold in range(n_folds):
            if fold not in dlfm_data:
                continue
            data = dlfm_data[fold]
            trials = np.arange(len(data['relative_ll'])) + initial_trials
            # Apply smoothing
            relative_ll_smooth = np.convolve(data['relative_ll'], np.ones(5)/5, mode='valid')
            trials_smooth = trials[:len(relative_ll_smooth)]
            
            for trial, rel_ll in zip(trials_smooth, relative_ll_smooth):
                all_data.append({
                    'trial #': trial,
                    'Method': 'DLFM',
                    'Fold': fold,
                    'Relative Log-Likelihood': rel_ll
                })
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if dlfm_data is not None:
        # Plot both methods
        sns.lineplot(x="trial #", y="Relative Log-Likelihood", hue="Method", 
                    data=df, ax=ax, palette=cols_traces[:2], linewidth=2, alpha=0.8)
        ax.set_title('MCMC vs DLFM Log-Likelihood Comparison')
    else:
        # Plot only MCMC
        sns.lineplot(x="trial #", y="Relative Log-Likelihood", 
                    data=df[df['Method'] == 'MCMC'], ax=ax, 
                    color=cols_traces[1], linewidth=2, alpha=0.8, label='MCMC')
        ax.set_title('MCMC Log-Likelihood (DLFM data not available)')
    
    ax.set_xlabel('Trial #')
    ax.set_ylabel('Relative Log-Likelihood')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, f"comprehensive_ll_comparison_{n_folds}folds.png"), dpi=400)
    plt.show()


def print_summary_stats():
    """Print summary statistics for log-likelihood"""
    mcmc_data = load_mcmc_data()
    dlfm_data = load_dlfm_data()
    
    print("=== Log-Likelihood Summary Statistics ===")
    print("\nMCMC Results:")
    for fold in range(n_folds):
        data = mcmc_data[fold]
        print(f"Fold {fold}: LL = {data['ll'][-1]:.2f}, "
              f"LL_bits = {data['ll_bits'][-1]:.2f}, "
              f"Relative_LL = {data['relative_ll'][-1]:.2f}")
    
    if dlfm_data is not None:
        print("\nDLFM Results:")
        for fold in range(n_folds):
            if fold in dlfm_data:
                data = dlfm_data[fold]
                print(f"Fold {fold}: LL = {data['ll'][-1]:.2f}, "
                      f"LL_bits = {data['ll_bits'][-1]:.2f}, "
                      f"Relative_LL = {data['relative_ll'][-1]:.2f}")
    else:
        print("\nDLFM Results: Not available")


if __name__ == "__main__":
    print_summary_stats()
    plot_mcmc_only()
    # plot_dlfm_only()
    # plot_comprehensive_comparison()
