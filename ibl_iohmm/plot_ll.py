import os
import numpy as np

root = os.path.dirname(os.path.abspath(__file__))
ibl_dir = os.path.join(root, "output", "results_IOHMM/mcmc/795")

num_states = K = 3   # number of discrete states = [engaged, disengaged, right/left-bias]
obs_dim = 1           # number of observed dimensions
num_categories = 2    # number of categories for output = [0, 1(rightward choice=_ibl_trials.choice=1)]
input_dim = 4         # input dimensions = [stimulus = contrastRight - contrastLeft, 
                                            #bias = 1, 
                                            #prev_choice = _ibl_trials.choice, 
                                            #prev_stimulus_side(win-stay, lose-switch) = prev_contrastR/L]
initial_trials = 100

n_folds = 4


null_ll_per_trial = np.log(0.5)
ll_list = []
ll_bits_list = []
ll_relative_list = []
for fold in range(n_folds):
    fold_dir = os.path.join(ibl_dir, f"fold_{fold}")
    ll = np.load(os.path.join(fold_dir, f"ibl_gibbs_PG_LL_atseed0_gibbs_5000.npy"))
    ps = np.load(os.path.join(fold_dir, f"ibl_gibbs_PG_Ps_atseed0_gibbs_5000.npy"))
    n_trials = ps.shape[0]
    ll_bits = ll / (np.log(2) * n_trials)
    relative_ll = (ll / n_trials) - null_ll_per_trial
    ll_list.append(ll)
    ll_bits_list.append(ll_bits)
    ll_relative_list.append(relative_ll)

for fold in range(n_folds):
    print(f"Fold {fold}: LL = {ll_list[fold]:.2f}, \
                         LL_bits = {ll_bits_list[fold]:.2f}, \
                         Relative_LL = {ll_relative_list[fold]:.2f}")
