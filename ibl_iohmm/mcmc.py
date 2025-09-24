import numpy as np
import numpy.random as npr
import random
import ssm
import time
import argparse
import multiprocessing as mp
import os
import scipy.stats as st

from ssm.input_selection import input_selection
from ssm.util import one_hot, find_permutation, permute_params

# * * * Set the parameters of the GLM-HMM * * *
num_states = K = 3    # number of discrete states = [engaged, disengaged, right/left-bias]
obs_dim = 1           # number of observed dimensions
num_categories = 2    # number of categories for output = [0, 1(rightward choice=_ibl_trials.choice=1)]
input_dim = 4         # input dimensions = [stimulus = contrastRight - contrastLeft, 
                      #                     bias = 1, 
                      #                     prev_choice = _ibl_trials.choice, 
                      #                     prev_stimulus_side(win-stay, lose-switch) = prev_contrastR/L]
initial_trials = 100

# * * * os dirs * * *
root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root, "output", "ibl_data")
input_features = np.load(os.path.join(data_dir, "input_features.npy"))
observations = np.load(os.path.join(data_dir, "observations.npy"))
print(f"Loaded data: inputs {input_features.shape}, observations {observations.shape}")
additional_trials = observations.shape[0] - initial_trials
output_dir = os.path.join(root, "output", "results_IOHMM", "mcmc", str(additional_trials+1))
os.makedirs(output_dir, exist_ok=True)


def iohmm_real_data_gibbs(
        initial_inputs, 
        initial_observations,
        remaining_inputs,
        remaining_observations,
        K, 
        test_iohmm, 
        test_observations, 
        test_inputs, 
        method = 'gibbs', 
        **kwargs
    ):
    """ Using real IBL data with Gibbs sampling for fitting the model"""

    print("Using real IBL data for IO-HMMs; using "+str(method)+" for fitting the model")

    M = initial_inputs[0].shape[1]
    T = len(remaining_inputs[0])
    t_step = 10

    # Use real observations for initial samples
    init_time_bins = len(initial_inputs[0])
    observations = []
    observations.append(initial_observations)

    # To keep track of inference after every time_step: 
    obsparams_list = np.empty((T+1, K, M))
    pi0_list = np.empty((T+1, K))
    Ps_list = np.empty((T+1, K, K))
    posteriorcov = np.empty((T+1))
    ll_list = np.empty((T+1))

    # Run inference using the initial dataset
    if method=='gibbs_parallel':
        obsparams_sampled, Ps_sampled, pi0_sampled, fit_ll, pzts_persample  = test_iohmm.fit(observations, inputs=initial_inputs, method=method, **kwargs ) 
    else:
        obsparams_sampled, Ps_sampled, pi0_sampled, fit_ll, pzts_persample  = test_iohmm.fit(observations, inputs=initial_inputs, method=method, **kwargs )
    
    # Store the model parameters
    obsparams_list[0] = np.mean(obsparams_sampled, axis=0)
    pi0_list[0] = np.mean(pi0_sampled, axis=0)
    Ps_list[0] = np.mean(Ps_sampled, axis=0)
    ravelled_obsparams = np.reshape(obsparams_sampled, (obsparams_sampled.shape[0], obsparams_sampled.shape[1] * obsparams_sampled.shape[2]))
    ravelled_Ps = np.reshape(Ps_sampled[:,:,:-1], (Ps_sampled.shape[0], (K) * (K-1)))
    params = np.hstack((ravelled_obsparams, ravelled_Ps))
    cov = np.cov(params, rowvar = False)
    posteriorcov[0] =  0.5*np.linalg.slogdet(cov)[1] 
    ll_list[0] = test_iohmm.log_likelihood(test_observations, inputs=test_inputs)

    # Process remaining data sequentially
    prev_most_likely_states = test_iohmm.most_likely_states(observations[0], input=initial_inputs[0])
    selected_inputs = []
    inputs = initial_inputs
    for t in range(0, T, t_step):
        print("Computing parameters of IO-HMM using "+str(t+1+init_time_bins)+" samples")
        # Get next real input and observation
        t_step = min(t_step, T-t)
        x_new = remaining_inputs[0][t: t+t_step]
        obs_new = remaining_observations[t: t+t_step].reshape(t_step, -1)
        
        # Append real data to the training set
        inputs[0] = np.concatenate((inputs[0], x_new.reshape(t_step, -1)), axis=0)
        observations[0] = np.concatenate((observations[0], obs_new), axis=0)

        # Run inference using the extended dataset
        if method=="gibbs_parallel":
            obsparams_sampled, Ps_sampled, pi0_sampled, fit_ll, pzts_persample  = test_iohmm.fit(observations, inputs=inputs, method=method, **kwargs)
        else:
            obsparams_sampled, Ps_sampled, pi0_sampled, fit_ll, pzts_persample  = test_iohmm.fit(observations, inputs=inputs, method=method, **kwargs)

        # Find permutation and permute model
        curr_most_likely_states = test_iohmm.most_likely_states(observations[0], input=inputs[0])
        # Use current states for permutation
        perm = find_permutation(prev_most_likely_states, curr_most_likely_states[:-t_step], K, K)
        test_iohmm.permute(perm)

        obsparams_sampled = obsparams_sampled[:,perm,:]
        Ps_sampled = Ps_sampled[:,perm,:]
        Ps_sampled = Ps_sampled[:,:,perm]
        pi0_sampled = pi0_sampled[:,perm]

        # Store the model parameters
        obsparams_list[t+1] = np.mean(obsparams_sampled, axis=0)
        Ps_list[t+1] = np.mean(Ps_sampled, axis=0)
        pi0_list[t+1] = np.mean(pi0_sampled, axis=0)
        ravelled_obsparams = np.reshape(obsparams_sampled, (obsparams_sampled.shape[0], obsparams_sampled.shape[1] * obsparams_sampled.shape[2]))
        ravelled_Ps = np.reshape(Ps_sampled[:,:,:-1], (Ps_sampled.shape[0], (K) * (K-1)))
        params = np.hstack((ravelled_obsparams, ravelled_Ps))
        cov = np.cov(params, rowvar = False)
        posteriorcov[t+1] = 0.5 * np.linalg.slogdet(cov)[1]
        ll_list[t+1] = test_iohmm.log_likelihood(test_observations, inputs=test_inputs)

        # To store selected inputs at each step
        selected_inputs.append(x_new)
        prev_most_likely_states = curr_most_likely_states

    return pi0_list, Ps_list, obsparams_list, posteriorcov, ll_list, selected_inputs


def run_n_fold_cv(input_features, observations, args):
    seed = args.seed
    num_gibbs_samples = args.num_gibbs_samples
    method = args.fitting_method
    
    total_trials = len(observations)
    n_folds = 3
    fold_size = total_trials // n_folds
    
    for fold in range(n_folds):
        print(f"\n=== Processing Fold {fold + 1}/{n_folds} ===")
        
        # Split data into train and test
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else total_trials
        
        # Test data for this fold
        test_inputs = input_features[test_start:test_end]
        test_observations = observations[test_start:test_end][:, np.newaxis]
        
        # Training data (all other folds)
        train_inputs = np.concatenate([
            input_features[:test_start], 
            input_features[test_end:]
        ], axis=0)
        train_observations = np.concatenate([
            observations[:test_start], 
            observations[test_end:]
        ], axis=0)
        
        print(f"Train: {train_inputs.shape}, Test: {test_inputs.shape}")
        
        # Further split training data into initial and remaining
        # initial_size = min(initial_trials, train_size // 2)
        
        initial_inputs = [train_inputs[:initial_trials]]
        initial_observations = train_observations[:initial_trials].reshape(-1, 1)
        remaining_inputs = [train_inputs[initial_trials:]]
        remaining_observations = train_observations[initial_trials:]

        # Make a new IOHMM for each fold
        test_iohmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                            observation_kwargs=dict(C=num_categories), transitions="standard")
        trans_init = 0.95 * np.eye(K) + \
           np.abs(np.random.multivariate_normal(mean=np.zeros(K*K), cov=0.05 * np.eye(K*K)).reshape(K, K))
        trans_init = trans_init / trans_init.sum(axis=1, keepdims=True)
        # test_iohmm.transitions.params = np.log(trans_init[np.newaxis, :])

        start_time = time.time()
        if method=='gibbs':
            pi0_list, Ps_list, weights_list, post_cov, ll_list, selected_inputs = \
                iohmm_real_data_gibbs(initial_inputs, initial_observations, remaining_inputs, remaining_observations, num_states, test_iohmm, test_observations, test_inputs, method = "gibbs", num_iters = num_gibbs_samples, burnin = args.num_gibbs_burnin)
        elif method=='gibbs_PG':
           pi0_list, Ps_list, weights_list, post_cov, ll_list, selected_inputs = \
            iohmm_real_data_gibbs(initial_inputs, initial_observations, remaining_inputs, remaining_observations, num_states, test_iohmm, test_observations, test_inputs, method = "gibbs", polyagamma=True, num_iters = num_gibbs_samples, burnin = args.num_gibbs_burnin)
        elif method=='gibbs_parallel':
            pi0_list, Ps_list, weights_list, post_cov, ll_list, selected_inputs = \
                iohmm_real_data_gibbs(initial_inputs, initial_observations, remaining_inputs, remaining_observations, num_states, test_iohmm, test_observations, test_inputs, method = "gibbs_parallel")
        end_time = time.time()
        total_time = end_time - start_time

        # Save results for this fold
        fold_output_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)
        np.save(os.path.join(fold_output_dir, f"ibl_{method}_LL_atseed{seed}_gibbs_{num_gibbs_samples}.npy"), ll_list)
        np.save(os.path.join(fold_output_dir, f"ibl_{method}_weights_atseed{seed}_gibbs_{num_gibbs_samples}.npy"), weights_list)
        np.save(os.path.join(fold_output_dir, f"ibl_{method}_Ps_atseed{seed}_gibbs_{num_gibbs_samples}.npy"), Ps_list)
        np.save(os.path.join(fold_output_dir, f"ibl_{method}_posteriorcovariance_atseed{seed}_gibbs_{num_gibbs_samples}.npy"), post_cov)
        np.save(os.path.join(fold_output_dir, f"ibl_{method}_total_time_atseed{seed}_gibbs_{num_gibbs_samples}.npy"), total_time)
        # np.save(os.path.join(fold_output_dir, f"ibl_{method}_selectedinputs_atseed{seed}_gibbs_{num_gibbs_samples}.npy"), selected_inputs)
        # np.save(os.path.join(fold_output_dir, f"test_inputs.npy"), test_inputs)
        # np.save(os.path.join(fold_output_dir, f"test_observations.npy"), test_observations)
        
        print(f"Fold {fold + 1} completed. Time taken: {total_time:.2f} seconds")
    
    print(f"\nAll folds completed!")


def main(args: argparse.Namespace):
    seed = args.seed
    np.random.seed(seed)
    npr.seed(seed)
    
    # Run 5-fold cross-validation
    run_n_fold_cv(input_features, observations, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run IOHMM mcmc experiments')
    parser.add_argument('--seed', type=int, default='0',
                        help='Enter random seed')
    parser.add_argument('--fitting_method', type=str, default='gibbs_PG',
                        help='choose one of gibbs/gibbs_parallel/gibbs_PG')     
    parser.add_argument('--num_gibbs_samples', type=int, default='2000')
    parser.add_argument('--num_gibbs_burnin', type=int, default='500')

    args = parser.parse_args()
    
    main(args)