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

root = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(root, "output", "results_IOHMM", "mcmc")
os.makedirs(output_dir, exist_ok=True)

def iohmm_random_gibbs(seed, T, initial_inputs, K, true_iohmm, test_iohmm, input_list, method = 'gibbs', **kwargs):
    """ Randomly selecting inputs and using Gibbs sampling for fitting the model"""

    print("Random input selection for IO-HMMs; using "+str(method)+" for fitting the model")

    # Fixing random seed
    npr.seed(seed)
    # True parameters
    M = input_list.shape[1]
    true_pi0 = np.exp(true_iohmm.params[0])
    true_Ps = np.exp(true_iohmm.params[1])
    true_obsparams = np.reshape(true_iohmm.params[2], (K,M))

    # Observations for initial samples
    init_time_bins = len(initial_inputs[0])
    # Since we only have one sessiond
    latents, obs = true_iohmm.sample(init_time_bins, input=initial_inputs[0])
    observations = []
    observations.append(obs)
    zs = []
    zs.append(latents)

    # To keep track of inference after every time_step: 
    obsparams_list = np.empty((T+1, K, M))
    pi0_list = np.empty((T+1, K))
    Ps_list = np.empty((T+1, K, K))
    error_obsparams = np.empty((T+1))
    error_Ps = np.empty((T+1))
    posteriorcov = np.empty((T+1))

    # Run inference using the existing dataset, also returns the posterior over states which is useful for calcumating MI
    if method=='gibbs_parallel':
        obsparams_sampled, Ps_sampled, pi0_sampled, fit_ll, pzts_persample  = test_iohmm.fit(observations, inputs=initial_inputs, method=method, zs=zs, **kwargs ) 
    else:
        obsparams_sampled, Ps_sampled, pi0_sampled, fit_ll, pzts_persample  = test_iohmm.fit(observations, inputs=initial_inputs, method=method, **kwargs )
    
    # Store the model parameters
    obsparams_list[0] = np.mean(obsparams_sampled, axis=0)
    pi0_list[0] = np.mean(pi0_sampled, axis=0)
    Ps_list[0] = np.mean(Ps_sampled, axis=0)
    ravelled_obsparams = np.reshape(obsparams_sampled, (obsparams_sampled.shape[0],obsparams_sampled.shape[1]*obsparams_sampled.shape[2]))
    ravelled_Ps = np.reshape(Ps_sampled[:,:,:-1], (Ps_sampled.shape[0], (K)*(K-1)))
    params = np.hstack((ravelled_obsparams,ravelled_Ps))
    cov = np.cov(params, rowvar = False)
    posteriorcov[0] =  0.5*np.linalg.slogdet(cov)[1] 

    # To store selected inputs at each step
    selected_inputs = []
        
    inputs = initial_inputs

    for t in range(T):
        print("Computing parameters of IO-HMM using "+str(t+1+init_time_bins)+" samples")
        # Select next input from a list of possible input input
        index = np.random.choice(np.arange(len(input_list)))
        x_new = input_list[index,:]
        # Obtain output from the true model
        z_new, obs_new = true_iohmm.sample(T=1, input = np.reshape(np.array(x_new), (1,M)), prefix=(zs[0], observations[0]))
        # Append this to the list of inputs and outputs
        observations[0] = np.concatenate((observations[0], obs_new), axis=0)
        inputs[0] = np.concatenate((inputs[0],np.reshape(np.array(x_new), (1,M))),  axis=0)
        zs[0] = np.concatenate((zs[0], z_new))

        # Run inference using the now new dataset
        if method=="gibbs_parallel":
            obsparams_sampled, Ps_sampled, pi0_sampled, fit_ll, pzts_persample  = test_iohmm.fit(observations, inputs=inputs, method=method, zs=zs, **kwargs)
        else:
            obsparams_sampled, Ps_sampled, pi0_sampled, fit_ll, pzts_persample  = test_iohmm.fit(observations, inputs=inputs, method=method, **kwargs)

        perm = find_permutation(zs[0], test_iohmm.most_likely_states(observations[0], input=inputs[0]), K, K)
        test_iohmm.permute(perm)

        obsparams_sampled = obsparams_sampled[:,perm,:]
        Ps_sampled = Ps_sampled[:,perm,:]
        Ps_sampled = Ps_sampled[:,:,perm]
        pi0_sampled = pi0_sampled[:, perm]

        # Store the model parameters
        obsparams_list[t+1] = np.mean(obsparams_sampled, axis=0)
        Ps_list[t+1] = np.mean(Ps_sampled, axis=0)
        pi0_list[t+1] = np.mean(pi0_sampled, axis=0)
        ravelled_obsparams = np.reshape(obsparams_sampled, (obsparams_sampled.shape[0],obsparams_sampled.shape[1]*obsparams_sampled.shape[2]))
        ravelled_Ps = np.reshape(Ps_sampled[:,:,:-1], (Ps_sampled.shape[0], (K)*(K-1)))
        params = np.hstack((ravelled_obsparams,ravelled_Ps))
        cov = np.cov(params, rowvar = False)
        posteriorcov[t+1] =  0.5*np.linalg.slogdet(cov)[1]  

        # To store selected inputs at each step
        selected_inputs.append(x_new)

    return pi0_list, Ps_list, obsparams_list, posteriorcov, selected_inputs


def run_iohmm_mcmc(args: argparse.Namespace):
    seed = args.seed
    np.random.seed(seed)
    num_gibbs_samples = args.num_gibbs_samples
    num_trials_per_sess = 100

    # Set the parameters of the GLM-HMM
    num_states = 3   # number of discrete states
    obs_dim = 1           # number of observed dimensions
    num_categories = 2    # number of categories for output
    input_dim = 2         # input dimensions

    # Make a GLM-HMM which will be our data generator
    true_iohmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                    observation_kwargs=dict(C=num_categories), transitions="standard")
    gen_weights = np.array([[[6, 1]], [[2, -3]], [[2, 3]]])
    gen_log_trans_mat = np.log(np.array([[[0.98, 0.01, 0.01], [0.05, 0.92, 0.03], [0.02, 0.03, 0.94]]]))
    true_iohmm.observations.params = gen_weights
    true_iohmm.transitions.params = gen_log_trans_mat
    gen_trans_mat = np.exp(gen_log_trans_mat)[0]

    num_sess = 1 # number of example sessions
    initial_trials = 100
    initial_inputs = np.ones((num_sess, initial_trials, input_dim)) # initialize inpts array
    stim_vals = np.arange(-5,5,step=0.01).tolist() # Stimuli values 
    initial_inputs[:,:,0] = np.random.choice(stim_vals, (num_sess, initial_trials)) # generate random sequence of potential inputs
    initial_inputs = list(initial_inputs) #convert inpts to correct format

    stimuli_list = np.ones((len(stim_vals), input_dim))
    stimuli_list[:,0] = stim_vals # list of all potential inputs

    # Generate a sequence of latents and choices for each session
    true_latents, true_choices = [], []
    for sess in range(num_sess):
        true_z, true_y = true_iohmm.sample(initial_trials, input=initial_inputs[sess])
        true_latents.append(true_z)
        true_choices.append(true_y)

    # Create a new test iohmm
    test_iohmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                        observation_kwargs=dict(C=num_categories), transitions="standard")
    # Reshaping weights to compute error later
    true_weights = np.reshape(gen_weights, (num_states, input_dim))

    # Train iohmm using random sampling
    method = args.fitting_method
    if method=='gibbs':
        pi0_list, Ps_list, weights_list, post_cov, selected_inputs = iohmm_random_gibbs(seed, num_trials_per_sess, initial_inputs, num_states, true_iohmm, test_iohmm, stimuli_list, method = "gibbs", num_iters = num_gibbs_samples, burnin = args.num_gibbs_burnin)
    elif method=='gibbs_PG':
       pi0_list, Ps_list, weights_list, post_cov, selected_inputs = iohmm_random_gibbs(seed, num_trials_per_sess, initial_inputs, num_states, true_iohmm, test_iohmm, stimuli_list, method = "gibbs", polyagamma=True, num_iters = num_gibbs_samples, burnin = args.num_gibbs_burnin)
    elif method=='gibbs_parallel':
        pi0_list, Ps_list, weights_list, post_cov, selected_inputs = iohmm_random_gibbs(seed, num_trials_per_sess, initial_inputs, num_states, true_iohmm, test_iohmm, stimuli_list, method = "gibbs_parallel")
    
    error_weights = np.linalg.norm(np.linalg.norm(weights_list-true_weights, axis=1), axis=1)
    error_Ps = np.linalg.norm(np.linalg.norm(Ps_list-gen_trans_mat, axis=1), axis=1)
    np.save(os.path.join(output_dir, "random_"+method+"_weights_atseed"+str(seed)+"_gibbs_"+str(num_gibbs_samples)+".npy"), weights_list)
    np.save(os.path.join(output_dir, "random_"+method+"_errorinweights_atseed"+str(seed)+"_gibbs_"+str(num_gibbs_samples)+".npy"), error_weights)
    np.save(os.path.join(output_dir, "random_"+method+"_Ps_atseed"+str(seed)+"_gibbs_"+str(num_gibbs_samples)+".npy"), Ps_list)
    np.save(os.path.join(output_dir, "random_"+method+"_errorinPs_atseed"+str(seed)+"_gibbs_"+str(num_gibbs_samples)+".npy"), error_Ps)
    np.save(os.path.join(output_dir, "random_"+method+"_posteriorcovariance_atseed"+str(seed)+"_gibbs_"+str(num_gibbs_samples)+".npy"), post_cov)
    np.save(os.path.join(output_dir, "random_"+method+"_selectedinputs_atseed"+str(seed)+"_gibbs_"+str(num_gibbs_samples)+".npy"), selected_inputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run IOHMM mcmc experiments')
    parser.add_argument('--seed', type=int, default='1',
                        help='Enter random seed')
    parser.add_argument('--fitting_method', type=str, default='gibbs_PG',
                        help='choose one of gibbs/gibbs_parallel/gibbs_PG')     
    parser.add_argument('--num_gibbs_samples', type=int, default='400')
    parser.add_argument('--num_gibbs_burnin', type=int, default='100')

    args = parser.parse_args()
    
    run_iohmm_mcmc(args)