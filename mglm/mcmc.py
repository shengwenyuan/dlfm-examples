import argparse
import autograd.numpy.random as npr
import numpy as np 
import os
import time
from utils_mglm.mglms import MGLM 
# from mglm_random import mglm_random


parser = argparse.ArgumentParser(description='Run MLR experiments')
parser.add_argument('--seed', type=int, default='0',
                    help='Enter random seed')
args = parser.parse_args()
seed = args.seed
np.random.seed(seed)

# * * * Set parameters of MGLM * * *
# num_states = 2  # number of discrete states
num_states = 3  # number of discrete states
obs_dim = 1  # data dimension
# input_dim = 2 # input dimension
input_dim = 10 # input dimension
num_categories = 2 # binary output for now
sigma = 1.5

initial_T = 100
T = 1000

# * * * os dirs * * *
root = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(root, "output", "results_MGLM", "mcmc", str(T+1))
os.makedirs(output_dir, exist_ok=True)


def mglm_random(seed, T, initial_inputs, K, true_mglm, test_mglm, input_list, burnin = 150, n_iters=300):
    """ Random sampling for fitting the model"""
    print("Fitting MGLM using random sampling")

    # Fixing random seed
    npr.seed(seed)
    # True parameters
    M = input_list.shape[1]
    true_pi = true_mglm.params[0]
    true_ws = np.reshape(true_mglm.params[1], (K,M))

    # Observations for initial samples
    init_time_bins = initial_inputs.shape[0]
    zs, observations = true_mglm.sample_y(init_time_bins, inputs=initial_inputs)

    # To keep track of inference after every time_step: 
    # ws_list contains posterior means for w
    # pis_list contains mean of pis
    weights_list = np.empty((T+1, K, M))
    pis_list = np.empty((T+1, K))
    posteriorcov = np.empty((T+1))


    # Run inference using the existing dataset, also returns the posterior over states which is useful for calcumating MI
    weights_sampled, pis_sampled, ll = test_mglm.fit_gibbs(observations, initial_inputs, burnin=burnin, n_iters = n_iters)

    # Store the model parameters
    weights_list[0] = np.mean(weights_sampled, axis=0)
    pis_list[0] = np.mean(pis_sampled, axis=0)
    # Store posterior covariance over model parameters
    if K>1:
        posteriorcov[0] = np.log(np.abs(np.var(pis_sampled[:,0])))
        for k in range(K):
            posteriorcov[0] = posteriorcov[0] +  np.linalg.slogdet(np.cov(weights_sampled[:,k], rowvar=False))[1] 
    else:
        posteriorcov[0] = np.linalg.slogdet(np.cov(weights_sampled, rowvar=False))[1] 

    # To store selected inputs at each step
    selected_inputs = []
        
    inputs = initial_inputs
    for t in range(T):
        init_samples = len(initial_inputs)
        print("Computing parameters of MGLM using "+str(t+1+init_samples)+" samples")
        # Select next stimuli randomly
        index = np.random.choice(np.arange(len(input_list)))
        x_new = input_list[index,:]

        # Obtain output from the true model
        z_new, observation_new = true_mglm.sample_y(T=1, inputs = np.reshape(np.array(x_new), (1,M)))
        # Append this to the list of inputs and outputs
        observations = np.concatenate((observations, observation_new), axis=0)
        inputs = np.concatenate((inputs,np.reshape(np.array(x_new), (1,M))),  axis=0)
        zs = np.concatenate((zs, z_new))

        # Run inference using the now new dataset
        initialize_w = np.reshape(weights_list[t], (K, 1, M))
        weights_sampled, pis_sampled, ll = test_mglm.fit_gibbs(observations, inputs, burnin=burnin, n_iters = n_iters, initialize = [pis_list[t], initialize_w])
        
        # Permute
        pis = np.mean(pis_sampled, axis=0)
        ws = np.mean(weights_sampled, axis=0)
        # permuting
        if K==2:
            if np.abs(pis[0]-true_pi[0])>np.abs(pis[0]-true_pi[1]):
                ws[0], ws[1] = (ws[1]).copy(), (ws[0]).copy()
                pis[0], pis[1] = (pis[1]).copy(), (pis[0]).copy()

        # Store the model parameters
        weights_list[t+1] = ws
        pis_list[t+1] = pis
        
        if K>1:
            posteriorcov[t+1] = np.abs(np.var(pis_sampled[:,0]))
            for k in range(K):
                posteriorcov[t+1] = posteriorcov[t+1] +  np.linalg.slogdet(np.cov(weights_sampled[:,k], rowvar=False))[1]   
        else:
            posteriorcov[t+1] = np.linalg.slogdet(np.cov(weights_sampled, rowvar=False))[1] 


        # To store selected inputs at each step
        selected_inputs.append(x_new)

    return pis_list, weights_list, selected_inputs, posteriorcov


## Set parameters
# true_pi0 = np.array([0.6, 0.4])
# true_weights = np.array([[3,-6], [3, 6]])
true_pi0 = np.array([0.4, 0.3, 0.3])
true_weights = np.array([[-1.47, 0.07, 0.16, -2.02, 0.14, 0.33, 0.71, 0.80, 1.53, -0.26], 
                         [-0.12, 1.38, -1.25, 0.88, -0.80, 1.33, -1.43, -0.42, 0.90, -0.47],
                         [1.14, -1.33, 0.16, 0.23, -1.20, -0.90, 1.40, 0.98, -1.11, 0.60]])
true_weights_2 = np.reshape(true_weights, (num_states, num_categories-1, input_dim))
true_mglm = MGLM(K=num_states, D=obs_dim, M=input_dim, C=num_categories, prior_sigma=sigma)
true_mglm.params = [true_pi0, true_weights_2]

# List of possible inputs
stim_vals = np.arange(-10,10,step=0.01).tolist()
input_list = np.ones((len(stim_vals), input_dim))
input_list[:,0] = stim_vals
# Initial inputs
initial_inputs = np.ones((initial_T, input_dim)) # initialize inpts array
initial_inputs[:,0] = np.random.choice(stim_vals, initial_T) # generate random sequence of input

# Sample observations from true mixture of GLMs
zs, observations = true_mglm.sample_y(initial_T, initial_inputs)


# Train MGLM with random sampling-------------------------------------------------------------------------------------------------------------
test_mglm = MGLM(K=num_states, D=obs_dim, M=input_dim, C=num_categories, prior_sigma=sigma)

# Start timing
start_time = time.time()

pis_list, weights_list, selected_inputs, posteriorcov = mglm_random(seed, T, initial_inputs, num_states, true_mglm, test_mglm, input_list, burnin = 150, n_iters=300)

# End timing
end_time = time.time()
total_time = end_time - start_time

error_in_weights = np.linalg.norm(weights_list - true_weights, axis=(1,2))
error_in_pis = np.linalg.norm(pis_list - true_pi0, axis=1)
np.save(os.path.join(output_dir, "random_atseed"+str(seed) + "_weights.npy"), weights_list)
np.save(os.path.join(output_dir, "random_atseed"+str(seed) + "_error_in_weights.npy"), error_in_weights)
np.save(os.path.join(output_dir, "random_atseed"+str(seed) + "_pis.npy"), pis_list)
np.save(os.path.join(output_dir, "random_atseed"+str(seed) + "_error_in_pis.npy"), error_in_pis)
np.save(os.path.join(output_dir, "random_atseed"+str(seed) + "_posteriorcov.npy"), posteriorcov)
np.save(os.path.join(output_dir, "random_atseed"+str(seed) + "_selectedinputs.npy"), selected_inputs)
np.save(os.path.join(output_dir, "random_atseed"+str(seed) + "_total_time.npy"), total_time)
print(f"Total time taken for MGLM random sampling: {total_time:.2f} seconds")