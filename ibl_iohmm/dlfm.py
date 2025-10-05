import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cvxpy as cp
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import argparse

from scipy.special import logsumexp
from ssm.util import find_permutation


class HMM4eval:
    def __init__(self, num_states, weights=None, transition_matrix=None, pi0=None):
        self.K = num_states
        self.weights = weights  # Shape: (K, n_features)
        self.P_tr = transition_matrix  # Shape: (K, K)
        if pi0 is None:
            self.pi0 = np.ones(self.K) / self.K  # Uniform initial distribution
        else:
            self.pi0 = pi0
    
    def log_likelihood(self, observations, inputs):
        assert self.weights is not None and self.P_tr is not None, "set ws and Ps"
        T = len(observations)
        eps = 1e-10
        
        # Using p(y=1|x,z_k) = 1 / (1 + exp(-x * theta_k))
        log_emissions = np.zeros((T, self.K))
        for k in range(self.K):
            logits = inputs @ self.weights[k]
            probs = 1 / (1 + np.exp(-logits))
            # Bernoulli log-likelihood
            log_emissions[:, k] = (observations * np.log(probs + eps) + 
                                  (1 - observations) * np.log(1 - probs + eps))
        
        # Forward algorithm
        log_alpha = np.zeros((T, self.K))
        log_alpha[0] = np.log(self.pi0 + eps) + log_emissions[0]
        for t in range(1, T):
            for j in range(self.K):
                # Using alpha_t = sum(alpha_{t-1} * a) * b
                log_alpha[t, j] = logsumexp(
                    log_alpha[t-1] + np.log(self.P_tr[:, j] + eps)
                ) + log_emissions[t, j]
        
        return logsumexp(log_alpha[-1])

sns.set_theme(style='ticks', font_scale=1.5)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = ['sans-serif']

# * * * parameters of the GLM-HMM * * *
num_factors = K = 3 # latent states
input_dim = M = 4

initial_trials = 100

# * * * os dirs * * *
root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root, "output", "ibl_data")
input_features = np.load(os.path.join(data_dir, "input_features.npy"))
observations = np.load(os.path.join(data_dir, "observations.npy"))
print(f"Loaded data: inputs {input_features.shape}, observations {observations.shape}")
additional_trials = T = observations.shape[0] - initial_trials
output_dir = os.path.join(root, "output", "results_IOHMM", "dlfm", str(T+1))
os.makedirs(output_dir, exist_ok=True)


# Plotting results
def plot_results(labels, z, m, coefs, thetas):
    fig, axs = plt.subplots(1, 2, figsize=(8, 3), width_ratios=(1.2, 1))

    axs[0].plot(labels, linestyle='dashed', color='k', linewidth=1, zorder=10)
    axs[0].plot(np.argmax(z.value, axis=-1), color='r', linewidth=2)

    inputs = np.linspace(-5, 5, m)
    inputs = np.vstack([inputs, np.ones(m)]).T
    for i in range(K):
        axs[1].plot(inputs[:, 0], 1 / (1 + np.exp(-inputs @ coefs[i])),
                    linestyle='dashed', color='k', zorder=10)
        axs[1].plot(inputs[:, 0], 1 / (1 + np.exp(-inputs @ thetas[i].value)))

    axs[0].set_xlabel('$t$')
    axs[0].set_ylabel('latent factor')
    axs[0].set_yticks([0, 1, 2])
    axs[0].set_yticklabels([1, 2, 3])

    axs[1].set_xlabel(r'$\bar{x}$')
    axs[1].set_ylabel(r'$1/(1 + \\exp(-x^T \\theta))$', fontsize=15)

    plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, "dlpm_results.png"), dpi=300)
    plt.show()

# Estimate transition probabilities
def get_transition_probabilities(z, m):
    p_tr_hat = np.zeros((K, K))
    z_hat = np.argmax(z, axis=-1)
    for zi in range(K):
        z_idx = np.where(z_hat == zi)[0]
        if len(z_idx) == 0: continue
        z_idx = np.delete(z_idx, np.where(z_idx == m - 1)[0])
        z_dest, nz_num = np.unique(z_hat[z_idx + 1], return_counts=True) # z_hat[z_idx + 1] = z_next
        for dest, count in zip(z_dest, nz_num): # TODO: explicit assignment
            p_tr_hat[zi, dest] = count / len(z_idx)

    # print("\nEstimated transition probabilities:")
    return p_tr_hat


def iohmm_dlfm(num_samples, features, observations, prev_z=None):
    xs = features  # ndarray: dataset features
    ys = observations  # ndarray: dataset observations
    m = xs.shape[0]  # int: number of samples in the dataset
    n = xs.shape[-1]
    assert m == num_samples, f"trial number mismatch {m} != {num_samples}"

    # Hyperparameters
    eps = 1e-4  # float: termination criterion

    # P-problem
    lbd_theta = 0.5  # regularization weight
    thetas = []  # list of cp.Variable objects: model parameters; here is weights
    r = []  # list of cp.Expression objects: loss functions
    for k in range(K):
        thetas.append(cp.Variable(n))
        r.append(-(cp.multiply(ys, xs @ thetas[-1]) - cp.logistic(xs @ thetas[-1])))

    ztil = cp.Parameter((m, K), nonneg=True)
    Pobj = cp.sum(cp.multiply(ztil, cp.vstack(r).T))
    Preg = lbd_theta * cp.sum(cp.norm2(cp.vstack(thetas), axis=1))  # cp.Expression: regularization on model parameters
    Pconstr = [
        thetas[0][0] <= 0,
        thetas[1][0] <= 0,
        thetas[2][0] <= 0,
    ]  # list of cp.Constraint objects: model parameter constraints
    # Pconstr = None
    Pprob = cp.Problem(cp.Minimize(Pobj + Preg), Pconstr)
    assert Pprob.is_dcp()

    # F-problem
    lbd_z = 1  # regularization weight
    rtil = cp.Parameter((K, m))
    z = cp.Variable((m, K))
    Fobj = cp.sum(cp.multiply(z, rtil.T))
    Freg = lbd_z * cp.sum(cp.kl_div(z[:-1], z[1:]))  # cp.Expression: regularization on latent factors
    Fconstr = [z >= 0, z <= 1, cp.sum(z, axis=1) == 1]
    Fprob = cp.Problem(cp.Minimize(Fobj + Freg), Fconstr)
    assert Fprob.is_dcp()

    # Solve, terminate when the F- and P-objective converge
    i = 0
    while True:
        i += 1
        if ztil.value is None:
            ztil.value = np.random.dirichlet(np.ones(K), size=m)
        else:
            ztil.value = np.abs(z.value)
        try:
            Pprob.solve(
                solver=cp.CLARABEL, 
                max_iter=1000, 
                tol_gap_abs=eps*0.1,
                tol_gap_rel=eps*0.1,
                reduced_tol_gap_abs=eps, 
                reduced_tol_gap_rel=eps
            )
        except cp.SolverError:
            try:
                Pprob.solve(solver=cp.SCS, max_iters=5000, eps=eps*0.1, normalize=True)
                print(f"Iteration {i}: <solver=SCS> P-problem value: {Pobj.value}.")
            except cp.SolverError:
                print(f"Iteration {i}: F-problem failed to solve.")
                continue

        rtil.value = cp.vstack(r).value
        try:
            Fprob.solve(
                solver=cp.CLARABEL, 
                max_iter=1000, 
                tol_gap_abs=eps*0.1,
                tol_gap_rel=eps*0.1,
                reduced_tol_gap_abs=eps, 
                reduced_tol_gap_rel=eps
            )
        except cp.SolverError:
            try:
                Fprob.solve(solver=cp.SCS, max_iters=5000, eps=eps*0.1, normalize=True)
                print(f"Iteration {i}: <solver=SCS> F-problem value: {Fobj.value}.")
            except cp.SolverError:
                print(f"Iteration {i}: F-problem failed to solve.")
                continue

        if np.abs(Pobj.value - Fobj.value) < eps or i > 150:
            break
    print(f"Iteration {i}: P-problem value: {Pobj.value}, F-problem value: {Fobj.value}, gap: {np.abs(Pobj.value - Fobj.value)}.")

    thetas_val = np.array([theta.value for theta in thetas])
    z_val = z.value
    if prev_z is not None:
        perm = find_permutation(np.argmax(prev_z, axis=-1), np.argmax(z.value, axis=-1)[:-1], K, K)
        thetas_val = thetas_val[perm, :]
        z_val = z_val[:, perm]

    return thetas_val, z_val


def iohmm_dlfm_real_data(initial_inputs, initial_observations, remaining_inputs, remaining_observations, test_inputs, test_observations):
    M = initial_inputs.shape[1]
    T = len(remaining_inputs)
    
    features = initial_inputs.copy()
    observations_data = initial_observations.copy()
    theta_list = []
    p_tr_hat_list = []
    ll_list = []

    # init a iohmm for log-likelihood computation
    eval_iohmm = HMM4eval(num_factors)

    num_samples = len(initial_observations)
    thetas_val, z_val = iohmm_dlfm(num_samples, features, observations_data)
    theta_list.append(thetas_val)
    p_tr_hat_list.append(get_transition_probabilities(z_val, num_samples))
    
    for t in range(T):
        print(f"Processing trial {t+1}/{T}")
        prev_z = z_val
        # Add next real data point
        x_new = remaining_inputs[t]
        obs_new = remaining_observations[t]
        
        features = np.vstack([features, x_new])
        observations_data = np.append(observations_data, obs_new)
        
        num_samples = len(observations_data)
        thetas_val, z_val = iohmm_dlfm(num_samples, features, observations_data, prev_z)
        ptr_hat = get_transition_probabilities(z_val, num_samples)
        eval_iohmm.weights = thetas_val
        eval_iohmm.P_tr = ptr_hat
        test_ll = eval_iohmm.log_likelihood(test_observations, inputs=test_inputs)

        theta_list.append(thetas_val)
        p_tr_hat_list.append(ptr_hat)
        ll_list.append(test_ll)

    return theta_list, p_tr_hat_list, ll_list


def run_n_fold_cv(input_features, observations, args):
    seed = args.seed
    total_trials = len(observations)
    n_folds = 3
    fold_size = total_trials // n_folds
    
    for fold in range(n_folds):
        print(f"\n=== Processing Fold {fold + 1}/{n_folds} ===")
        
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else total_trials

        test_inputs = input_features[test_start:test_end]
        test_observations = observations[test_start:test_end]
        
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
        initial_inputs = train_inputs[:initial_trials]
        initial_observations = train_observations[:initial_trials]
        remaining_inputs = train_inputs[initial_trials:]
        remaining_observations = train_observations[initial_trials:]

        start_time = time.time()
        theta_list, p_tr_hat_list, ll_list = \
            iohmm_dlfm_real_data(initial_inputs, initial_observations, remaining_inputs, remaining_observations, test_inputs, test_observations)
        end_time = time.time()
        total_time = end_time - start_time
        
        fold_output_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)
        np.save(os.path.join(fold_output_dir, f"ibl_dlfm_lls_atseed{seed}.npy"), ll_list)
        np.save(os.path.join(fold_output_dir, f"ibl_dlfm_weights_atseed{seed}.npy"), theta_list)
        np.save(os.path.join(fold_output_dir, f"ibl_dlfm_Ps_atseed{seed}.npy"), p_tr_hat_list)
        np.save(os.path.join(fold_output_dir, f"ibl_dlfm_total_time_atseed{seed}.npy"), total_time)
        print(f"Fold {fold + 1} completed. Time taken: {total_time:.2f} seconds")
    
    print(f"\nAll folds completed!")


def main(args):
    seed = args.seed
    np.random.seed(seed)
    
    run_n_fold_cv(input_features, observations, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run IOHMM dlfm experiments')
    parser.add_argument('--seed', type=int, default='1',
                        help='Enter random seed')
    args = parser.parse_args()
    main(args)