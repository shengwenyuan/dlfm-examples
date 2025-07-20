import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cvxpy as cp
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from scipy.optimize import linear_sum_assignment
from ssm.util import find_permutation

sns.set_theme(style='ticks', font_scale=1.5)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = ['sans-serif']

# * * * parameters of the GLM-HMM * * *
num_factors = K = 3 # latent states
initial_trials = 100
additional_trials = T = 1000

# * * * os dirs * * *
root = os.path.dirname(os.path.abspath(__file__))
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


def iohmm_dlfm(num_samples, features, observations, labels):
    xs = features  # ndarray: dataset features
    ys = observations  # ndarray: dataset observations
    m = xs.shape[0]  # int: number of samples in the dataset
    n = xs.shape[-1]
    assert m == num_samples, f"trial number mismatch {m} != {num_samples}"

    # Hyperparameters
    eps = 1e-6  # float: termination criterion

    # P-problem
    # K = 3
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
        thetas[0][0] >= 0,
        thetas[1][0] >= 0,
        thetas[2][0] >= 0,
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
        Pprob.solve(reduced_tol_gap_abs=5e-4, reduced_tol_gap_rel=5e-4)

        rtil.value = cp.vstack(r).value
        # Fprob.solve()
        Fprob.solve(reduced_tol_gap_abs=5e-4, reduced_tol_gap_rel=5e-4)

        print(f"Iteration {i}: P-problem value: {Pobj.value}, F-problem value: {Fobj.value}, gap: {np.abs(Pobj.value - Fobj.value)}.")
        if np.abs(Pobj.value - Fobj.value) < eps or i > 300:
            break

    perm = find_permutation(labels, np.argmax(z.value, axis=-1), K, K)
    thetas_val = np.array([theta.value for theta in thetas])
    thetas_val = thetas_val[perm, :]
    z_val = z.value[:, perm]

    return thetas_val, z_val


def main(seed):
    # Generate dataset
    coefs = np.array([[6, 1], [2, -3], [2, 3]]) # weights
    p_tr = np.array([[0.98, 0.01, 0.01], [0.05, 0.92, 0.03], [0.02, 0.03, 0.95]]) # transition probabilities

    features = np.random.uniform(-5, 5, initial_trials)
    features = np.vstack([features, np.ones(initial_trials)]).T
    observations = np.zeros(initial_trials)
    labels = np.zeros(initial_trials, dtype=int)
    s = 0
    for i, feat in enumerate(features):
        observations[i] = 1 if np.random.uniform() < 1 / (1 + np.exp(-feat @ coefs[s])) else 0
        labels[i] = s
        s = np.random.choice(num_factors, p=p_tr[s])

    # Solve the IO-HMM with DLFM
    theta_list = []
    p_tr_hat_list = []
    for t in range(T + 1):
        num_samples = initial_trials + t
        thetas_val, z_val = iohmm_dlfm(num_samples, features, observations, labels)
        theta_list.append(thetas_val)
        p_tr_hat_list.append(get_transition_probabilities(z_val, num_samples))

        # update the dataset with a new sample
        feature = np.random.uniform(-5, 5, 1)
        feature = np.vstack([feature, np.ones(1)]).T
        features = np.append(features, feature, axis=0)
        observations = np.append(observations, 1 if np.random.uniform() < 1 / (1 + np.exp(-feature @ coefs[s])) else 0)
        labels = np.append(labels, s)
        s = np.random.choice(num_factors, p=p_tr[s])


    # Saving results
    error_weights = np.linalg.norm(np.linalg.norm(coefs - np.array(theta_list), axis=1), axis=1)
    error_Ps = np.linalg.norm(np.linalg.norm(p_tr - np.array(p_tr_hat_list), axis=1), axis=1)
    np.save(os.path.join(output_dir, "dlfm_errorinweights_atseed" + str(seed) + ".npy"), error_weights)
    np.save(os.path.join(output_dir, "dlfm_errorinPs_atseed" + str(seed) + ".npy"), error_Ps)


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    main(seed)