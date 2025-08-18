import numpy as np
import cvxpy as cp
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

from scipy.optimize import linear_sum_assignment

sns.set_theme(style='ticks', font_scale=1.5)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = ['sans-serif']

# generate dataset
K = 2
initial_trials = 100
additional_trials = T = 1000
num_features = 2
sigma = 1.5

# * * * os dirs * * *
root = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(root, "output", "results_MGLM", "dlfm", str(T+1))
os.makedirs(output_dir, exist_ok=True)


def plot_params(thetas, components):
    fig, axs = plt.subplots(1, 1, figsize=(4, 3))
    # ground truth coefficients
    for i, comp in enumerate(components):
        axs.plot(comp['coef'], linestyle='dashed', color='k', zorder=10)
    # estimated coefficients
    for theta in thetas:
        axs.plot(theta.value, linewidth=1.5, marker='o')
    axs.set_xticks(np.arange(0, 10, 2))
    axs.set_xticklabels(np.arange(1, 11, 2))
    axs.set_xlabel('indices')
    axs.set_ylabel(r'$\theta$')
    plt.tight_layout()

def get_pi_probabilities(z, m):
    pi_probs = []
    for k in range(K):
        pi_k = np.sum(z[:, k]) / m
        pi_probs.append(pi_k)
    return np.array(pi_probs)

def _compute_state_overlap(z1, z2, K1=None, K2=None):
    assert z1.dtype == int and z2.dtype == int
    assert z1.shape == z2.shape
    assert z1.min() >= 0 and z2.min() >= 0

    K1 = z1.max() + 1 if K1 is None else K1
    K2 = z2.max() + 1 if K2 is None else K2

    overlap = np.zeros((K1, K2))
    for k1 in range(K1):
        for k2 in range(K2):
            overlap[k1, k2] = np.sum((z1 == k1) & (z2 == k2))
    return overlap

def find_permutation(z1, z2, K1=None, K2=None):
    overlap = _compute_state_overlap(z1, z2, K1=K1, K2=K2)
    K1, K2 = overlap.shape

    tmp, perm = linear_sum_assignment(-overlap)
    assert np.all(tmp == np.arange(K1)), "All indices should have been matched!"

    # Pad permutation if K1 < K2
    if K1 < K2:
        unused = np.array(list(set(np.arange(K2)) - set(perm)))
        perm = np.concatenate((perm, unused))

    return perm

def mglm_dlfm(features, observations, labels):
    xs = features  # ndarray: dataset features
    ys = observations  # ndarray: dataset observations
    m = xs.shape[0]  # int: number of samples in the dataset
    n = xs.shape[1]

    ### hyperparameters
    eps = 1e-6  # float: termination criterion TODO: raise?

    ### P-problem
    thetas = []  # list of cp.Variable objects: model parameters
    r = []  # list of cp.Expression objects: loss functions
    for k in range(K):
        thetas.append(cp.Variable(n))
        r.append(cp.square(xs @ thetas[-1] - ys))

    ztil = cp.Parameter((m, K), nonneg=True)
    Pobj = cp.sum(cp.multiply(ztil, cp.vstack(r).T))
    Preg = 0  # cp.Expression: regularization on model parameters
    Pconstr = [
        thetas[0][0] >= 0,
        thetas[0][1] >= 0,
    ]  # list of cp.Constraint objects: model parameter constraints TODO
    Pprob = cp.Problem(cp.Minimize(Pobj + Preg), Pconstr)
    assert Pprob.is_dcp()

    ### F-problem
    rtil = cp.Parameter((K, m))
    z = cp.Variable((m, K))
    Fobj = cp.sum(cp.multiply(z, rtil.T))
    Freg = 0  # cp.Expression: regularization on latent factors
    Fconstr = [z >= 0, z <= 1, cp.sum(z, axis=1) == 1]
    Fprob = cp.Problem(cp.Minimize(Fobj + Freg), Fconstr)
    assert Fprob.is_dcp()

    ### solve, terminate when the F- and P-objective converge
    i = 0
    while True:
        i += 1
        if ztil.value is None:
            ztil.value = np.random.dirichlet(np.ones(K), size=m)
        else:
            ztil.value = np.abs(z.value)
        Pprob.solve()
        # Pprob.solve(reduced_tol_gap_abs=5e-4, reduced_tol_gap_rel=5e-4)

        rtil.value = cp.vstack(r).value
        Fprob.solve()
        # Fprob.solve(reduced_tol_gap_abs=5e-4, reduced_tol_gap_rel=5e-4)

        print(f'Iteration {i}: P-problem value: {Pobj.value}, F-problem value: {Fobj.value}, gap: {np.abs(Pobj.value - Fobj.value)}.')
        if np.abs(Pobj.value - Fobj.value) < eps or i > 300: # TODO: set max iterations
            break

    # permuting
    perm = find_permutation(labels, np.argmax(z.value, axis=-1), K, K)
    thetas_val = np.array([t.value for t in thetas])
    thetas_val = thetas_val[perm, :]
    z_val = z.value[:, perm]

    acc = np.sum(np.argmax(z_val, axis=-1) == labels) / len(labels)
    print(f'Factor identification accuracy: {acc}.')

    return thetas_val, z_val

def main(seed):
    components = [
        {'coef': np.array([3, -6]), 'p': 0.6},
        {'coef': np.array([3, 6]), 'p': 0.4},
    ]

    features = np.random.uniform(-10, 10, (initial_trials, num_features))
    observations = np.zeros(initial_trials)
    labels = np.zeros(initial_trials, dtype=int)

    component_choices = np.random.choice(len(components), size=initial_trials, p=[c['p'] for c in components])
    for i, comp_idx in enumerate(component_choices):
        comp = components[comp_idx]
        observations[i] = comp['coef'] @ features[i] + np.random.normal(0, sigma)
        labels[i] = comp_idx
    
    # Solve the MGLM with DLFM
    theta_list = []
    pi_hat_list = []
    
    # Start timing for the entire optimization process
    start_time = time.time()
    
    for t in range(T + 1):
        num_samples = initial_trials + t
        print(f"\nUsing {num_samples} samples")

        thetas_val, z_val = mglm_dlfm(features[:num_samples], observations[:num_samples], labels[:num_samples])
        theta_list.append(thetas_val)
        pi_hat = get_pi_probabilities(z_val, num_samples)
        pi_hat_list.append(pi_hat)

        # update the dataset with a new sample
        feature = np.random.uniform(-10, 10, (1, num_features))
        features = np.append(features, feature, axis=0)
        comp_idx = np.random.choice(len(components), size=1, p=[c['p'] for c in components])
        observations = np.append(observations, components[comp_idx[0]]['coef'] @ feature[0] + np.random.normal(0, sigma))
        labels = np.append(labels, comp_idx[0])
    
    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    

    # Saving results
    error_weights = np.linalg.norm(np.array([comp['coef'] for comp in components]) - np.array(theta_list), axis=(1,2))
    error_pis_hat = np.linalg.norm(np.array([comp['p'] for comp in components]) - np.array(pi_hat_list), axis=1)
    np.save(os.path.join(output_dir, f"dlfm_error_in_weights_atseed{seed}.npy"), error_weights)
    np.save(os.path.join(output_dir, f"dlfm_error_in_pis_atseed{seed}.npy"), error_pis_hat)
    np.save(os.path.join(output_dir, f"dlfm_selectedinputs_atseed{seed}.npy"), features)
    np.save(os.path.join(output_dir, f"dlfm_total_time_atseed{seed}.npy"), total_time)
    print(f"Total time taken for MGLM with DLFM: {total_time:.2f} seconds")


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    main(seed)