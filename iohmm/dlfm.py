import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cvxpy as cp
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os

root = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(root, "output", "results_IOHMM", "dlpm")
os.makedirs(output_dir, exist_ok=True)

sns.set_theme(style='ticks', font_scale=1.5)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = ['sans-serif']

np.random.seed(364)

# Generate dataset
num_samples = 1000
num_factors = 3
coefs = np.array([[-2, 0], [2, 6], [3, -5]])
p_tr = np.array([[0.9, 0.05, 0.05], [0.01, 0.98, 0.01], [0.03, 0.02, 0.95]])

features = np.random.uniform(-5, 5, num_samples)
features = np.vstack([features, np.ones(num_samples)]).T

observations = np.zeros(num_samples)
labels = np.zeros(num_samples, dtype=int)

s = 0
for i, feat in enumerate(features):
    observations[i] = 1 if np.random.uniform() < 1 / (1 + np.exp(-feat @ coefs[s])) else 0
    labels[i] = s
    s = np.random.choice(num_factors, p=p_tr[s])

# Problem data
xs = features  # ndarray: dataset features
ys = observations  # ndarray: dataset observations
m = xs.shape[0]  # int: number of samples in the dataset
n = xs.shape[-1]

# Hyperparameters
eps = 1e-6  # float: termination criterion

# P-problem
K = 3
lbd_theta = 0.5  # regularization weight
thetas = []  # list of cp.Variable objects: model parameters
r = []  # list of cp.Expression objects: loss functions
for k in range(K):
    thetas.append(cp.Variable(n))
    r.append(-(cp.multiply(ys, xs @ thetas[-1]) - cp.logistic(xs @ thetas[-1])))

ztil = cp.Parameter((m, K), nonneg=True)
Pobj = cp.sum(cp.multiply(ztil, cp.vstack(r).T))
Preg = lbd_theta * cp.sum(cp.norm2(cp.vstack(thetas), axis=1))  # cp.Expression: regularization on model parameters
Pconstr = [
    thetas[0][0] <= 0,
    thetas[1][0] >= 0,
    thetas[2][0] >= 0,
]  # list of cp.Constraint objects: model parameter constraints
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
    Pprob.solve()

    rtil.value = cp.vstack(r).value
    Fprob.solve()

    print(f"Iteration {i}: P-problem value: {Pobj.value}, F-problem value: {Fobj.value}, gap: {np.abs(Pobj.value - Fobj.value)}.")
    if np.abs(Pobj.value - Fobj.value) < eps:
        break

# Plotting results
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
plt.savefig(os.path.join(output_dir, "dlpm_results.png"), dpi=300)
plt.show()

# Estimate transition probabilities
p_tr_hat = np.zeros_like(p_tr)
z_hat = np.argmax(z.value, axis=-1)
for zi in range(K):
    z_idx = np.where(z_hat == zi)[0]
    z_idx = np.delete(z_idx, np.where(z_idx == m - 1)[0])
    _, nz_num = np.unique(z_hat[z_idx + 1], return_counts=True)
    p_tr_hat[zi] = nz_num / len(z_idx)
    
print("\nEstimated transition probabilities:")
print(p_tr_hat)