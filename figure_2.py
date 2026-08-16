# Simulation and plot for Figure 2 of the main text.
#
# Two groups of n/2 observations with low-rank covariances sharing a common
# latent basis. Prediction risk against p/n, one curve per sweep value rho.
# Dotted vertical lines mark the predicted switching thresholds.
#
# Usage:
#     python figure_2.py                 # paper settings, writes latent_private_sweep_model1.pdf

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import jax
import jax.numpy as jnp


# ============================ paper configuration ============================

SEED = 0
ORTHOGONAL_SEED = 314159         # fixes the random latent basis U_d
D_LIST = range(10, 1001, 5)
RHO_LIST = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
GROUP_SIZES = (50, 50)
NUM_SIM = 100
A_DIM = 0.45                     # dim A = a d, private to group 1
C_DIM = 0.10                     # dim C = c d, shared by both groups
EIG_MODE = "ones"
OUT_PDF = "latent_private_sweep_model1.pdf"


# ============================ core simulation ============================

def make_data_fn(sqrt_dSig_i, sqrt_nSig_i, beta):
    def data_fn(key, sqrt_nSig):
        d = sqrt_dSig_i.shape[0]
        n = sqrt_nSig.shape[0]

        key, z_key, eps_key = jax.random.split(key, 3)
        z_i = jax.random.normal(z_key, (d, n), dtype=beta.dtype)
        eps_i = jax.random.normal(eps_key, (n,), dtype=beta.dtype)

        X_i = sqrt_dSig_i @ z_i @ sqrt_nSig
        y_i = X_i.T @ beta + eps_i
        return X_i, y_i

    def single_test_data_fn(key):
        # One fresh test observation from this group.
        return data_fn(key, jnp.eye(1, dtype=beta.dtype))

    def gram_rhs_fn(key):
        X_i, y_i = data_fn(key, sqrt_nSig_i)
        return X_i @ X_i.T, X_i @ y_i

    return jax.jit(single_test_data_fn), jax.jit(gram_rhs_fn)


def ridgeless_regression_risk(
    sqrt_dSig_list,
    sqrt_nSig_list,
    beta,
    num_sim=100,
    seed=0,
    rcond=1e-6,
):
    dshape_list = [sqrt_dSig.shape for sqrt_dSig in sqrt_dSig_list]
    assert all(len(shape) == 2 and shape[0] == shape[1] for shape in dshape_list)
    assert all(shape[0] == dshape_list[0][0] for shape in dshape_list)
    d = dshape_list[0][0]

    nshape_list = [sqrt_nSig.shape for sqrt_nSig in sqrt_nSig_list]
    assert all(len(shape) == 2 and shape[0] == shape[1] for shape in nshape_list)
    group_sizes = [shape[0] for shape in nshape_list]

    assert len(sqrt_dSig_list) == len(sqrt_nSig_list)
    assert beta.shape == (d,)

    fn_list = [
        make_data_fn(sqrt_dSig_i, sqrt_nSig_i, beta)
        for sqrt_dSig_i, sqrt_nSig_i in zip(sqrt_dSig_list, sqrt_nSig_list)
    ]
    data_fn_list = [fn[0] for fn in fn_list]
    gram_rhs_fn_list = [fn[1] for fn in fn_list]

    m = len(group_sizes)
    total_n = sum(group_sizes)
    probs = jnp.asarray(group_sizes, dtype=beta.dtype) / total_n

    key = jax.random.PRNGKey(seed)
    data_key, key = jax.random.split(key)

    gram_rhs_list = [
        jax.vmap(
            lambda i: gram_rhs_fn(
                jax.random.fold_in(jax.random.fold_in(data_key, i), j)
            )
        )(jnp.arange(num_sim))
        for j, gram_rhs_fn in enumerate(gram_rhs_fn_list)
    ]

    gram_list = jnp.array([gram_rhs[0] for gram_rhs in gram_rhs_list])
    rhs_list = jnp.array([gram_rhs[1] for gram_rhs in gram_rhs_list])

    gram_over_sim = gram_list.sum(axis=0) / total_n
    rhs_over_sim = rhs_list.sum(axis=0) / total_n

    hat_beta_over_sim = jax.vmap(
        lambda gram, rhs: jnp.linalg.lstsq(gram, rhs, rcond=rcond)[0]
    )(gram_over_sim, rhs_over_sim)

    group_key, test_key, key = jax.random.split(key, num=3)
    group_indices = jax.vmap(
        lambda i: jax.random.choice(
            jax.random.fold_in(group_key, i), m, p=probs
        )
    )(jnp.arange(num_sim))

    # Python-side indexing into data_fn_list is safer than indexing
    # a Python list by a JAX scalar.
    group_indices_host = np.asarray(group_indices).astype(int).tolist()

    data_new_over_sim = [
        data_fn_list[group_idx](jax.random.fold_in(test_key, sim_idx))
        for sim_idx, group_idx in enumerate(group_indices_host)
    ]

    x_new_over_sim = jnp.array([data_new[0] for data_new in data_new_over_sim])
    y_new_over_sim = jnp.array([data_new[1] for data_new in data_new_over_sim])

    return jax.vmap(
        lambda x_new, hat_beta, y_new: (x_new.T @ hat_beta - y_new) ** 2
    )(x_new_over_sim, hat_beta_over_sim, y_new_over_sim)


# ============================ latent covariance helpers ============================

def make_random_orthogonal(d, seed, dtype=np.float32):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d)).astype(dtype)
    Q, R = np.linalg.qr(A)

    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs[None, :]

    return jnp.asarray(Q, dtype=jnp.float32)


def block_eigenvalues(k, mode="ones", low=0.5, high=1.5):
    if k <= 0:
        return np.zeros((0,), dtype=np.float32)
    if mode == "ones":
        return np.ones(k, dtype=np.float32)
    if mode == "linear":
        return np.linspace(low, high, k, dtype=np.float32)
    raise ValueError(f"Unknown eigenvalue mode: {mode}")


def sqrt_factor_from_latent_diag(U, latent_eigs):
    latent_eigs = jnp.asarray(latent_eigs, dtype=U.dtype)
    return U * jnp.sqrt(jnp.maximum(latent_eigs, 0.0))[None, :]


def latent_uniform_beta(U):
    d = U.shape[0]
    beta_latent = jnp.ones(d, dtype=U.dtype) / np.sqrt(d)
    return U @ beta_latent


def pad_peaks(peaks, max_len=2):
    out = np.full(max_len, np.nan, dtype=np.float32)
    for i, peak in enumerate(peaks[:max_len]):
        out[i] = peak
    return out


# ============================ private-factor sweep ============================

def predicted_peaks_private_sweep(rho, a=A_DIM, c=C_DIM, tol=1e-12):
    r2 = rho + c

    if r2 < a - tol:
        return [1.0 / (2.0 * a), 1.0 / (2.0 * r2)]
    if abs(r2 - a) <= tol:
        return [1.0 / (2.0 * a)]
    return [1.0 / (a + c + rho)]


def make_private_sweep_covariances(d, rho, U, a=A_DIM, c=C_DIM, eig_mode=EIG_MODE):
    kA = int(round(a * d))
    kC = int(round(c * d))
    kB = int(round(rho * d))

    overflow = kA + kC + kB - d
    if overflow > 0:
        kB = max(0, kB - overflow)

    latent1 = np.zeros(d, dtype=np.float32)
    latent2 = np.zeros(d, dtype=np.float32)

    A_start = 0
    C_start = A_start + kA
    B_start = C_start + kC

    latent1[A_start:A_start + kA] = block_eigenvalues(kA, eig_mode)
    latent1[C_start:C_start + kC] = block_eigenvalues(kC, eig_mode)

    latent2[B_start:B_start + kB] = block_eigenvalues(kB, eig_mode)
    latent2[C_start:C_start + kC] = block_eigenvalues(kC, eig_mode)

    return [
        sqrt_factor_from_latent_diag(U, latent1),
        sqrt_factor_from_latent_diag(U, latent2),
    ]


def latent_private_sweep_model1(
    d_list=D_LIST,
    rho_list=RHO_LIST,
    group_sizes=GROUP_SIZES,
    num_sim=NUM_SIM,
    seed=SEED,
    orthogonal_seed=ORTHOGONAL_SEED,
    eig_mode=EIG_MODE,
):
    d_list = list(d_list)
    rho_list = list(rho_list)

    sqrt_nSig_list = [jnp.eye(n_i, dtype=jnp.float32) for n_i in group_sizes]

    # Cache U_d so all rho curves for the same d use the same latent basis.
    U_cache = {}

    risks_list_over_rho = []
    predicted_peaks = []

    for rho_idx, rho in enumerate(rho_list):
        risks_list = []
        predicted_peaks.append(pad_peaks(predicted_peaks_private_sweep(rho), max_len=2))

        for d in d_list:
            print(f"private sweep: rho={rho}, d={d}", flush=True)

            if d not in U_cache:
                U_cache[d] = make_random_orthogonal(d, seed=orthogonal_seed + d)
            U = U_cache[d]

            sqrt_dSig_list = make_private_sweep_covariances(
                d=d, rho=rho, U=U, eig_mode=eig_mode,
            )
            beta = latent_uniform_beta(U)

            risks = ridgeless_regression_risk(
                sqrt_dSig_list=sqrt_dSig_list,
                sqrt_nSig_list=sqrt_nSig_list,
                beta=beta,
                num_sim=num_sim,
                seed=seed + 100000 * rho_idx + d,
            )
            risks_list.append(np.asarray(risks))

        risks_list_over_rho.append(risks_list)

    return (
        np.asarray(d_list),
        np.asarray(rho_list),
        np.asarray(group_sizes),
        np.asarray(predicted_peaks),
        np.asarray(risks_list_over_rho),
    )


# ============================ plot ============================

def plot_sweep(d_list, rho_list, group_sizes, predicted_peaks, risks_list_over_rho, out_pdf):
    gamma_list = d_list / np.sum(group_sizes)

    color_list = [
        'tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple',
        'tab:brown', 'black', 'tab:cyan', 'tab:pink', 'tab:olive', 'crimson'
    ]
    linestyle_list = [
        'solid', 'solid', 'solid', 'dashed', 'dotted',
        'dashed', 'solid', 'solid', 'solid', 'solid', 'solid'
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
    panel_idxes = [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]]

    for ax, idxes in zip(axes, panel_idxes):
        for idx in idxes:
            risks_list = risks_list_over_rho[idx]
            rho = rho_list[idx]
            color = color_list[idx % len(color_list)]
            linestyle = linestyle_list[idx % len(linestyle_list)]

            mean_risks_list = risks_list.mean(axis=tuple(range(1, risks_list.ndim)))
            ste_risks_list = (
                risks_list.std(axis=tuple(range(1, risks_list.ndim)))
                / np.sqrt(risks_list.shape[1])
            )

            ax.plot(gamma_list, mean_risks_list, color=color,
                    label=f'rho={rho:.2f}', linestyle=linestyle)

            fill_color = np.array(colors.to_rgba(color))
            fill_color[3] *= 0.2
            ax.fill_between(gamma_list,
                            mean_risks_list - 2 * ste_risks_list,
                            mean_risks_list + 2 * ste_risks_list,
                            facecolor=fill_color)

            for peak in predicted_peaks[idx]:
                if np.isfinite(peak) and gamma_list[0] <= peak <= gamma_list[-1]:
                    ax.axvline(peak, color=color, linestyle=':', linewidth=1.0, alpha=0.6)

        ax.set_xticks(np.arange(0, 10.1, 0.5))
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.2)
        ax.set_ylabel('test error')

    axes[-1].set_xlabel(r'$p/n$')
    fig.suptitle('Latent dimension sweep')

    legend_by_label = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            legend_by_label.setdefault(label, handle)

    fig.legend(list(legend_by_label.values()), list(legend_by_label.keys()),
               loc='upper right', bbox_to_anchor=(1.02, 0.89))

    fig.savefig(out_pdf, dpi=100, bbox_inches='tight')
    print(f"saved {out_pdf}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=str, default=OUT_PDF)
    args = parser.parse_args()

    plot_sweep(*latent_private_sweep_model1(seed=args.seed), out_pdf=args.out)
