# Simulation and plot for Figure 4 of the main text.
#
# Three full-rank groups of n/3 = 50 observations each. Group g has covariance
# Sigma_g = Q(theta_g) diag(a I_{p/2}, b I_{p/2}) Q(theta_g)^T with rotation
# angles theta = 0, pi/12, pi/6, so the covariances are positive definite and
# do not commute. The prediction risk still has a single peak at p/n = 1.
#
# Usage:
#     python figure_4.py                 # paper settings, writes three_rotated_covariances_model_1ii.pdf

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
A_EIG = 1.0                          # large eigenvalue of every group covariance
B_EIG = 0.1                          # small eigenvalue of every group covariance
ANGLE_DEGREES = (0.0, 15.0, 30.0)    # = 0, pi/12, pi/6
GROUP_SIZES = (50, 50, 50)
D_LIST = range(10, 1001, 10)
NUM_SIM = 100
OUT_PDF = "three_rotated_covariances_model_1ii.pdf"


# ============================ core simulation ============================

def make_data_fn(sqrt_dSig_i, sqrt_nSig_i, beta):
    def data_fn(key, sqrt_nSig=jnp.eye(1)):
        d = sqrt_dSig_i.shape[0]
        n = sqrt_nSig.shape[0]

        key, z_key, eps_key = jax.random.split(key, 3)

        z_i = jax.random.normal(z_key, (d, n), dtype=beta.dtype)
        eps_i = jax.random.normal(eps_key, (n,), dtype=beta.dtype)

        X_i = sqrt_dSig_i @ z_i @ sqrt_nSig
        y_i = X_i.T @ beta + eps_i

        return X_i, y_i

    def gram_rhs_fn(key):
        X_i, y_i = data_fn(key, sqrt_nSig_i)
        return X_i @ X_i.T, X_i @ y_i

    return jax.jit(data_fn), jax.jit(gram_rhs_fn)


def ridgeless_regression_risk(
    sqrt_dSig_list,
    sqrt_nSig_list,
    beta,
    num_sim=1,
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
    data_key, group_key, test_key = jax.random.split(key, 3)

    gram_rhs_list = [
        jax.vmap(
            lambda sim_idx: gram_rhs_fn(
                jax.random.fold_in(
                    jax.random.fold_in(data_key, sim_idx),
                    group_idx,
                )
            )
        )(jnp.arange(num_sim))
        for group_idx, gram_rhs_fn in enumerate(gram_rhs_fn_list)
    ]

    gram_list = jnp.array([gram_rhs[0] for gram_rhs in gram_rhs_list])
    rhs_list = jnp.array([gram_rhs[1] for gram_rhs in gram_rhs_list])

    gram_over_sim = gram_list.sum(axis=0) / total_n
    rhs_over_sim = rhs_list.sum(axis=0) / total_n

    hat_beta_over_sim = jax.vmap(
        lambda gram, rhs: jnp.linalg.lstsq(gram, rhs, rcond=rcond)[0]
    )(gram_over_sim, rhs_over_sim)

    group_indices = jax.vmap(
        lambda sim_idx: jax.random.choice(
            jax.random.fold_in(group_key, sim_idx),
            m,
            p=probs,
        )
    )(jnp.arange(num_sim))

    # A unique test key per simulation, so simulations that select the same
    # group do not reuse the same test point.
    group_indices_np = np.asarray(group_indices)

    data_new_over_sim = [
        data_fn_list[int(group_idx)](jax.random.fold_in(test_key, sim_idx))
        for sim_idx, group_idx in enumerate(group_indices_np)
    ]

    x_new_over_sim = jnp.array([data_new[0] for data_new in data_new_over_sim])
    y_new_over_sim = jnp.array([data_new[1] for data_new in data_new_over_sim])

    risks = jax.vmap(
        lambda x_new, hat_beta, y_new: (x_new.T @ hat_beta - y_new) ** 2
    )(x_new_over_sim, hat_beta_over_sim, y_new_over_sim)

    return risks.squeeze()


# ============================ rotated covariances ============================

def block_rotation(d, theta, dtype=jnp.float32):
    if d % 2 != 0:
        raise ValueError(
            "This block-rotation construction assumes d is even. "
            "Use an even d_list, for example range(10, 1001, 10)."
        )

    q = d // 2
    eye_q = jnp.eye(q, dtype=dtype)

    c = jnp.asarray(jnp.cos(theta), dtype=dtype)
    s = jnp.asarray(jnp.sin(theta), dtype=dtype)

    top = jnp.concatenate([c * eye_q, -s * eye_q], axis=1)
    bottom = jnp.concatenate([s * eye_q, c * eye_q], axis=1)

    return jnp.concatenate([top, bottom], axis=0)


def rotated_sqrt_covariance(d, theta, a=A_EIG, b=B_EIG, dtype=jnp.float32):
    if not (a > 0 and b > 0):
        raise ValueError("Both a and b must be positive for Model 1(ii).")

    if a == b:
        raise ValueError(
            "Need a != b. If a == b, Sigma(theta) is a scalar multiple "
            "of the identity and rotations do not create heterogeneity."
        )

    if d % 2 != 0:
        raise ValueError(
            "This construction assumes d is even. "
            "Use d_list = range(10, 1001, 10), or modify the block structure."
        )

    q = d // 2
    Q = block_rotation(d, theta, dtype=dtype)

    sqrt_diag = jnp.concatenate([
        jnp.sqrt(jnp.asarray(a, dtype=dtype)) * jnp.ones(q, dtype=dtype),
        jnp.sqrt(jnp.asarray(b, dtype=dtype)) * jnp.ones(q, dtype=dtype),
    ])

    return Q @ jnp.diag(sqrt_diag) @ Q.T


def three_rotated_covariances_model_1ii(
    a=A_EIG,
    b=B_EIG,
    angle_degrees=ANGLE_DEGREES,
    group_sizes=GROUP_SIZES,
    d_list=D_LIST,
    num_sim=NUM_SIM,
    seed=SEED,
    dtype=jnp.float32,
):
    angle_degrees = tuple(float(x) for x in angle_degrees)
    angle_radians = tuple(np.pi * x / 180.0 for x in angle_degrees)

    group_sizes = tuple(int(x) for x in group_sizes)
    sqrt_nSig_list = [jnp.eye(n_i, dtype=dtype) for n_i in group_sizes]

    d_list = list(d_list)
    risks_list = []

    for d in d_list:
        print(f"three rotated covariances: d={d}", flush=True)

        sqrt_dSig_list = [
            rotated_sqrt_covariance(d=d, theta=theta, a=a, b=b, dtype=dtype)
            for theta in angle_radians
        ]

        beta = jnp.ones(d, dtype=dtype) / jnp.sqrt(jnp.asarray(d, dtype=dtype))

        risks = ridgeless_regression_risk(
            sqrt_dSig_list=sqrt_dSig_list,
            sqrt_nSig_list=sqrt_nSig_list,
            beta=beta,
            num_sim=num_sim,
            seed=seed,
        )

        risks_list.append(np.asarray(risks))

    return np.asarray(d_list), np.asarray(group_sizes), np.asarray(risks_list)


# ============================ plot ============================

def plot_risk(d_list, group_sizes, risks_list, out_pdf):
    gamma_list = d_list / np.sum(group_sizes)

    # Expected shape is (num_d, num_sim). Squeeze away trailing singletons.
    risks_array = np.squeeze(np.asarray(risks_list))
    if risks_array.ndim == 1:
        risks_array = risks_array[:, None]
    elif risks_array.ndim > 2:
        risks_array = risks_array.reshape(risks_array.shape[0], -1)

    mean_risks = risks_array.mean(axis=1)
    ste_risks = risks_array.std(axis=1, ddof=1) / np.sqrt(risks_array.shape[1])

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    color = "tab:blue"

    ax.plot(gamma_list, mean_risks, color=color, linewidth=2)

    fill_color = np.array(colors.to_rgba(color))
    fill_color[3] *= 0.2
    ax.fill_between(gamma_list, mean_risks - 2 * ste_risks, mean_risks + 2 * ste_risks,
                    facecolor=fill_color, linewidth=0)

    ax.set_xlabel(r"$p/n$")
    ax.set_ylabel("test error")
    ax.set_title("Positive definite covariance matrices")

    ax.set_xticks(np.arange(0, max(gamma_list) + 0.5, 0.5))
    ax.set_xlim(gamma_list[0], gamma_list[-1])

    upper = np.nanpercentile(mean_risks + 2 * ste_risks, 99)
    ax.set_ylim(0, max(1.0, 1.1 * upper))

    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=100, bbox_inches="tight")
    print(f"saved {out_pdf}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=str, default=OUT_PDF)
    args = parser.parse_args()

    plot_risk(*three_rotated_covariances_model_1ii(seed=args.seed), out_pdf=args.out)
