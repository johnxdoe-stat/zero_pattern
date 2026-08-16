# Simulation and plot for Figure 1 of the main text.
#
# Held-out test MSE against gamma = p / n_train for an unaugmented design and
# for random within-block permutations of block size b = 2 and b = 3.
#
# Usage:
#     python figure_1.py                 # paper settings, writes test_mse_vs_gamma.pdf

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)   # must be set before any array work
import jax.numpy as jnp


# ============================ paper configuration ============================

SEED = 0
B_LIST = [2, 3]          # block / partition sizes
K = 5                    # augmented copies per sample
SIG_EPS = 0.1            # noise standard deviation
N_TRAIN = 100
N_TEST = 1000
SIM = 800                # Monte Carlo trials per gamma per case
CHUNK = 32               # trials vmapped at once; raise it on a large GPU
RCOND = 1e-10            # relative SVD cutoff for the ridgeless solve
GAMMA_MIN, GAMMA_MAX, GAMMA_STEP = 0.10, 12.0, 0.10
OUT_PDF = "test_mse_vs_gamma.pdf"


def gamma_grid(gamma_min, gamma_max, gamma_step):
    # Add half a step so the right endpoint survives floating point error.
    return np.round(np.arange(gamma_min, gamma_max + 0.5 * gamma_step, gamma_step), 6)


def gamma_to_d(g, n_train):
    return max(2, int(round(g * n_train)))


# ============================ augmentation primitives ============================

def block_permute_batch(V, b, k, key):
    n, d = V.shape
    nfull = (d // b) * b
    nblocks = nfull // b
    r = d - nfull
    pieces = []

    if nblocks > 0:
        key, kb = jax.random.split(key)
        bulk = V[:, :nfull].reshape(n, nblocks, b)
        u = jax.random.uniform(kb, (n, k, nblocks, b))
        perm = jnp.argsort(u, axis=-1)
        bulk_b = jnp.broadcast_to(bulk[:, None, :, :], (n, k, nblocks, b))
        permuted = jnp.take_along_axis(bulk_b, perm, axis=-1).reshape(n, k, nfull)
        pieces.append(permuted)

    if r > 0:
        key, kt = jax.random.split(key)
        tail = V[:, nfull:]
        u_t = jax.random.uniform(kt, (n, k, r))
        perm_t = jnp.argsort(u_t, axis=-1)
        tail_b = jnp.broadcast_to(tail[:, None, :], (n, k, r))
        permuted_tail = jnp.take_along_axis(tail_b, perm_t, axis=-1)
        pieces.append(permuted_tail)

    return jnp.concatenate(pieces, axis=-1)


def lstsq_min_norm(A, y, rcond):
    U, s, Vt = jnp.linalg.svd(A, full_matrices=False)
    cutoff = rcond * s[0]
    safe = jnp.where(s > cutoff, s, 1.0)
    s_inv = jnp.where(s > cutoff, 1.0 / safe, 0.0)
    return (Vt.T * s_inv) @ (U.T @ y)


# ============================ per-trial functions ============================

def make_unaug_trial(n_train, n_test, d, sig_eps, rcond):
    beta = jnp.full(d, 1.0 / np.sqrt(d))
    n_total = n_train + n_test

    def trial(key):
        kV, keps = jax.random.split(key, 2)
        V_all = jax.random.normal(kV, (n_total, d))
        eps_all = sig_eps * jax.random.normal(keps, (n_total,))
        y_all = V_all @ beta + eps_all

        bhat = lstsq_min_norm(V_all[:n_train], y_all[:n_train], rcond)
        return jnp.mean((V_all[n_train:] @ bhat - y_all[n_train:]) ** 2)

    return trial


def make_aug_trial(n_train, n_test, k, d, b, sig_eps, rcond):
    beta = jnp.full(d, 1.0 / np.sqrt(d))
    n_total = n_train + n_test

    def trial(key):
        kV, keps, kperm, knoise = jax.random.split(key, 4)
        V_all = jax.random.normal(kV, (n_total, d))
        eps_all = sig_eps * jax.random.normal(keps, (n_total,))

        V_train = V_all[:n_train]
        V_test = V_all[n_train:]
        y_test = V_test @ beta + eps_all[n_train:]

        # Each training sample is replaced by k within-block permuted copies.
        # The response noise is drawn independently for every copy.
        aug = block_permute_batch(V_train, b, k, kperm)
        signal = aug @ beta
        noise = sig_eps * jax.random.normal(knoise, (n_train, k))

        A = aug.reshape(n_train * k, d)
        y_aug = (signal + noise).reshape(n_train * k)
        bhat = lstsq_min_norm(A, y_aug, rcond)
        return jnp.mean((V_test @ bhat - y_test) ** 2)

    return trial


# ============================ Monte Carlo driver ============================

def collect(trial_closure, key, sim, chunk):
    fn = jax.jit(jax.vmap(trial_closure))
    n_chunks = (sim + chunk - 1) // chunk
    keys = jax.random.split(key, n_chunks * chunk)
    outs = [np.asarray(fn(keys[c * chunk:(c + 1) * chunk])) for c in range(n_chunks)]
    return np.concatenate(outs, axis=0)[:sim]


def mean_se(vals):
    vals = np.asarray(vals)
    mean = vals.mean(axis=0)
    if vals.shape[0] <= 1:
        return mean, np.zeros_like(mean)
    return mean, vals.std(axis=0, ddof=1) / np.sqrt(vals.shape[0])


# ============================ main ============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=str, default=OUT_PDF)
    args = parser.parse_args()

    gammas = gamma_grid(GAMMA_MIN, GAMMA_MAX, GAMMA_STEP)
    base = jax.random.PRNGKey(args.seed)

    unaug_mean = np.empty(len(gammas))
    unaug_se = np.empty(len(gammas))
    for gi, g in enumerate(gammas):
        d = gamma_to_d(g, N_TRAIN)
        key = jax.random.fold_in(jax.random.fold_in(base, 10_000), gi)
        vals = collect(make_unaug_trial(N_TRAIN, N_TEST, d, SIG_EPS, RCOND),
                       key, SIM, CHUNK)
        unaug_mean[gi], unaug_se[gi] = mean_se(vals)
        print(f"[unaugmented] gamma={g:6.3f} d={d:5d} test_mse={unaug_mean[gi]:.4e} "
              f"({gi + 1}/{len(gammas)})", flush=True)

    aug_mean, aug_se = {}, {}
    for b in B_LIST:
        aug_mean[b] = np.empty(len(gammas))
        aug_se[b] = np.empty(len(gammas))
        for gi, g in enumerate(gammas):
            d = gamma_to_d(g, N_TRAIN)
            key = jax.random.fold_in(jax.random.fold_in(base, 1000 * b), gi)
            vals = collect(make_aug_trial(N_TRAIN, N_TEST, K, d, b, SIG_EPS, RCOND),
                           key, SIM, CHUNK)
            aug_mean[b][gi], aug_se[b][gi] = mean_se(vals)
            print(f"[b={b}] gamma={g:6.3f} d={d:5d} test_mse={aug_mean[b][gi]:.4e} "
                  f"({gi + 1}/{len(gammas)})", flush=True)

    # ------------------------------ plot ------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    curves = [("unaugmented", unaug_mean, unaug_se)]
    curves += [(f"b={b}", aug_mean[b], aug_se[b]) for b in B_LIST]

    for label, mean, se in curves:
        line = ax.plot(gammas, mean, linewidth=1.8, label=label)[0]
        ax.fill_between(gammas, mean - 2.0 * se, mean + 2.0 * se,
                        alpha=0.18, color=line.get_color(), linewidth=0)

    ax.set_xlabel(r"$\gamma = p/n_{\mathrm{train}}$")
    ax.set_ylabel(r"held-out test MSE")
    ax.set_title("Multiple test error peak in augmented data")
    ax.set_ylim(0, 5)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
