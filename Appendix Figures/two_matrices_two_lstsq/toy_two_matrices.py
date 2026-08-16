# Simulation for Figure 7 of the appendix.

import argparse
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

SEED = 0
HERE = Path(__file__).resolve().parent


def make_data_fn(sqrt_dSig_i, sqrt_nSig_i, beta):
    def data_fn(key, sqrt_nSig = jnp.eye(1)):
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
    assert jnp.array([ (len(shape) == 2 and shape[0] == shape[1]) for shape in dshape_list], dtype=bool).prod()
    assert jnp.array([ shape[0] == dshape_list[0][0] for shape in dshape_list], dtype=bool).prod()
    d = dshape_list[0][0]

    nshape_list = [sqrt_nSig.shape for sqrt_nSig in sqrt_nSig_list]
    assert jnp.array([ (len(shape) == 2 and shape[0] == shape[1]) for shape in nshape_list], dtype=bool).prod()
    group_sizes = [shape[0] for shape in nshape_list]

    assert len(sqrt_dSig_list) == len(sqrt_nSig_list)
    assert beta.shape == (d,)

    fn_list = [ make_data_fn(sqrt_dSig_i, sqrt_nSig_i, beta) for sqrt_dSig_i, sqrt_nSig_i in zip(sqrt_dSig_list, sqrt_nSig_list) ]
    data_fn_list = [fn[0] for fn in fn_list]
    gram_rhs_fn_list = [fn[1] for fn in fn_list]

    m = len(group_sizes)
    total_n = sum(group_sizes)
    probs = jnp.asarray(group_sizes, dtype=beta.dtype) / total_n

    key = jax.random.PRNGKey(seed)
    data_key, key = jax.random.split(key)

    gram_rhs_list = [jax.vmap(
                        lambda i: gram_rhs_fn(jax.random.fold_in(jax.random.fold_in(data_key, i),j))
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

    group_key, d_key, key = jax.random.split(key, num=3)
    group_indices = jax.vmap(
        lambda i: jax.random.choice(jax.random.fold_in(group_key,i), m, p=probs)
    )(jnp.arange(num_sim))


    data_new_over_sim = [data_fn_list[idx](jax.random.fold_in(d_key, idx)) for idx in group_indices]
    x_new_over_sim = jnp.array([data_new[0] for data_new in data_new_over_sim])
    y_new_over_sim = jnp.array([data_new[1] for data_new in data_new_over_sim])

    return jax.vmap(
        lambda x_new, hat_beta, y_new: (x_new.T @ hat_beta - y_new) ** 2
    )(x_new_over_sim, hat_beta_over_sim, y_new_over_sim)


def two_matrices_two(folder, seed=SEED):
    d_list = range(10, 2001, 5)
    signal_ratio_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    group_sizes = [100, 100]
    sqrt_nSig_list = [jnp.eye(n_i) for n_i in group_sizes]

    Path(folder).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=f'{folder}/run.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

    risks_list_over_signal = []
    for signal_ratio in signal_ratio_list:
        risks_list = []
        for d in d_list:
            logging.log(logging.INFO, f'Processing signal_ratio={signal_ratio}, d={d}...')
            signal_size = round(d * signal_ratio)
            sqrt_dSig_list = [
                jnp.diag(jnp.concatenate([jnp.ones(round(d/2)), jnp.zeros(d - round(d/2))])),
                jnp.diag(jnp.concatenate([jnp.zeros(d - signal_size), jnp.ones(signal_size)])),
            ]
            beta = jnp.ones(d) / jnp.sqrt(d)

            risks = ridgeless_regression_risk(
                sqrt_dSig_list,
                sqrt_nSig_list,
                beta,
                num_sim=100,
                seed=seed,
            )
            risks_list.append(risks)
        risks_list_over_signal.append(risks_list)

    risks_list_over_signal = jnp.array(risks_list_over_signal)

    np.savez(f'{folder}/output',
             d_list=d_list,
             signal_ratio_list=signal_ratio_list,
             group_sizes=group_sizes,
             risks_list_over_signal=risks_list_over_signal
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    two_matrices_two(HERE / "log", seed=args.seed)
