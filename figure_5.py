# Simulation and plot for Figure 5 of the main text.
#
# The WMT19 English-German transformer embedding table (1024 columns) is
# augmented with 256 i.i.d. Gaussian columns, giving p = 1280. Ridge regression
# with a small fixed ridge is fitted on random subsets of the rows and the
# held-out test MSE is plotted against p / n_train. The two peaks sit at
# p/n = 1 (full-design interpolation) and p/n = p / 256 = 5 (the added block).
#
# Requires transformers and torch: the embedding table is downloaded from the
# Hugging Face hub on first run.
#
# Usage:
#     python figure_5.py                    # paper settings, writes embedding_plus_synthetic_gaussian.png

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ============================ paper configuration ============================

SEED = 0
HF_MODEL = "facebook/wmt19-en-de"

CENTER_EMBEDDING_BLOCK = True
NUM_NEW_COLS = 256                    # appended Gaussian columns
AUGMENTED_NORMAL_VARIANCE = 3.0       # entrywise variance of the appended columns

LAMBDA_EMPIRICAL = 1e-5               # absolute ridge on the augmented design
NOISE_VARIANCE_EMPIRICAL = 1.0
TEST_SIZE = 5000
N_REPEATS_EMPIRICAL = 20
DN_GRID = np.logspace(np.log10(0.2), np.log10(10.0), 200)

OUT_PNG = "embedding_plus_synthetic_gaussian.png"


# ============================ 1. load the embedding table ============================

def extract_embedding_matrix(module):
    params = list(module.named_parameters())
    candidates = [
        (name, parameter)
        for name, parameter in params
        if parameter.dim() == 2
        and "embed" in name.lower()
        and "position" not in name.lower()
    ]

    decoder = [item for item in candidates if "decoder" in item[0].lower()]
    encoder = [item for item in candidates if "encoder" in item[0].lower()]

    if decoder:
        name, parameter = max(decoder, key=lambda item: item[1].shape[0])
    elif encoder:
        name, parameter = max(encoder, key=lambda item: item[1].shape[0])
    elif candidates:
        name, parameter = max(candidates, key=lambda item: item[1].shape[0])
    else:
        two_dimensional = [(name, p) for name, p in params if p.dim() == 2]
        if not two_dimensional:
            raise ValueError("No two-dimensional parameter was found.")
        name, parameter = max(two_dimensional, key=lambda item: item[1].shape[0])

    print(f"[extract] using parameter {name!r} with shape {tuple(parameter.shape)}")
    return parameter.detach().cpu().numpy().astype(np.float64)


def load_embedding(hf_model):
    from transformers import FSMTForConditionalGeneration

    model = FSMTForConditionalGeneration.from_pretrained(hf_model)
    matrix = extract_embedding_matrix(model)

    if matrix.ndim != 2:
        raise ValueError(f"Expected a matrix, received shape {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError("The loaded matrix contains NaN or infinite values.")

    print(f"[load] model={hf_model!r}, shape={matrix.shape}, dtype={matrix.dtype}")
    return matrix


# ============================ 2. build the augmented design ============================

def build_augmented_design(embedding, seed=SEED):
    if AUGMENTED_NORMAL_VARIANCE < 0:
        raise ValueError("AUGMENTED_NORMAL_VARIANCE must be nonnegative.")

    if CENTER_EMBEDDING_BLOCK:
        embedding_block = embedding - embedding.mean(axis=0, keepdims=True)
    else:
        embedding_block = embedding.copy()

    rng = np.random.default_rng(seed)
    added_block = rng.normal(
        loc=0.0,
        scale=np.sqrt(AUGMENTED_NORMAL_VARIANCE),
        size=(embedding_block.shape[0], NUM_NEW_COLS),
    )

    design = np.concatenate([embedding_block, added_block], axis=1)
    n_all, p = design.shape

    print(f"embedding block: {embedding_block.shape}")
    print(f"added Gaussian block: {added_block.shape}")
    print(f"augmented design: {design.shape}")
    print(f"realized added-block entry variance: {np.var(added_block):.6g}")
    print(f"predicted full-design interpolation location: p/n = 1")
    print(f"predicted added-block location: p/n = {p / NUM_NEW_COLS:.6g}")

    return design


# ============================ 3. empirical ridge sweep ============================

def fit_ridge_by_svd(X_train, y_train, ridge):
    if ridge <= 0:
        raise ValueError("The ridge parameter must be strictly positive.")

    n_train = X_train.shape[0]
    scaled = X_train / np.sqrt(n_train)

    left_vectors, singular_values, right_vectors_transposed = np.linalg.svd(
        scaled, full_matrices=False,
    )

    transformed_response = left_vectors.T @ y_train / np.sqrt(n_train)
    coefficients = (
        singular_values / (singular_values**2 + ridge)
    ) * transformed_response

    return right_vectors_transposed.T @ coefficients


def sweep_empirical_test_error(
    design,
    beta_true,
    dn_grid=DN_GRID,
    n_repeats=N_REPEATS_EMPIRICAL,
    test_size=TEST_SIZE,
    ridge=LAMBDA_EMPIRICAL,
    noise_variance=NOISE_VARIANCE_EMPIRICAL,
    seed=SEED + 100_000,
):
    rng = np.random.default_rng(seed)
    n_rows, p = design.shape

    dn_actual = np.zeros(len(dn_grid))
    mse_mean = np.full(len(dn_grid), np.nan)
    mse_q05 = np.full(len(dn_grid), np.nan)
    mse_q95 = np.full(len(dn_grid), np.nan)

    noise_standard_deviation = np.sqrt(noise_variance)

    for index, target_ratio in enumerate(dn_grid):
        n_train = max(2, min(int(round(p / target_ratio)), n_rows - 1))
        n_test = min(test_size, n_rows - n_train)

        if n_test < 1:
            raise ValueError("Not enough rows remain for a disjoint test set.")

        dn_actual[index] = p / n_train

        test_mse_values = []
        for _ in range(n_repeats):
            chosen = rng.choice(n_rows, size=n_train + n_test, replace=False)
            train_indices = chosen[:n_train]
            test_indices = chosen[n_train:]

            X_train = design[train_indices]
            X_test = design[test_indices]

            y_train = X_train @ beta_true + rng.normal(
                loc=0.0, scale=noise_standard_deviation, size=n_train,
            )
            y_test = X_test @ beta_true + rng.normal(
                loc=0.0, scale=noise_standard_deviation, size=n_test,
            )

            beta_hat = fit_ridge_by_svd(X_train, y_train, ridge)
            test_mse_values.append(np.mean((X_test @ beta_hat - y_test) ** 2))

        values = np.asarray(test_mse_values, dtype=np.float64)
        mse_mean[index] = np.nanmean(values)
        mse_q05[index] = np.nanpercentile(values, 5)
        mse_q95[index] = np.nanpercentile(values, 95)

        print(f"p/n={dn_actual[index]:8.4f} n_train={n_train:6d} n_test={n_test:5d} "
              f"test_mse={mse_mean[index]:.6g} ({index + 1}/{len(dn_grid)})", flush=True)

    return dn_actual, mse_mean, mse_q05, mse_q95


# ============================ 4. plot ============================

def plot_test_error(dn_actual, mse_mean, mse_q05, mse_q95, out_png):
    plt.figure(figsize=(8, 5.5))

    plt.fill_between(dn_actual, mse_q05, mse_q95, alpha=0.2,
                     label="central 90% empirical range")
    plt.plot(dn_actual, mse_mean, marker="o", markersize=3, label="mean test MSE")

    plt.xscale("log")
    plt.xlabel("p / n_train")
    plt.ylabel("test MSE")
    plt.title("Empirical fixed-ridge regression test error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=100, bbox_inches="tight")
    print(f"saved {out_png}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--hf-model", type=str, default=HF_MODEL)
    parser.add_argument("--out", type=str, default=OUT_PNG)
    args = parser.parse_args()

    design = build_augmented_design(load_embedding(args.hf_model), seed=args.seed)

    p = design.shape[1]
    beta_true = np.ones(p, dtype=np.float64) / np.sqrt(p)
    assert np.isclose(np.linalg.norm(beta_true), 1.0)

    peak_data = sweep_empirical_test_error(
        design, beta_true, seed=args.seed + 100_000,
    )

    peak = int(np.nanargmax(peak_data[1]))
    print(f"empirical mean test MSE peak: p/n={peak_data[0][peak]:.6g}, "
          f"value={peak_data[1][peak]:.8g}")

    plot_test_error(*peak_data, out_png=args.out)
