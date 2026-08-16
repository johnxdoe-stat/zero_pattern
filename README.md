This repository reproduces every figure in the paper. Figures 1–5 in the main
text and Figures 6–15 in the appendix.

## Requirements

- Python 3.9 or newer
- `jax`, `numpy`, `matplotlib` — Figures 1–4 and all appendix figures
- `transformers`, `torch` — Figure 5 only
- `jupyter` — appendix plotting notebook only

```
pip install numpy matplotlib jax jupyter
pip install transformers torch          # only needed for Figure 5
```

JAX runs on CPU out of the box. The simulations reported in the paper were run
on GPUs; on CPU they are much slower.

## Repository layout

```
.
├── README.md
├── figure_1.py                 Figure 1
├── figure_2.py                 Figure 2
├── figure_3.py                 Figure 3
├── figure_4.py                 Figure 4
├── figure_5.py                 Figure 5
└── Appendix Figures/
    ├── plotter.ipynb           one cell per appendix figure
    ├── two_matrices_one/                                  Figure 6
    ├── two_matrices_two_lstsq/                            Figure 7
    ├── three_matrices/                                    Figure 8
    ├── two_matrices_diminishing_eval/                     Figure 9
    ├── two_matrices_diminishing_eval_beyond_one/          Figure 10
    ├── two_matrices_diminishing_eval_overlap/             Figure 11
    ├── two_matrices_diminishing_eval_overlap_beyond_one/  Figure 12
    ├── two_matrices_non_diag/                             Figure 13
    ├── two_matrices_non_diag_two/                         Figure 14
    └── two_matrices_non_diag_change_size/                 Figure 15
```

## Main text figures

Each script is self-contained, it runs its own simulation and then writes the
figure. No other file is needed.

```
python figure_1.py
python figure_2.py
python figure_3.py
python figure_4.py
python figure_5.py
```

| Figure | Script | Output file |
| --- | --- | --- |
| 1 | `figure_1.py` | `test_mse_vs_gamma.pdf` |
| 2 | `figure_2.py` | `latent_private_sweep_model1.pdf` |
| 3 | `figure_3.py` | `latent_private_sweep_model1_nongaussian.pdf` |
| 4 | `figure_4.py` | `three_rotated_covariances_model_1ii.pdf` |
| 5 | `figure_5.py` | `embedding_plus_synthetic_gaussian.png` |

Common flags:

| Flag | Scripts | Meaning |
| --- | --- | --- |
| `--seed` | all | Random seed. Default `0`. |
| `--out` | all | Output file name. |
| `--dist` | `figure_3.py` | Design entry law. Default `student_t3`; `gaussian`, `rademacher`, `uniform`, `laplace` and `student_t5` are also available. |
| `--hf-model` | `figure_5.py` | Hugging Face checkpoint. Default `facebook/wmt19-en-de`, downloaded on first run. |

## Appendix figures

These take two steps: run the simulation, then plot it.

**Step 1 — simulate.** Each subfolder holds one script that produces the data
for exactly one figure. Run it from inside its own folder:

```
cd "Appendix Figures/two_matrices_one"
python toy_two_matrices.py
```

This writes `log/output.npz` (and a progress file `log/run.log`) inside that
same subfolder. Every script accepts `--seed`, default `0`. Repeat for each
figure you want.

**Step 2 — plot.** From the `Appendix Figures` folder, open the notebook and run
the cell for the figure you want. Each cell is independent and is labelled with
its figure number on the first line.

```
cd "Appendix Figures"
jupyter notebook plotter.ipynb
```

| Figure | Subfolder | Script | Output file |
| --- | --- | --- | --- |
| 6 | `two_matrices_one` | `toy_two_matrices.py` | `two_matrices.pdf` |
| 7 | `two_matrices_two_lstsq` | `toy_two_matrices.py` | `two_matrices_reverse_full.pdf` |
| 8 | `three_matrices` | `toy_three_matrices.py` | `three_matrices.pdf` |
| 9 | `two_matrices_diminishing_eval` | `toy_diminishing_eval.py` | `two_matrices_diminishing_eval.pdf` |
| 10 | `two_matrices_diminishing_eval_beyond_one` | `toy_diminishing_eval.py` | `two_matrices_diminishing_eval_beyond_one.pdf` |
| 11 | `two_matrices_diminishing_eval_overlap` | `toy_diminishing_eval.py` | `two_matrices_diminishing_eval_overlap.pdf` |
| 12 | `two_matrices_diminishing_eval_overlap_beyond_one` | `toy_diminishing_eval.py` | `two_matrices_diminishing_eval_overlap_beyond_one.pdf` |
| 13 | `two_matrices_non_diag` | `toy_non_diag.py` | `two_matrices_non_diag.pdf` |
| 14 | `two_matrices_non_diag_two` | `toy_non_diag.py` | `two_matrices_non_diag_two.pdf` |
| 15 | `two_matrices_non_diag_change_size` | `toy_non_diag.py` | `two_matrices_non_diag_change_size.pdf` |

The notebook writes each PDF into the `Appendix Figures` folder.
