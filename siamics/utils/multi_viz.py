import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D

def tme_compare(base_dir, run_names, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    paths = [
        os.path.join(base_dir, run_name, "tme", "evaluate", "eval", "eval_res.npz")
        for run_name in run_names
    ]

    all_model_names     = []
    all_pred_props      = []
    all_true_props      = []
    all_cell_type_names = None

    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            if all_cell_type_names is None:
                all_cell_type_names = data["cell_type_names"]
            m = data["model_name"]
            model_name = m.item() if isinstance(m, np.ndarray) else m
            all_model_names.append(model_name)
            all_pred_props.append(data["pred_prop"])
            all_true_props.append(data["true_prop"])

    all_cell_type_names = [name.split('_', 1)[0] for name in all_cell_type_names]
    all_pred_props = np.stack(all_pred_props, axis=0)
    all_true_props = np.stack(all_true_props, axis=0)

    n_models, n_samples, n_celltypes = all_pred_props.shape

    # compute metrics
    all_cell_type_pccs  = np.zeros((n_models, n_celltypes))
    all_cell_type_rmses = np.zeros((n_models, n_celltypes))
    all_cell_type_cccs  = np.zeros((n_models, n_celltypes))
    all_cell_type_sccs  = np.zeros((n_models, n_celltypes))

    for i in range(n_models):
        for j in range(n_celltypes):
            pred = all_pred_props[i, :, j]
            true = all_true_props[i, :, j]

            if np.std(pred) == 0 or np.std(true) == 0:
                pcc = np.nan
            else:
                pcc, _ = pearsonr(pred, true)
            all_cell_type_pccs[i, j] = pcc

            all_cell_type_rmses[i, j] = np.sqrt(np.mean((pred - true) ** 2))

            if len(np.unique(pred)) < 2 or len(np.unique(true)) < 2:
                scc = np.nan
            else:
                scc, _ = spearmanr(pred, true)
            all_cell_type_sccs[i, j] = scc

            mean_pred = pred.mean()
            mean_true = true.mean()
            var_pred  = pred.var()
            var_true  = true.var()
            cov       = ((pred - mean_pred) * (true - mean_true)).mean()
            denom = var_pred + var_true + (mean_pred - mean_true) ** 2
            all_cell_type_cccs[i, j] = (2 * cov) / denom if denom != 0 else np.nan

    # cell type PCC heatmap
    cell_type_pccs = pd.DataFrame(
        data   = all_cell_type_pccs,
        index  = all_model_names,
        columns= all_cell_type_names
    )
    fig, ax = plt.subplots(figsize=(9, 2.5))

    heat = sns.heatmap(
        cell_type_pccs,
        annot=True,
        fmt=".2f",
        cmap="Oranges",
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar=False,
        ax=ax
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(heat.get_children()[0], cax=cax, label="Value")

    plt.tight_layout()
    heatmap_path = os.path.join(out_dir, "cell_type_pcc_heatmap.png")
    fig.savefig(heatmap_path)
    plt.close(fig)

    # true vs pred prop
    cell_names = [name.split('_', 1)[0] for name in all_cell_type_names]
    fig, axes = plt.subplots(
        n_models, n_celltypes,
        figsize=(3 * n_celltypes, 3 * n_models),
        squeeze=False
    )

    for i in range(n_models):
        for j in range(n_celltypes):
            ax = axes[i, j]
            x = all_true_props[i, :, j]
            y = all_pred_props[i, :, j]

            ax.scatter(x, y, alpha=0.6)

            m, b = np.polyfit(x, y, 1)
            x_line = np.array([x.min(), x.max()])
            ax.plot(x_line, m * x_line + b, color="red", linewidth=1.0)

            pcc = all_cell_type_pccs[i, j]
            scc = all_cell_type_sccs[i, j]
            ccc = all_cell_type_cccs[i, j]
            rmse = all_cell_type_rmses[i, j]

            handles = [
                Line2D([], [], linestyle="none", label=f"PCC = {pcc:.2f}"),
                Line2D([], [], linestyle="none", label=f"SCC = {scc:.2f}"),
                Line2D([], [], linestyle="none", label=f"CCC = {ccc:.2f}"),
                Line2D([], [], linestyle="none", label=f"RMSE= {rmse:.2f}")
            ]

            legend = ax.legend(
                handles=handles,
                loc="upper right",
                frameon=True,             # draw the box
                framealpha=0.7,           # semi‐transparent
                edgecolor="black",
                fontsize="small",
                handlelength=1.0,    # shorten the little marker‐line on the left
                handletextpad=0.4,   # reduce space between the marker and the text
                labelspacing=1.0     # keep vertical spacing roughly at default
            )
            legend.get_frame().set_facecolor("white")

            ax.grid(False)

            if i == 0:
                ax.set_title(cell_names[j], fontsize="medium")

        axes[i, 0].set_ylabel(all_model_names[i], fontsize="medium")

    # shared X and Y labels for the whole figure
    fig.subplots_adjust(left=0.15, right=0.90, top=0.95, bottom=0.10, wspace=0.3, hspace=0.3)
    fig.text(
        0.5, 0.03, "True prop",
        ha="center", va="center", fontsize="large"
    )
    fig.text(
        0.07, 0.5, "Pred prop",
        ha="center", va="center", rotation="vertical", fontsize="large"
    )

    scatter_grid_path = os.path.join(out_dir, "true_vs_pred_scatter_grid.png")
    fig.savefig(scatter_grid_path)
    plt.close(fig)

    # per cell type boxplots
    pcc_by_model  = [all_cell_type_pccs[i]  for i in range(n_models)]
    rmse_by_model = [all_cell_type_rmses[i] for i in range(n_models)]
    scc_by_model  = [all_cell_type_sccs[i]  for i in range(n_models)]
    ccc_by_model  = [all_cell_type_cccs[i]  for i in range(n_models)]

    fig, axes = plt.subplots(
        1, 4,
        figsize=(16, 1.5 * n_models),
        sharey=True
    )

    titles = ["Per Cell Type PCC", "Per Cell Type RMSE", "Per Cell Type SCC", "Per Cell Type CCC"]
    data = [pcc_by_model, rmse_by_model, scc_by_model, ccc_by_model]
    xlabels = ["Pearson r", "RMSE", "Spearman r", "CCC"]

    for ax, d, title, xl in zip(axes, data, titles, xlabels):
        ax.boxplot(d, vert=False, labels=all_model_names)
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.grid(False)

    plt.tight_layout()
    boxplots_celltype_path = os.path.join(out_dir, "per_celltype_boxplots.png")
    fig.savefig(boxplots_celltype_path)
    plt.close(fig)

    # per sample boxplots
    pcc_per_sample = np.zeros((n_models, n_samples))
    scc_per_sample = np.zeros((n_models, n_samples))
    rmse_per_sample = np.zeros((n_models, n_samples))
    ccc_per_sample = np.zeros((n_models, n_samples))

    for i in range(n_models):
        for j in range(n_samples):
            y_true = all_true_props[i, j, :]
            y_pred = all_pred_props[i, j, :]

            pcc_val, _ = pearsonr(y_true, y_pred)
            pcc_per_sample[i, j] = pcc_val

            scc_val, _ = spearmanr(y_true, y_pred)
            scc_per_sample[i, j] = scc_val

            rmse_per_sample[i, j] = np.sqrt(np.mean((y_pred - y_true) ** 2))

            mu_t = y_true.mean()
            mu_p = y_pred.mean()
            var_t = y_true.var(ddof=0)
            var_p = y_pred.var(ddof=0)
            cov_tp = np.mean((y_true - mu_t) * (y_pred - mu_p))
            ccc_val = (2 * cov_tp) / (var_t + var_p + (mu_t - mu_p) ** 2)
            ccc_per_sample[i, j] = ccc_val

    pcc_by_model = [pcc_per_sample[i, :].tolist()  for i in range(n_models)]
    scc_by_model = [scc_per_sample[i, :].tolist()  for i in range(n_models)]
    rmse_by_model = [rmse_per_sample[i, :].tolist() for i in range(n_models)]
    ccc_by_model = [ccc_per_sample[i, :].tolist()  for i in range(n_models)]

    fig, axes = plt.subplots(
        1, 4,
        figsize=(20, 1.5 * n_models),
        sharey=True
    )

    titles = ["PerSample Pearson r (PCC)", "Per Sample Spearman r (SCC)", "Per Sample RMSE", "Per Sample CCC"]
    data = [pcc_by_model, scc_by_model, rmse_by_model, ccc_by_model]
    xlabels = ["Pearson r", "Spearman r", "RMSE", "Concordance Correlation Coefficient"]

    for ax, d, title, xl in zip(axes, data, titles, xlabels):
        ax.boxplot(d, vert=False, labels=all_model_names)
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.grid(False)

    plt.tight_layout()
    boxplots_sample_path = os.path.join(out_dir, "per_sample_boxplots.png")
    fig.savefig(boxplots_sample_path)
    plt.close(fig)