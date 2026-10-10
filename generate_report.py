#!/usr/bin/env python3
"""
generate_report.py - merges EVERY dimension run found in results/<tag>/dim_*/ into one report.

  python generate_report.py                 # tag "default"
  python generate_report.py --tag smoke_test
  python generate_report.py --tag default --metrics macro_f1 mcc pr_auc balanced_accuracy

Outputs (results/<tag>/report/):
  comparison_table.{md,tex,csv}        mean +/- std per dimension x model (LaTeX: booktabs, best in bold)
  improvement_table.{md,csv}           change vs. smallest dimension and vs. majority baseline
  summary_all_metrics.csv              every metric, mean and std
  fig1_metrics_vs_dim.{png,pdf}        quality metrics vs. dimension
  fig2_pca_variance.{png,pdf}          cumulative explained variance
  fig3_compute_scaling.{png,pdf}       training time / latency / qubits / params
  fig4_vqc_loss_curves.{png,pdf}       VQC training loss per dimension
  fig5_vqc_confusion.{png,pdf}         VQC confusion matrices per dimension
  fig6_heatmaps.{png,pdf}              model x dimension heatmaps
  report.md                            narrative summary + warnings
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parent
COLORS = {"Majority": "#7f7f7f", "SVM": "#0072B2", "RF": "#009E73", "VQC": "#D55E00"}
MARKERS = {"Majority": "x", "SVM": "o", "RF": "s", "VQC": "D"}
NICE = {"balanced_accuracy": "Balanced accuracy", "macro_f1": "Macro F1", "mcc": "MCC",
        "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "minority_recall": "Minority recall",
        "minority_precision": "Minority precision", "minority_f1": "Minority F1", "accuracy": "Accuracy",
        "train_time_s": "Train time (s)", "latency_ms": "Latency (ms)",
        "latency_pipeline_ms": "Pipeline latency (ms)"}
TABLE_METRICS = ["balanced_accuracy", "macro_f1", "mcc", "pr_auc", "minority_recall"]
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


# ------------------------------------------------------------------ loading / aggregation
def load_all(run_dir):
    frames, metas, losses = [], {}, {}
    for d in sorted(run_dir.glob("dim_*")):
        f = d / "records.csv"
        if not f.exists():
            continue
        frames.append(pd.read_csv(f))
        key = "raw" if d.name == "dim_raw" else int(d.name.split("_")[1])
        metas[key] = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        losses[key] = json.loads((d / "loss_history.json").read_text()) if (d / "loss_history.json").exists() else {}
    if not frames:
        raise SystemExit(f"[!] No dim_*/records.csv found in {run_dir}. Run qml_benchmark.py first.")
    return pd.concat(frames, ignore_index=True), metas, losses


def aggregate(df):
    num = [c for c in df.columns if df[c].dtype.kind in "fi" and c not in
           ("dim", "split_id", "seed", "tn", "fp", "fn", "tp")]
    g = df.groupby(["space", "dim", "model"])
    mean, std = g[num].mean(), g[num].std(ddof=1)
    out = mean.add_suffix("_mean").join(std.add_suffix("_std"))
    out["n_runs"] = g.size()
    for c in ("tn", "fp", "fn", "tp"):
        out[c] = g[c].sum()
    return out.reset_index()


def fmt(m, s, n, nd=3):
    if pd.isna(m):
        return "-"
    return f"{m:.{nd}f} ± {s:.{nd}f}" if (n > 1 and not pd.isna(s)) else f"{m:.{nd}f}"


def row_label(r):
    return f"{int(r['dim'])} (raw)" if r["space"] == "raw" else str(int(r["dim"]))


def order_rows(agg):
    mo = {"Majority": 0, "SVM": 1, "RF": 2, "VQC": 3}
    a = agg.copy()
    a["_s"] = (a["space"] == "pca").astype(int)
    a["_m"] = a["model"].map(mo)
    return a.sort_values(["_s", "dim", "_m"]).drop(columns=["_s", "_m"])


# ------------------------------------------------------------------ tables
def build_tables(agg, metrics, out):
    a = order_rows(agg)
    cols = metrics + ["train_time_s", "latency_ms"]
    rows = []
    for _, r in a.iterrows():
        row = {"d": row_label(r), "Model": r["model"],
               "qubits": "-" if pd.isna(r.get("n_qubits_mean")) else int(r["n_qubits_mean"])}
        for c in cols:
            row[NICE[c]] = fmt(r.get(c + "_mean"), r.get(c + "_std"), r["n_runs"],
                               3 if c not in ("train_time_s", "latency_ms") else 2)
        rows.append(row)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(out / "comparison_table.csv", index=False)
    (out / "comparison_table.md").write_text(md_table(tbl))

    # LaTeX with best (non-majority) value in bold per (d, metric)
    best = {}
    for (sp, d), grp in a.groupby(["space", "dim"]):
        for c in metrics:
            g2 = grp[grp["model"] != "Majority"]
            if g2[c + "_mean"].notna().any():
                best[(sp, d, c)] = g2.loc[g2[c + "_mean"].idxmax(), "model"]
    head = ["$d$", "Model", "Qubits"] + [NICE[c] for c in cols]
    lines = ["\\begin{tabular}{" + "ll" + "r" + "c" * len(cols) + "}", "\\toprule",
             " & ".join(head) + " \\\\", "\\midrule"]
    prev = None
    for _, r in a.iterrows():
        key = (r["space"], r["dim"])
        if prev is not None and key != prev:
            lines.append("\\midrule")
        prev = key
        cells = [row_label(r), r["model"], "-" if pd.isna(r.get("n_qubits_mean")) else str(int(r["n_qubits_mean"]))]
        for c in cols:
            s = fmt(r.get(c + "_mean"), r.get(c + "_std"), r["n_runs"], 3 if c not in ("train_time_s", "latency_ms") else 2)
            s = s.replace("±", "$\\pm$")
            if c in metrics and best.get((r["space"], r["dim"], c)) == r["model"]:
                s = "\\textbf{" + s + "}"
            cells.append(s)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "comparison_table.tex").write_text("\n".join(lines))
    return tbl


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def improvement_table(agg, out):
    pca = agg[agg["space"] == "pca"]
    dims = sorted(pca["dim"].unique())
    rows = []
    for model in [m for m in ["SVM", "RF", "VQC"] if m in set(pca["model"])]:
        sub = pca[pca["model"] == model].set_index("dim")
        base_dim = dims[0]
        for d in dims:
            if d not in sub.index:
                continue
            row = {"Model": model, "d": d}
            for c in ["macro_f1", "balanced_accuracy", "mcc"]:
                v = sub.loc[d, c + "_mean"]
                row[f"{NICE[c]}"] = round(v, 4)
                if base_dim in sub.index:
                    row[f"Δ {NICE[c]} vs d={base_dim}"] = round(v - sub.loc[base_dim, c + "_mean"], 4)
                maj = pca[(pca["model"] == "Majority") & (pca["dim"] == d)]
                if len(maj):
                    row[f"Δ {NICE[c]} vs majority"] = round(v - maj.iloc[0][c + "_mean"], 4)
            rows.append(row)
    tbl = pd.DataFrame(rows)
    tbl.to_csv(out / "improvement_table.csv", index=False)
    (out / "improvement_table.md").write_text(md_table(tbl) if len(tbl) else "", encoding="utf-8")
    return tbl


# ------------------------------------------------------------------ figures
def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=300)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def set_dim_axis(ax, dims):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(dims))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("PCA feature dimension $d$")


def fig_metrics(agg, metrics, out):
    pca, raw = agg[agg["space"] == "pca"], agg[agg["space"] == "raw"]
    dims = sorted(pca["dim"].unique())
    ms = metrics[:4]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    for ax, c in zip(axes.ravel(), ms):
        for model in ["Majority", "SVM", "RF", "VQC"]:
            s = pca[pca["model"] == model].sort_values("dim")
            if s.empty:
                continue
            err = s[c + "_std"].fillna(0).values
            ax.errorbar(s["dim"], s[c + "_mean"], yerr=err, color=COLORS[model], marker=MARKERS[model],
                        ls=":" if model == "Majority" else "-", capsize=3, lw=1.4, ms=5, label=model)
        for model in ["SVM", "RF"]:
            r = raw[raw["model"] == model]
            if len(r):
                ax.axhline(r.iloc[0][c + "_mean"], color=COLORS[model], ls="--", lw=1,
                           alpha=0.7, label=f"{model} (raw 1024-D)")
        set_dim_axis(ax, dims)
        ax.set_ylabel(NICE[c])
        ax.set_title(NICE[c])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 4), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "fig1_metrics_vs_dim.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "fig1_metrics_vs_dim.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_variance(metas, out):
    best = max((k for k in metas if k != "raw" and metas[k].get("pca_cumvar")), default=None)
    if best is None:
        return
    cv = metas[best]["pca_cumvar"]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(range(1, len(cv) + 1), cv, color="#444", lw=1.5)
    for d in sorted(k for k in metas if k != "raw" and metas[k].get("pca_cumvar")):
        v = metas[d]["pca_cumvar"][-1]
        ax.scatter([d], [v], color=COLORS["VQC"], zorder=3)
        ax.annotate(f"d={d}\n{100 * v:.1f}%", (d, v), textcoords="offset points", xytext=(4, -22), fontsize=7)
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.02)
    save(fig, out, "fig2_pca_variance")


def fig_compute(agg, out):
    pca = agg[agg["space"] == "pca"]
    dims = sorted(pca["dim"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1))
    for model in ["SVM", "RF", "VQC"]:
        s = pca[pca["model"] == model].sort_values("dim")
        if s.empty:
            continue
        axes[0].errorbar(s["dim"], s["train_time_s_mean"], yerr=s["train_time_s_std"].fillna(0),
                         color=COLORS[model], marker=MARKERS[model], capsize=3, label=model)
        axes[1].errorbar(s["dim"], s["latency_ms_mean"], yerr=s["latency_ms_std"].fillna(0),
                         color=COLORS[model], marker=MARKERS[model], capsize=3, label=f"{model} (model)")
        if "latency_pipeline_ms_mean" in s:
            axes[1].plot(s["dim"], s["latency_pipeline_ms_mean"], color=COLORS[model], ls="--",
                         marker=MARKERS[model], mfc="none", alpha=0.7, label=f"{model} (PCA+model)")
    for ax, t in zip(axes[:2], ["Training time (s)", "Single-sample latency (ms)"]):
        ax.set_yscale("log")
        ax.set_ylabel(t)
        ax.set_title(t)
        set_dim_axis(ax, dims)
    axes[1].legend(ncol=1, frameon=False, fontsize=6.5)
    q = pca[pca["model"] == "VQC"].sort_values("dim")
    if len(q):
        x = np.arange(len(q))
        axes[2].bar(x - 0.2, q["n_qubits_mean"], 0.4, color=COLORS["VQC"], label="simulated qubits")
        axes[2].bar(x + 0.2, q["n_params_mean"] / 10, 0.4, color="#999", label="trainable params / 10")
        for xi, (_, r) in zip(x, q.iterrows()):
            axes[2].text(xi - 0.2, r["n_qubits_mean"], f"{int(r['n_qubits_mean'])}", ha="center", va="bottom", fontsize=7)
            axes[2].text(xi + 0.2, r["n_params_mean"] / 10, f"{int(r['n_params_mean'])}", ha="center", va="bottom", fontsize=7)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([int(d) for d in q["dim"]])
        axes[2].set_xlabel("PCA feature dimension $d$")
        axes[2].set_title("VQC size")
        axes[2].legend(frameon=False)
    save(fig, out, "fig3_compute_scaling")


def fig_loss(losses, out):
    dims = [d for d in sorted(k for k in losses if k != "raw") if losses[d]]
    if not dims:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(dims)))
    for c, d in zip(cmap, dims):
        arr = np.array([v for v in losses[d].values()])
        mu, sd = arr.mean(0), arr.std(0)
        ep = np.arange(1, len(mu) + 1)
        ax.plot(ep, mu, color=c, lw=1.5, label=f"d={d}")
        if len(arr) > 1:
            ax.fill_between(ep, mu - sd, mu + sd, color=c, alpha=0.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training MSE")
    ax.legend(frameon=False)
    save(fig, out, "fig4_vqc_loss_curves")


def fig_confusion(agg, out):
    q = agg[(agg["space"] == "pca") & (agg["model"] == "VQC")].sort_values("dim")
    if q.empty:
        return
    n = len(q)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.8), squeeze=False)
    for ax, (_, r) in zip(axes[0], q.iterrows()):
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]], float)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i, j])}\n({100 * norm[i, j]:.0f}%)", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Majority", "Minority"]); ax.set_yticklabels(["Majority", "Minority"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"d = {int(r['dim'])}")
        ax.grid(False)
    save(fig, out, "fig5_vqc_confusion")


def fig_heatmaps(agg, out):
    a = agg.copy()
    a["col"] = a.apply(row_label, axis=1)
    colorder = [row_label(r) for _, r in order_rows(a).drop_duplicates(["space", "dim"]).iterrows()]
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
    for ax, c in zip(axes, ["macro_f1", "mcc"]):
        pv = a.pivot_table(index="model", columns="col", values=c + "_mean")
        pv = pv.reindex(index=[m for m in ["Majority", "SVM", "RF", "VQC"] if m in pv.index],
                        columns=[x for x in colorder if x in pv.columns])
        im = ax.imshow(pv.values, cmap="YlGnBu", vmin=min(0, np.nanmin(pv.values)), vmax=1, aspect="auto")
        ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns, rotation=30)
        ax.set_yticks(range(pv.shape[0])); ax.set_yticklabels(pv.index)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v = pv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if v > 0.6 else "black")
        ax.set_title(NICE[c]); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    save(fig, out, "fig6_heatmaps")


# ------------------------------------------------------------------ narrative
def write_report(agg, metas, tbl, imp, out, tag):
    lines = [f"# Benchmark report - `{tag}`", ""]
    any_meta = next((m for k, m in metas.items() if k != "raw" and m), None) or next(iter(metas.values()), {})
    a = any_meta.get("args", {})
    di = any_meta.get("data", {})
    if di.get("crop_source") == "synthetic":
        lines += ["> **SMOKE TEST on synthetic data - not a scientific result. Do not cite.**", ""]
    lines += ["## Setup",
              f"- Crops: {di.get('n_crops', '?')} from {di.get('n_images', '?')} images, class counts {di.get('class_counts', '?')}"
              f" (source: {di.get('crop_source', '?')}); minority/positive class id = {di.get('positive_class_id', '?')}",
              f"- Split: {a.get('split', '?')}" + (f" ({a.get('folds')} folds)" if a.get("split") == "cv" else "")
              + f"; seeds: {a.get('seeds', '?')}; VQC epochs {a.get('epochs', '?')}, lr {a.get('lr', '?')}, "
              f"layers {a.get('layers', '?')}, angle scaling `{a.get('angle_scale', '?')}`, threshold `{a.get('threshold', '?')}`",
              f"- Dimensions run: {sorted(k for k in metas if k != 'raw')}", ""]
    pca = agg[agg["space"] == "pca"]
    v = pca[pca["model"] == "VQC"].sort_values("dim")
    warns = []
    if len(v):
        packed = v[v["n_qubits_mean"] < v["dim"]]
        if len(packed):
            warns.append("For d > max-qubits several features share one qubit "
                         "(RX then RY rounds): d=" + ", ".join(str(int(x)) for x in packed["dim"]) +
                         f" use {', '.join(str(int(x)) for x in packed['n_qubits_mean'])} qubits. "
                         "Say so in the paper - these are not one-feature-per-qubit circuits.")
        dead = v[v["minority_recall_mean"] == 0]
        if len(dead):
            warns.append("VQC never predicts the minority class at d=" + ", ".join(str(int(x)) for x in dead["dim"])
                         + " (degenerate majority-class classifier).")
        maj = pca[pca["model"] == "Majority"].set_index("dim")
        for _, r in v.iterrows():
            if r["dim"] in maj.index and r["macro_f1_mean"] <= maj.loc[r["dim"], "macro_f1_mean"] + 0.02:
                warns.append(f"VQC macro-F1 at d={int(r['dim'])} is within 0.02 of the majority baseline.")
    pos = agg["n_pos_test_mean"].min() if "n_pos_test_mean" in agg else np.nan
    if not np.isnan(pos) and pos < 10:
        warns.append(f"Average minority samples per test split is only {pos:.1f}: differences between models are "
                     "within noise. Use --split cv and more seeds before drawing conclusions.")
    if (agg["n_runs"] < 3).all():
        warns.append("Fewer than 3 runs per cell: the ± values are not meaningful.")
    if warns:
        lines += ["## Warnings"] + [f"- {w}" for w in warns] + [""]
    lines += ["## Comparison table (mean ± std over splits × seeds)", "", md_table(tbl), ""]
    if len(imp):
        lines += ["## Improvement vs smallest dimension / majority baseline", "", md_table(imp), ""]
    if len(v):
        b = v.loc[v["macro_f1_mean"].idxmax()]
        lines += ["## Best VQC dimension", f"- By macro-F1: d={int(b['dim'])} ({b['macro_f1_mean']:.3f}); "
                  f"by MCC: d={int(v.loc[v['mcc_mean'].idxmax(), 'dim'])}", ""]
    lines += ["## Figures", "fig1_metrics_vs_dim, fig2_pca_variance, fig3_compute_scaling, fig4_vqc_loss_curves, "
              "fig5_vqc_confusion, fig6_heatmaps (PNG 300 dpi + vector PDF in this folder)."]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--tag", default="default")
    p.add_argument("--metrics", nargs="+", default=["macro_f1", "balanced_accuracy", "mcc", "pr_auc", "minority_recall"],
                   choices=[k for k in NICE if k not in ("train_time_s", "latency_ms", "latency_pipeline_ms")])
    args = p.parse_args(argv)
    run_dir = Path(args.results_dir) / args.tag
    out = run_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    df, metas, losses = load_all(run_dir)
    agg = aggregate(df)
    agg.to_csv(out / "summary_all_metrics.csv", index=False)
    tbl = build_tables(agg, args.metrics, out)
    imp = improvement_table(agg, out)
    fig_metrics(agg, args.metrics, out)
    fig_variance(metas, out)
    fig_compute(agg, out)
    fig_loss(losses, out)
    fig_confusion(agg, out)
    fig_heatmaps(agg, out)
    write_report(agg, metas, tbl, imp, out, args.tag)
    print(f"[+] Report written to {out}")
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
