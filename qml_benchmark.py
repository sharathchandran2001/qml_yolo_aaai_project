#!/usr/bin/env python3
"""
qml_benchmark.py - dimension-sweep benchmark: classical baselines vs Variational Quantum Classifier.

Examples
--------
  python qml_benchmark.py --dims 6                         # reproduce the R^6 setting
  python qml_benchmark.py --dims 6 12 24 48 --seeds 0 1 2  # full sweep, 3 seeds
  python qml_benchmark.py --dims 12 --split cv --folds 5   # grouped (by image) cross-validation
  python qml_benchmark.py --dims 6 12 --smoke-test --epochs 5   # pipeline check on SYNTHETIC data
Then:  python generate_report.py            # tables + charts across every dimension run so far

Leakage controls: PCA / angle scaler are fit on the ORIGINAL training crops only (before
oversampling); oversampling touches training data only; the test split is image-level.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold

from data_pipeline import extract_dataset, load_or_build, synthetic_dataset, to_binary
from metrics_utils import compute_metrics
from models.classical_ml import ClassicalClassifier

ROOT = Path(__file__).resolve().parent
MODEL_LABEL = {"majority": "Majority", "svm": "SVM", "rf": "RF", "vqc": "VQC"}
TRAIN_KEEP = ["accuracy", "balanced_accuracy", "macro_f1", "minority_recall"]


# ------------------------------------------------------------------ helpers
def balanced_oversample(y, rng, per_class=None):
    by = {c: np.where(y == c)[0] for c in (0, 1)}
    target = per_class or max(len(v) for v in by.values())
    out = []
    for c, ix in by.items():
        if len(ix) >= target:
            out.append(rng.choice(ix, target, replace=False))
        else:
            out.append(np.concatenate([ix, rng.choice(ix, target - len(ix), replace=True)]))
    idx = np.concatenate(out)
    rng.shuffle(idx)
    return idx


def fit_angle_scaler(Z, mode):
    """Maps PCA features into [0, pi]. Fit on TRAIN only; test values are clipped."""
    if mode == "feature":
        lo, hi = Z.min(0), Z.max(0)
        return lambda A: np.clip(np.pi * (A - lo) / np.maximum(hi - lo, 1e-12), 0.0, np.pi)
    m = max(float(np.max(np.abs(Z))), 1e-12)  # 'global': preserves relative PC variance
    return lambda A: 0.5 * np.pi * (1.0 + np.clip(A / m, -1.0, 1.0))


def time_call(fn, warmup, runs):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return (1000 * float(np.mean(ts)), 1000 * float(np.std(ts))) if ts else (np.nan, np.nan)


def make_splits(data, ybin, mode, folds):
    idx = np.arange(len(ybin))
    if mode == "fixed":
        tr, te = idx[data["split"] == "train"], idx[data["split"] == "val"]
        if len(tr) == 0 or len(te) == 0:
            raise SystemExit("[!] --split fixed needs images in data/images/train AND data/images/val. "
                             "Use --split cv instead.")
        return [(tr, te)]
    sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=0)
    splits = []
    for tr, te in sgkf.split(idx, ybin, groups=data["groups"]):
        if len(set(ybin[tr])) == 2 and len(set(ybin[te])) == 2:
            splits.append((tr, te))
        else:
            print("[!] Skipping a CV fold that lacks both classes in train or test.")
    if not splits:
        raise SystemExit("[!] No valid CV folds (too few images containing the minority class).")
    return splits


# ------------------------------------------------------------------ core evaluation
def evaluate_dim(data, ybin, splits, dim, space, args, models, seeds):
    records, losses, meta = [], {}, {"pca_cumvar": None, "n_qubits": None, "features_per_qubit": None}
    X = data["X"]
    for sid, (tr, te) in enumerate(splits):
        Xtr, ytr, Xte, yte = X[tr], ybin[tr], X[te], ybin[te]
        if len(set(ytr)) < 2:
            print(f"[!] split {sid}: training data has one class only - skipped")
            continue
        if yte.sum() < 10:
            print(f"[!] split {sid}: only {int(yte.sum())} minority samples in the test set - "
                  "metrics will be extremely noisy (consider --split cv).")
        for seed in seeds:
            rng = np.random.default_rng(seed)
            if space == "pca":
                if dim > min(len(Xtr), Xtr.shape[1]):
                    print(f"[!] dim {dim} > #train crops ({len(Xtr)}) - PCA impossible, skipped")
                    return records, losses, meta
                pca = PCA(n_components=dim, random_state=seed).fit(Xtr)  # BEFORE oversampling
                F_tr, F_te = pca.transform(Xtr), pca.transform(Xte)
                var = float(pca.explained_variance_ratio_.sum())
                if meta["pca_cumvar"] is None:
                    meta["pca_cumvar"] = np.cumsum(pca.explained_variance_ratio_).tolist()
                scaler = fit_angle_scaler(F_tr, args.angle_scale)
                chain = lambda raw, p=pca: p.transform(raw)
            else:
                F_tr, F_te, var, scaler = Xtr, Xte, np.nan, None
                chain = lambda raw: raw
            bal = balanced_oversample(ytr, rng, args.oversample_per_class)

            for mname in models:
                if mname == "vqc" and space != "pca":
                    continue
                if mname == "vqc":
                    from models.quantum_ml import QuantumClassifierModel
                    model = QuantumClassifierModel(
                        n_features=dim, n_layers=args.layers, lr=args.lr, epochs=args.epochs,
                        batch_size=args.batch_size, max_qubits=args.max_qubits, seed=seed,
                        init=args.init, mem_budget_gb=args.mem_gb, verbose=args.verbose,
                        allow_large=args.allow_large)
                    A_tr, A_te = scaler(F_tr), scaler(F_te)
                    t0 = time.perf_counter()
                    model.fit(A_tr[bal], ytr[bal])
                    train_t = time.perf_counter() - t0
                    if args.threshold == "train_opt":
                        model.calibrate_threshold(A_tr, ytr)  # original train data only
                    predict, score_fn = model.predict, model.decision_function
                    M_tr, M_te = A_tr, A_te
                    full_chain = lambda raw, c=chain, s=scaler: s(c(raw))
                    n_params, n_qubits = model.n_params, model.n_qubits
                    fpq, thr = model.features_per_qubit, model.threshold
                    final_loss = model.history["loss"][-1]
                    losses[f"s{sid}_seed{seed}"] = model.history["loss"]
                    meta["n_qubits"], meta["features_per_qubit"] = n_qubits, fpq
                else:
                    clf = ClassicalClassifier(mname, seed=seed)
                    t0 = time.perf_counter()
                    clf.train(F_tr[bal], ytr[bal])
                    train_t = time.perf_counter() - t0
                    predict, score_fn = clf.predict, clf.scores
                    M_tr, M_te, full_chain = F_tr, F_te, chain
                    n_params = n_qubits = fpq = np.nan
                    thr = final_loss = np.nan

                rec = {"dim": dim, "space": space, "model": MODEL_LABEL[mname], "split_id": sid,
                       "seed": seed, "n_train": len(ytr), "n_train_balanced": len(bal),
                       "n_test": len(yte), "n_pos_train": int(ytr.sum()), "n_pos_test": int(yte.sum()),
                       "pca_var_explained": var, "n_qubits": n_qubits, "features_per_qubit": fpq,
                       "n_params": n_params, "train_time_s": train_t, "final_loss": final_loss,
                       "threshold": thr}
                rec.update(compute_metrics(yte, predict(M_te), score_fn(M_te)))
                trm = compute_metrics(ytr, predict(M_tr), score_fn(M_tr), prefix="train_")
                rec.update({"train_" + k: trm["train_" + k] for k in TRAIN_KEEP})
                if args.latency_runs > 0:
                    x_model, x_raw = M_te[:1], Xte[:1]
                    rec["latency_ms"], rec["latency_std_ms"] = time_call(
                        lambda: predict(x_model), args.latency_warmup, args.latency_runs)
                    rec["latency_pipeline_ms"], _ = time_call(
                        lambda: predict(full_chain(x_raw)), args.latency_warmup, args.latency_runs)
                records.append(rec)
                print(f"   [{space}{dim if space == 'pca' else ''}] split {sid} seed {seed} "
                      f"{MODEL_LABEL[mname]:<8} bal.acc={rec['balanced_accuracy']:.3f} "
                      f"macroF1={rec['macro_f1']:.3f} MCC={rec['mcc']:.3f} "
                      f"minRecall={rec['minority_recall']:.3f} train={train_t:.1f}s")
    return records, losses, meta


def versions():
    import sklearn
    v = {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__}
    try:
        import pennylane
        v["pennylane"] = pennylane.__version__
    except Exception:
        pass
    return v


def save_run(out_dir, space, dim, records, losses, meta, args, data_info, wall):
    d = out_dir / (f"dim_{dim:03d}" if space == "pca" else "dim_raw")
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(d / "records.csv", index=False)
    (d / "loss_history.json").write_text(json.dumps(losses))
    meta = {**meta, "dim": dim, "space": space, "args": vars(args), "data": data_info,
            "versions": versions(), "wall_time_s": wall,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return d


def print_summary(records, title):
    if not records:
        return
    df = pd.DataFrame(records)
    cols = [c for c in ["balanced_accuracy", "macro_f1", "mcc", "pr_auc", "roc_auc",
                        "minority_recall", "train_time_s", "latency_ms"] if c in df]
    print(f"\n----- {title} (mean over {df.groupby('model').size().iloc[0]} run(s)) -----")
    print(df.groupby("model")[cols].mean().round(4).to_string())
    print()


# ------------------------------------------------------------------ CLI
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dims", type=int, nargs="+", default=[6], help="PCA feature dimensions, e.g. 6 12 24 48")
    p.add_argument("--models", nargs="+", default=["majority", "svm", "rf", "vqc"],
                   choices=list(MODEL_LABEL))
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--split", choices=["fixed", "cv"], default="fixed",
                   help="fixed = data/images/train vs val (image-level); cv = grouped k-fold by image")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--skip-full", action="store_true", help="skip the raw 1024-D classical baselines")
    # data
    p.add_argument("--images-dir", default=str(ROOT / "data" / "images"))
    p.add_argument("--crop-source", choices=["labels", "yolo"], default="labels")
    p.add_argument("--weights", default=None, help="YOLO weights (only for --crop-source yolo)")
    p.add_argument("--refresh", action="store_true", help="re-extract crops, ignore cache")
    p.add_argument("--oversample-per-class", type=int, default=None,
                   help="samples per class after oversampling (default: majority count)")
    # VQC
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-qubits", type=int, default=12,
                   help="cap on simulated qubits; dims above it pack several features per qubit")
    p.add_argument("--allow-large", action="store_true", help="allow >20 simulated qubits (huge RAM/time)")
    p.add_argument("--mem-gb", type=float, default=1.0, help="memory budget used to cap the batch size")
    p.add_argument("--init", choices=["uniform", "small"], default="uniform")
    p.add_argument("--angle-scale", choices=["global", "feature"], default="global")
    p.add_argument("--threshold", choices=["zero", "train_opt"], default="zero")
    p.add_argument("--verbose", action="store_true", help="print per-epoch VQC loss")
    # latency / output
    p.add_argument("--latency-warmup", type=int, default=20)
    p.add_argument("--latency-runs", type=int, default=100, help="0 disables latency timing")
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--tag", default="default", help="sub-folder name, lets you keep separate experiments")
    p.add_argument("--smoke-test", action="store_true", help="SYNTHETIC data, pipeline check only")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.smoke_test and args.tag == "default":
        args.tag = "smoke_test"
    out_dir = Path(args.results_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        print("[!] SMOKE TEST: synthetic data - results are NOT scientific and go to results/smoke_test.")
        data = synthetic_dataset()
    else:
        weights = args.weights or str(ROOT / "runs" / "yolov8s_aaai_experiment" / "weights" / "best.pt")
        if args.crop_source == "yolo" and not Path(weights).exists():
            weights = str(ROOT / "yolov8s.pt")
        cache = ROOT / "data" / "processed" / f"crops_{args.crop_source}.npz"
        data = load_or_build(cache, lambda: extract_dataset(
            args.images_dir, source=args.crop_source, weights=weights), args.refresh)

    ybin, info = to_binary(data["y"])
    data_info = {**info, "n_crops": int(len(ybin)), "n_images": int(len(set(data["groups"]))),
                 "crop_source": "synthetic" if args.smoke_test else args.crop_source}
    print(f"[+] {data_info['n_crops']} crops from {data_info['n_images']} images | "
          f"class counts {info['class_counts']} | positive (minority) class id = {info['positive_class_id']}")
    splits = make_splits(data, ybin, args.split, args.folds)
    for i, (tr, te) in enumerate(splits):
        print(f"    split {i}: train={len(tr)} (minority {int(ybin[tr].sum())}) | "
              f"test={len(te)} (minority {int(ybin[te].sum())})")

    if not args.skip_full:
        t0 = time.time()
        raw_models = [m for m in args.models if m != "vqc"]
        rec, loss, meta = evaluate_dim(data, ybin, splits, data["X"].shape[1], "raw", args, raw_models, args.seeds)
        save_run(out_dir, "raw", data["X"].shape[1], rec, loss, meta, args, data_info, time.time() - t0)
        print_summary(rec, f"raw {data['X'].shape[1]}-D classical baselines")

    for d in args.dims:
        print(f"\n================ PCA dimension d = {d} ================")
        t0 = time.time()
        rec, loss, meta = evaluate_dim(data, ybin, splits, d, "pca", args, args.models, args.seeds)
        if not rec:
            continue
        folder = save_run(out_dir, "pca", d, rec, loss, meta, args, data_info, time.time() - t0)
        print_summary(rec, f"PCA d={d}")
        print(f"[+] saved -> {folder}")
    print(f"\nDone. Next: python generate_report.py --tag {args.tag}")


if __name__ == "__main__":
    sys.exit(main())
