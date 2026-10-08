import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
from polynomial import design, ridge_path, fit_model, predict

def mean_squared_error(a, b):
    return float(np.mean((a-b)**2))

def r2_score(a, b):
    return float(1-np.sum((a-b)**2)/np.sum((a-a.mean())**2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="BT2024197")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--search-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    alphas = np.logspace(-8, 4, 13)
    results = {}
    for variant, max_degree in [(1, 10), (2, 20)]:
        path = Path(args.data_dir) / f"BT2024197_train_var{variant}.csv"
        df = pd.read_csv(path)
        features = [f"x{i}" for i in range(1, 7 if variant == 1 else 4)]
        assert list(df.columns) == features + ["y"]
        x, y = df[features].to_numpy(), df.y.to_numpy()
        assert np.isfinite(x).all() and np.isfinite(y).all()
        indices = np.random.default_rng(197).permutation(len(x))
        audit, dev = indices[:200], indices[200:]
        partitions = np.array_split(np.random.default_rng(4197).permutation(len(dev)), 5)
        folds = [(np.setdiff1d(np.arange(len(dev)), va), va) for va in partitions]
        records = []
        for basis in ["monomial", "legendre"]:
            for degree in range(1, max_degree + 1):
                a = design(x[dev], degree, basis)
                losses = []
                for tr, va in folds:
                    preds = ridge_path(a[tr], y[dev[tr]], a[va], alphas)
                    losses.append(np.mean((preds-y[dev[va], None])**2, axis=0))
                losses = np.array(losses)
                for i, alpha in enumerate(alphas):
                    records.append(dict(basis=basis, degree=degree, terms=a.shape[1],
                        alpha=float(alpha), cv_mse=float(losses[:, i].mean()),
                        cv_se=float(losses[:, i].std(ddof=1)/np.sqrt(5))))
                best = int(np.argmin(losses.mean(0)))
                print(f"var{variant} {basis:8s} degree={degree:2d} terms={a.shape[1]:5d} CV={losses.mean(0)[best]:.6g} alpha={alphas[best]:g}", flush=True)
                pd.DataFrame(records).to_csv(out / f"search_var{variant}.csv", index=False)
        if args.search_only:
            continue
        chosen = min(records, key=lambda r: (r["cv_mse"], r["degree"]))
        model = fit_model(x[dev], y[dev], chosen["degree"], chosen["basis"], chosen["alpha"])
        yp = predict(model, x[audit])
        boundary = np.any(np.abs(x[audit]) >= 1-1e-10, axis=1)
        stats = dict(selected=chosen, audit_mse=mean_squared_error(y[audit], yp),
            audit_r2=r2_score(y[audit], yp), audit_n=len(audit), dev_n=len(dev),
            boundary_audit_n=int(boundary.sum()),
            boundary_audit_mse=mean_squared_error(y[audit][boundary], yp[boundary]),
            train_rows=len(df), features=features, train_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        pd.DataFrame(dict(row=audit, actual=y[audit], predicted=yp)).to_csv(out / f"audit_var{variant}.csv", index=False)
        final = fit_model(x, y, chosen["degree"], chosen["basis"], chosen["alpha"])
        np.savez_compressed(out / f"model_var{variant}.npz", **final)
        stats["full_training_mse"] = mean_squared_error(y, predict(final, x))
        results[f"var{variant}"] = stats
        (out / "metrics.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
