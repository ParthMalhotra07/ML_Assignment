import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path('src').resolve()))
from polynomial import design, ridge_path
from search_sparse import lasso_path

out = Path('results/model_review')
out.mkdir(parents=True, exist_ok=True)
metrics = json.loads(Path('results/final_metrics.json').read_text())
seeds = [4197, 4207, 4217]
records = []
if (out/'fold_results.csv').exists():
    records = pd.read_csv(out/'fold_results.csv').to_dict('records')

variants = [int(sys.argv[1])] if len(sys.argv)>1 else [1, 2]
if any(v not in [1, 2] for v in variants):
    raise ValueError('Variant must be 1 or 2')
records = [r for r in records if int(r['variant']) not in variants]
for variant in variants:
    path = Path(f'BT2024197/BT2024197_train_var{variant}.csv')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metrics[f'var{variant}']['train_sha256']
    frame = pd.read_csv(path)
    x, y = frame.drop(columns='y').to_numpy(), frame.y.to_numpy()
    order = np.random.default_rng(197).permutation(len(x))
    dev = order[200:]
    xd, yd = x[dev], y[dev]
    degrees = range(3, 8) if variant == 1 else range(7, 14)
    alphas = np.logspace(-2, 3, 21)
    for degree in degrees:
        a = design(xd, degree, 'monomial')
        losses = []
        boundaries = []
        for seed in seeds:
            parts = np.array_split(np.random.default_rng(seed).permutation(len(dev)), 5)
            for fold, va in enumerate(parts):
                tr = np.setdiff1d(np.arange(len(dev)), va)
                yp = ridge_path(a[tr], yd[tr], a[va], alphas)
                errors = (yp - yd[va, None]) ** 2
                boundary = np.any(np.abs(xd[va]) >= 1-1e-10, axis=1)
                losses.append(errors.mean(0))
                boundaries.append(errors[boundary].mean(0))
                for j, alpha in enumerate(alphas):
                    records.append(dict(variant=variant, method='ridge', degree=degree,
                        alpha=float(alpha), seed=seed, fold=fold, mse=float(losses[-1][j]),
                        boundary_mse=float(boundaries[-1][j]), dual_gap=0.))
        best = np.argmin(np.mean(losses, axis=0))
        print(f'var{variant} ridge d={degree} alpha={alphas[best]:.6g} repeated CV MSE={np.mean(losses,axis=0)[best]:.6f}', flush=True)
        pd.DataFrame(records).to_csv(out/'fold_results.csv',index=False)
    if variant == 1:
        alphas = np.array([.03, .02, .01, .007, .005])
        for degree in [4, 5, 6]:
            a = design(xd, degree)
            losses = []
            for seed in seeds:
                parts = np.array_split(np.random.default_rng(seed).permutation(len(dev)), 5)
                for fold, va in enumerate(parts):
                    tr = np.setdiff1d(np.arange(len(dev)), va)
                    mu, sd = a[tr].mean(0), a[tr].std(0)
                    sd[sd < 1e-12] = 1
                    _, coef, gaps = lasso_path((a[tr]-mu)/sd, yd[tr]-yd[tr].mean(), alphas, max_iter=5000, tol=1e-6)
                    yp = ((a[va]-mu)/sd) @ coef + yd[tr].mean()
                    errors = (yp - yd[va, None])**2
                    boundary = np.any(np.abs(xd[va]) >= 1-1e-10, axis=1)
                    losses.append(errors.mean(0))
                    for j, alpha in enumerate(alphas):
                        records.append(dict(variant=variant, method='lasso', degree=degree,
                            alpha=float(alpha), seed=seed, fold=fold, mse=float(losses[-1][j]),
                            boundary_mse=float(errors[boundary,j].mean()), dual_gap=float(gaps[j])))
                    pd.DataFrame(records).to_csv(out/'fold_results.csv',index=False)
                    print(f'var1 lasso d={degree} seed={seed} fold={fold+1}/5 finished',flush=True)
            best = np.argmin(np.mean(losses,axis=0))
            print(f'var1 lasso d={degree} alpha={alphas[best]:g} repeated CV MSE={np.mean(losses,axis=0)[best]:.6f}',flush=True)

all_results = pd.DataFrame(records)
summary = all_results.groupby(['variant','method','degree','alpha']).agg(
    mean_mse=('mse','mean'), std_mse=('mse','std'), boundary_mse=('boundary_mse','mean'),
    maximum_dual_gap=('dual_gap','max'), folds=('fold','size')).reset_index()
summary = summary[summary.folds == 15]
summary.to_csv(out/'summary.csv',index=False)
for variant in [1,2]:
    valid = summary[(summary.variant==variant)&(summary.maximum_dual_gap<=.001)]
    print(valid.sort_values('mean_mse').head(8).to_string(index=False),flush=True)
