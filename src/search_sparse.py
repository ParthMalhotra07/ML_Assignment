import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import csv
import json
from pathlib import Path
import numpy as np
from polynomial import design


def lasso_path(z, target, alphas, max_iter=4000, tol=1e-5):

    n, p = z.shape
    coef = np.zeros(p)
    residual = target.copy()
    paths, gaps = [], []
    for alpha in alphas:
        correlation = z.T @ residual / n
        active = np.flatnonzero((np.abs(correlation) > alpha) | (coef != 0))
        for outer in range(20):
            za = z[:, active]
            gram = za.T @ za / n
            rhs = za.T @ target / n
            beta = coef[active].copy()
            for iteration in range(max_iter):
                largest = 0.
                for j in range(len(active)):
                    rho = rhs[j] - gram[j] @ beta + gram[j,j]*beta[j]
                    updated = np.sign(rho)*max(abs(rho)-alpha, 0)/max(gram[j,j], 1e-30)
                    largest = max(largest, abs(updated-beta[j]))
                    beta[j] = updated
                if largest < tol:
                    break
            coef[active] = beta
            residual = target-z@coef
            corr = z.T @ residual/n
            violations = np.flatnonzero((coef == 0) & (np.abs(corr) > alpha+tol))
            if not len(violations):
                break
            active = np.union1d(active, violations)
        dual = residual * min(1., alpha/max(np.max(np.abs(z.T@residual/n)), 1e-30))
        primal = residual@residual/(2*n)+alpha*np.abs(coef).sum()
        dual_obj = (target@target-(target-dual)@(target-dual))/(2*n)
        paths.append(coef.copy()); gaps.append(max(0., primal-dual_obj))
    return np.array(alphas), np.array(paths).T, np.array(gaps)


def main():
    out = Path('results')
    for variant, degrees in [(1, [3,4,5,6]), (2, [6,8,10,12])]:
        raw = np.loadtxt(f'BT2024197/BT2024197_train_var{variant}.csv', delimiter=',', skiprows=1)
        x, y = raw[:, :-1], raw[:, -1]
        dev = np.random.default_rng(197).permutation(len(x))[200:]
        parts = np.array_split(np.random.default_rng(4197).permutation(len(dev)), 5)
        alphas = np.array([.1, .03, .01, .003])
        cache = out / f'sparse_search_var{variant}.csv'
        records = []
        if cache.exists():
            with cache.open(newline='') as f:
                for row in csv.DictReader(f):
                    row = {k: (v if k in ['basis','method'] else float(v)) for k,v in row.items()}
                    row['degree'],row['terms'] = int(row['degree']),int(row['terms'])
                    if row['degree'] in degrees:
                        records.append(row)
        for degree in degrees:
            if any(r['degree'] == degree for r in records):
                print(f'var{variant} degree {degree}: using saved development-fold results', flush=True)
                continue
            a = design(x[dev], degree)
            losses, gaps = [], []
            for va in parts:
                tr = np.setdiff1d(np.arange(len(dev)), va)
                mu, sd = a[tr].mean(0), a[tr].std(0)
                sd[sd < 1e-12] = 1
                z = np.asfortranarray((a[tr]-mu)/sd)
                _, coefs, gap = lasso_path(z, y[dev[tr]]-y[dev[tr]].mean(),
                    alphas=alphas, max_iter=3000, tol=1e-5)
                pred = ((a[va]-mu)/sd) @ coefs + y[dev[tr]].mean()
                losses.append(np.mean((pred-y[dev[va], None])**2, axis=0))
                gaps.append(gap)
            losses = np.array(losses)
            for i, alpha in enumerate(alphas):
                records.append(dict(basis='monomial', method='lasso', degree=degree,
                    terms=a.shape[1], alpha=float(alpha), cv_mse=float(losses[:, i].mean()),
                    cv_se=float(losses[:, i].std(ddof=1)/np.sqrt(5)),
                    max_dual_gap=float(np.max(np.array(gaps)[:, i]))))
            with (out / f'sparse_search_var{variant}.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
            best = min(records[-len(alphas):], key=lambda r:r['cv_mse'])
            print(f'var{variant} sparse: {json.dumps(best)}', flush=True)


if __name__ == '__main__':
    main()
