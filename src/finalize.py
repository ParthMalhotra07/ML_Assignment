import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
from polynomial import design, fit_model, predict
from search_sparse import lasso_path
from train import mean_squared_error, r2_score


def fit_selected(x, y, chosen):
    if chosen['method'] == 'ridge':
        return fit_model(x,y,chosen['degree'],chosen['basis'],chosen['alpha'])
    a = design(x, chosen['degree'])
    mu, sd = a.mean(0), a.std(0)
    sd[sd < 1e-12] = 1
    _, coefs, gaps = lasso_path((a-mu)/sd,y-y.mean(),
        alphas=[v for v in [.1,.03,.01,.003] if v >= chosen['alpha']], tol=1e-7, max_iter=20000)
    return dict(degree=chosen['degree'], basis='monomial', alpha=chosen['alpha'],
        mean=mu, scale=sd, coef=coefs[:,-1], intercept=float(y.mean()), dual_gap=float(gaps[-1]))


def write_predictions(model, test, output, sample=None):
    prediction = predict(model, test.to_numpy())
    if sample is not None:
        columns = list(pd.read_csv(sample, nrows=0).columns)
        if 'y' not in columns or any(c not in list(test.columns)+['y'] for c in columns):
            raise ValueError(f'Unrecognized sample columns {columns}. Must explicitly resolve before exporting.')
    else:
        columns = ['y']
    completed = test.copy()
    completed['y'] = prediction
    completed[columns].to_csv(output, index=False, float_format='%.17g')
    check = pd.read_csv(output)
    assert len(check) == len(test) and list(check.columns) == columns
    assert np.isfinite(check.y).all() and np.allclose(check.y, prediction, rtol=1e-13, atol=1e-13)
    return prediction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='BT2024197')
    ap.add_argument('--sample', type=Path)
    ap.add_argument('--variant',type=int,choices=[1,2])
    args = ap.parse_args()
    out, submit = Path('results'), Path('submission')
    submit.mkdir(exist_ok=True)
    metrics_path = out/'final_metrics.json'
    summary = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    for variant in ([args.variant] if args.variant else [1,2]):
        ridge = pd.read_csv(out/f'search_var{variant}.csv'); ridge['method']='ridge'
        sparse = pd.read_csv(out/f'sparse_search_var{variant}.csv')

        sparse = sparse[sparse.max_dual_gap <= 1e-3]
        search = pd.concat([ridge,sparse], ignore_index=True)
        chosen = search.sort_values(['cv_mse','degree']).iloc[0].to_dict()
        chosen = {k: chosen[k] for k in ['basis','method','degree','terms','alpha','cv_mse','cv_se']}
        chosen['degree'], chosen['terms'] = int(chosen['degree']), int(chosen['terms'])
        tr_path = Path(args.data_dir)/f'BT2024197_train_var{variant}.csv'
        te_path = Path(args.data_dir)/f'BT2024197_test_var{variant}.csv'
        train, test = pd.read_csv(tr_path), pd.read_csv(te_path)
        x, y = train.drop(columns='y').to_numpy(), train.y.to_numpy()
        assert list(train.columns[:-1]) == list(test.columns)
        assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(test.to_numpy()).all()
        inds = np.random.default_rng(197).permutation(len(x)); audit, dev = inds[:200], inds[200:]
        model = fit_selected(x[dev], y[dev], chosen)
        yp = predict(model,x[audit])
        boundary = np.any(np.abs(x[audit]) >= 1-1e-10,axis=1)
        pd.DataFrame(dict(row=audit, actual=y[audit], predicted=yp)).to_csv(out/f'final_audit_var{variant}.csv',index=False)
        final = fit_selected(x,y,chosen)
        np.savez_compressed(out/f'final_model_var{variant}.npz',**final)
        ytest = write_predictions(final,test,submit/f'BT2024197_pred_var{variant}.csv',args.sample)
        data = dict(selected=chosen, audit_mse=mean_squared_error(y[audit],yp),
            audit_r2=r2_score(y[audit],yp), audit_n=len(audit), dev_n=len(dev),
            audit_boundary_n=int(boundary.sum()), audit_boundary_mse=mean_squared_error(y[audit][boundary],yp[boundary]),
            train_mse=mean_squared_error(y,predict(final,x)), nonzero_coefficients=int(np.count_nonzero(final['coef'])),
            final_dual_gap=final.get('dual_gap'), training_rows=len(train), test_rows=len(test),
            feature_columns=list(test.columns), train_target_std=float(np.std(y)),
            train_target_min=float(y.min()),train_target_max=float(y.max()),
            prediction_min=float(ytest.min()), prediction_max=float(ytest.max()),
            train_boundary_cell_fraction=float(np.mean(np.abs(x)>=1-1e-10)),
            test_boundary_cell_fraction=float(np.mean(np.abs(test.to_numpy())>=1-1e-10)),
            train_duplicate_features=int(train.duplicated(subset=list(test.columns)).sum()),
            test_duplicate_features=int(test.duplicated().sum()),
            sample_format_verified=args.sample is not None,
            train_sha256=hashlib.sha256(tr_path.read_bytes()).hexdigest(),
            test_sha256=hashlib.sha256(te_path.read_bytes()).hexdigest())
        summary[f'var{variant}'] = data
        print(json.dumps(data,indent=2),flush=True)
    metrics_path.write_text(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
