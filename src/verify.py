import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from polynomial import exponents, design, ridge_path, predict
from search_sparse import lasso_path


def main():

    rng=np.random.default_rng(46)
    for p,d in [(6,5),(3,10),(6,10),(3,20)]:
        powers=exponents(p,d)
        assert len(powers)==math.comb(p+d,d)-1
        assert np.max(powers.sum(1))==d and np.min(powers.sum(1))==1
    a=rng.normal(size=(90,12));y=rng.normal(size=90);b=rng.normal(size=(20,12))
    z=(a-a.mean(0))/a.std(0)
    for alpha in [.01,1,100]:
        beta=np.linalg.lstsq(np.vstack([z,np.sqrt(alpha)*np.eye(12)]),np.r_[y-y.mean(),np.zeros(12)],rcond=None)[0]
        expected=(b-a.mean(0))/a.std(0)@beta+y.mean()
        actual=ridge_path(a,y,b,[alpha])[:,0]
        assert np.allclose(actual,expected,atol=1e-9)

    q=np.linalg.qr(np.column_stack([np.ones(90),a]))[0][:,1:]*np.sqrt(90)
    yy=q@np.linspace(-1,1,12)
    _,coef,gap=lasso_path(q,yy,[.1,.03],tol=1e-9)
    expected=np.sign(np.linspace(-1,1,12))*np.maximum(np.abs(np.linspace(-1,1,12))-.03,0)
    assert np.allclose(coef[:,-1],expected,atol=1e-7) and gap[-1]<1e-8
    summary=json.loads(Path('results/final_metrics.json').read_text())
    checks={}
    for v in [1,2]:
        test=pd.read_csv(f'BT2024197/BT2024197_test_var{v}.csv')
        saved=np.load(f'results/final_model_var{v}.npz',allow_pickle=False)
        yp=predict(dict(saved),test.to_numpy())
        output=pd.read_csv(f'submission/BT2024197_pred_var{v}.csv')
        assert len(output)==1000 and 'y' in output
        assert np.isfinite(output.y).all() and np.allclose(yp,output.y,rtol=1e-13,atol=1e-13)
        audit=pd.read_csv(f'results/final_audit_var{v}.csv')
        mse=float(np.mean((audit.actual-audit.predicted)**2))
        r2=float(1-np.sum((audit.actual-audit.predicted)**2)/np.sum((audit.actual-audit.actual.mean())**2))
        assert np.isclose(mse,summary[f'var{v}']['audit_mse']) and np.isclose(r2,summary[f'var{v}']['audit_r2'])
        checks[f'var{v}']=dict(rows=len(output),columns=list(output),reload_predictions_match=True,metrics_recomputed=True)
    Path('results/verification.json').write_text(json.dumps(checks,indent=2))
    print('PASS: polynomial degree constraints, ridge reference solve, Lasso closed-form reference, model reload, row counts, finite predictions, and independently recomputed metrics.')


if __name__=='__main__':main()
