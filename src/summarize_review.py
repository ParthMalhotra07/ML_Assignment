import json
from pathlib import Path
import pandas as pd

root = Path('results/model_review')
folds = pd.read_csv(root/'fold_results.csv')
seed_summary = folds.groupby(['variant','method','degree','alpha','seed']).agg(mean_mse=('mse','mean'), folds=('fold','size')).reset_index()
seed_summary = seed_summary[seed_summary.folds==5]
print('Best completed degree-6 check:')
print(seed_summary[(seed_summary.variant==1)&(seed_summary.method=='lasso')&(seed_summary.degree==6)].sort_values('mean_mse').head(5).to_string(index=False))
summary = folds.groupby(['variant','method','degree','alpha']).agg(
    mean_mse=('mse','mean'), std_mse=('mse','std'), boundary_mse=('boundary_mse','mean'),
    maximum_dual_gap=('dual_gap','max'), folds=('fold','size')).reset_index()
summary = summary[summary.folds==15]
summary.to_csv(root/'summary.csv',index=False)
for variant in [1,2]:
    valid = summary[(summary.variant==variant)&(summary.maximum_dual_gap<=.001)]
    if valid.empty:
        continue
    best = valid.sort_values('mean_mse').iloc[0]
    base = dict(method='lasso',degree=5,alpha=.01) if variant==1 else dict(method='ridge',degree=10,alpha=1.)
    baseline = valid[(valid.method==base['method'])&(valid.degree==base['degree'])&(abs(valid.alpha-base['alpha'])<1e-12)]
    print(f'VARIANT {variant}')
    print(valid.sort_values('mean_mse').head(8).to_string(index=False))
    if not baseline.empty:
        baseline = baseline.iloc[0]
        print(f'Current model mean MSE {baseline.mean_mse:.8f}; best candidate {best.mean_mse:.8f}; improvement {(baseline.mean_mse-best.mean_mse)/baseline.mean_mse:.2%}')
        for seed in sorted(folds.seed.unique()):
            data = folds[(folds.variant==variant)&(folds.seed==seed)]
            def get_row(method,degree,alpha):
                return data[(data.method==method)&(data.degree==degree)&(abs(data.alpha-alpha)<1e-12)].mse.mean()
            b = get_row(base['method'],base['degree'],base['alpha'])
            c = get_row(best['method'],best['degree'],best['alpha'])
            print(f'Seed {seed}: current={b:.6f}, candidate={c:.6f}')
    if variant==2:
        print('Best penalty by degree:')
        print(valid.sort_values('mean_mse').groupby('degree',sort=True).first()[['alpha','mean_mse']].to_string())
