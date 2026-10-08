import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from polynomial import predict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', type=int, choices=[1,2], required=True)
    ap.add_argument('--data-dir', default='BT2024197')
    ap.add_argument('--output-dir', default='submission')
    ap.add_argument('--sample', type=Path)
    args = ap.parse_args()
    test = pd.read_csv(Path(args.data_dir)/f'BT2024197_test_var{args.variant}.csv')
    expected = [f'x{i}' for i in range(1,7 if args.variant==1 else 4)]
    if list(test.columns) != expected or not np.isfinite(test.to_numpy()).all():
        raise ValueError('Test columns or values are invalid')
    with np.load(f'results/final_model_var{args.variant}.npz', allow_pickle=False) as saved:
        model = dict(saved)
    test['y'] = predict(model,test.to_numpy())
    cols = list(pd.read_csv(args.sample,nrows=0).columns) if args.sample else ['y']
    if 'y' not in cols or any(c not in test.columns for c in cols):
        raise ValueError('Unrecognized sample format')
    Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    test[cols].to_csv(Path(args.output_dir)/f'BT2024197_pred_var{args.variant}.csv',index=False,float_format='%.17g')

if __name__ == '__main__':
    main()
