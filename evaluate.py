"""
evaluate.py
Reads outputs/cv_results.json and produces summary tables with manual metric calculations:
- TP, TN, FP, FN
- TPR, TNR, FPR, FNR
- Accuracy, Balanced Accuracy
- Precision, Recall, F1, Error Rate
- TSS, HSS
- ROC & AUC plots (some are produced during training)
"""
import os, json
import pandas as pd
import numpy as np

OUT_DIR = "outputs"
with open(os.path.join(OUT_DIR, "cv_results.json"), "r") as f:
    results = json.load(f)

def metrics_from_confusion(tp, tn, fp, fn):
    tp, tn, fp, fn = map(float, [tp, tn, fp, fn])
    P = tp + fn
    N = tn + fp
    TPR = tp / P if P>0 else 0
    TNR = tn / N if N>0 else 0
    FPR = fp / N if N>0 else 0
    FNR = fn / P if P>0 else 0
    Accuracy = (tp + tn) / (P + N) if (P+N)>0 else 0
    Balanced = 0.5*(TPR + TNR)
    Precision = tp / (tp + fp) if (tp+fp)>0 else 0
    Recall = TPR
    F1 = 2*Precision*Recall/(Precision+Recall) if (Precision+Recall)>0 else 0
    Error = 1 - Accuracy
    TSS = TPR - FPR
    HSS = (2*(tp*tn - fp*fn)) / ((tp+fn)*(fn+tn) + (tp+fp)*(fp+tn)) if ((tp+fn)*(fn+tn) + (tp+fp)*(fp+tn))!=0 else 0
    return dict(P=int(P), N=int(N), TPR=TPR, TNR=TNR, FPR=FPR, FNR=FNR, Accuracy=Accuracy, Balanced=Balanced, Precision=Precision, Recall=Recall, F1=F1, Error=Error, TSS=TSS, HSS=HSS)

# Build tables
dfs = {}
for model in results:
    rows = []
    for foldres in results[model]:
        tp = foldres['tp']; tn = foldres['tn']; fp = foldres['fp']; fn = foldres['fn']
        m = metrics_from_confusion(tp, tn, fp, fn)
        m['fold'] = foldres['fold']
        m['auc'] = foldres.get('auc', None)
        rows.append(m)
    df = pd.DataFrame(rows).sort_values('fold').set_index('fold')
    dfs[model] = df
    # Save per-model table
    df.to_csv(os.path.join(OUT_DIR, f"{model}_per_fold_metrics.csv"))
    # Save average row
    df.loc['average'] = df.mean(numeric_only=True)
    df.to_csv(os.path.join(OUT_DIR, f"{model}_per_fold_metrics_with_avg.csv"))

# Print brief summary
for model, df in dfs.items():
    print("Model:", model)
    print(df.loc[:, ['Accuracy','Balanced','Precision','Recall','F1','auc']].mean())
    print()

print("Detailed CSVs saved in outputs/")
