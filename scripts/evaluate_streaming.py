"""
Fast evaluator - single model at a time, unbuffered output.
Evaluates only BEST_PNEUMONIA_MODEL on the test set.
"""
import os, sys, json, gc
import numpy as np
from PIL import Image
import keras
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

BASE = r'c:\Users\Tanmay Kanagali\OneDrive\Documents\whatapp\HackFest'

def get_paths_labels(split):
    items = []
    for label_str, label_val in [('NORMAL', 0), ('PNEUMONIA', 1)]:
        d = os.path.join(BASE, 'chest_xray', split, label_str)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(('.jpeg', '.jpg', '.png')):
                items.append((os.path.join(d, f), label_val))
    return items

def preprocess(path):
    img = Image.open(path).convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
    return np.array(img, dtype=np.float32)

def evaluate_model(model, items, batch_size=16):
    preds, labels = [], []
    batch_imgs, batch_lbls = [], []
    for i, (path, lbl) in enumerate(items):
        batch_imgs.append(preprocess(path))
        batch_lbls.append(lbl)
        if len(batch_imgs) >= batch_size or i == len(items)-1:
            x = np.array(batch_imgs, dtype=np.float32)
            out = model.predict(x, verbose=0).flatten()
            preds.extend(out.tolist())
            labels.extend(batch_lbls)
            print(f"  Processed {len(preds)}/{len(items)}", flush=True)
            batch_imgs, batch_lbls = [], []
    return np.array(preds), np.array(labels)

test_items = get_paths_labels('test')
val_items  = get_paths_labels('val')
print(f"Test: {len(test_items)}, Val: {len(val_items)}", flush=True)

CANDIDATE_MODELS = [
    ('BEST_PNEUMONIA_MODEL',  r'HackFest\PneumoVision\backend\models\classification\best_models\BEST_PNEUMONIA_MODEL.keras'),
    ('best_final_model',      r'HackFest\PneumoVision\backend\models\classification\best_models\best_final_model.keras'),
    ('improved_best_model',   r'HackFest\PneumoVision\backend\models\classification\best_models\improved_best_model.keras'),
    ('viral_aware_stage2',    r'HackFest\PneumoVision\backend\models\classification\viral_aware\viral_aware_stage2.keras'),
    ('imagenet_stage2',       r'HackFest\PneumoVision\backend\models\classification\imagenet_transfer\imagenet_stage2.keras'),
]

results = []
for mname, mrel in CANDIDATE_MODELS:
    mpath = os.path.join(BASE, mrel)
    if not os.path.exists(mpath):
        print(f"SKIP: {mname}", flush=True)
        continue
    print(f"\n== Evaluating: {mname} ==", flush=True)
    try:
        model = keras.models.load_model(mpath)

        # Val threshold tuning
        print("  Val predictions...", flush=True)
        val_preds, y_val = evaluate_model(model, val_items, batch_size=8)
        best_thresh, best_f1 = 0.5, -1
        for t in np.linspace(0.2, 0.85, 14):
            vb = (val_preds >= t).astype(int)
            vf = f1_score(y_val, vb, zero_division=0)
            if vf > best_f1:
                best_f1, best_thresh = vf, t

        # Test evaluation
        print("  Test predictions...", flush=True)
        test_preds, y_test = evaluate_model(model, test_items, batch_size=16)
        tb = (test_preds >= best_thresh).astype(int)

        acc  = accuracy_score(y_test, tb)
        prec = precision_score(y_test, tb, zero_division=0)
        rec  = recall_score(y_test, tb, zero_division=0)
        f1   = f1_score(y_test, tb, zero_division=0)
        try:    auc = roc_auc_score(y_test, test_preds)
        except: auc = 0.5
        tn, fp, fn, tp = confusion_matrix(y_test, tb).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        r = dict(model_name=mname, path=mpath, threshold=round(float(best_thresh),3),
                 accuracy=round(float(acc),4), precision=round(float(prec),4),
                 recall=round(float(rec),4), specificity=round(float(spec),4),
                 f1=round(float(f1),4), auc=round(float(auc),4),
                 tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))
        results.append(r)
        print(f"  RESULT: Acc={acc*100:.2f}% F1={f1:.4f} Recall={rec*100:.2f}% Spec={spec*100:.2f}% AUC={auc:.4f} Thresh={best_thresh:.2f}", flush=True)
        print(f"  TP={tp} TN={tn} FP={fp} FN={fn}", flush=True)
        del model; gc.collect()
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)

results.sort(key=lambda r: (r['f1'], r['auc']), reverse=True)
print("\n=== RANKING ===", flush=True)
for i, r in enumerate(results):
    print(f"{i+1}. {r['model_name']} Acc={r['accuracy']*100:.2f}% F1={r['f1']:.4f} Recall={r['recall']*100:.2f}% AUC={r['auc']:.4f}", flush=True)

out = os.path.join(BASE, 'scripts', 'benchmark_results.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out}", flush=True)
if results:
    print(f"BEST: {results[0]['model_name']}", flush=True)
