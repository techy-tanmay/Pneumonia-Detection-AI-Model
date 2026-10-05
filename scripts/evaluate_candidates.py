import os
import glob
import numpy as np
from PIL import Image
import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def load_data(split_dir):
    images = []
    labels = []
    filepaths = []
    for label_str, label_val in [('NORMAL', 0), ('PNEUMONIA', 1)]:
        dir_path = os.path.join(split_dir, label_str)
        if not os.path.exists(dir_path):
            continue
        for fname in sorted(os.listdir(dir_path)):
            if fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                fpath = os.path.join(dir_path, fname)
                filepaths.append(fpath)
                labels.append(label_val)
    return filepaths, np.array(labels)

def preprocess_image(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    return arr

print("Loading validation and test datasets...")
val_paths, y_val = load_data('chest_xray/val')
test_paths, y_test = load_data('chest_xray/test')

print(f"Val images: {len(val_paths)} (Normal: {(y_val==0).sum()}, Pneumonia: {(y_val==1).sum()})")
print(f"Test images: {len(test_paths)} (Normal: {(y_test==0).sum()}, Pneumonia: {(y_test==1).sum()})")

# Preload batches for speed
X_val = np.array([preprocess_image(p) for p in val_paths])
X_test = np.array([preprocess_image(p) for p in test_paths])

model_paths = glob.glob(r'HackFest\PneumoVision\backend\models\classification\**\*.keras', recursive=True)
print(f"Found {len(model_paths)} candidate models.")

results = []

for mp in sorted(model_paths):
    mname = os.path.basename(mp)
    try:
        model = keras.models.load_model(mp)
        val_preds_raw = model.predict(X_val, batch_size=16, verbose=0).flatten()
        test_preds_raw = model.predict(X_test, batch_size=32, verbose=0).flatten()
        
        # Determine best threshold on val set (or default 0.5)
        best_thresh = 0.5
        best_val_f1 = -1
        for thresh in np.linspace(0.2, 0.8, 13):
            val_bin = (val_preds_raw >= thresh).astype(int)
            val_f1 = f1_score(y_val, val_bin, zero_division=0)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_thresh = thresh
        
        # Test evaluation with fixed threshold
        test_bin = (test_preds_raw >= best_thresh).astype(int)
        acc = accuracy_score(y_test, test_bin)
        prec = precision_score(y_test, test_bin, zero_division=0)
        rec = recall_score(y_test, test_bin, zero_division=0)
        f1 = f1_score(y_test, test_bin, zero_division=0)
        try:
            auc = roc_auc_score(y_test, test_preds_raw)
        except Exception:
            auc = 0.5
            
        tn, fp, fn, tp = confusion_matrix(y_test, test_bin).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        res = {
            "model_name": mname,
            "path": mp,
            "threshold": round(float(best_thresh), 3),
            "val_f1": round(float(best_val_f1), 4),
            "accuracy": round(float(acc), 4),
            "recall": round(float(rec), 4),
            "specificity": round(float(spec), 4),
            "precision": round(float(prec), 4),
            "f1": round(float(f1), 4),
            "auc": round(float(auc), 4),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
        }
        results.append(res)
        print(f"[{mname}] Thresh: {best_thresh:.2f} | Acc: {acc:.4f} | Recall: {rec:.4f} | Spec: {spec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    except Exception as e:
        print(f"[{mname}] Error: {e}")

# Sort by test F1 and Accuracy
results.sort(key=lambda r: (r['f1'], r['accuracy'], r['auc']), reverse=True)
print("\n" + "="*80)
print("RANKED CANDIDATES:")
print("="*80)
for i, r in enumerate(results[:5]):
    print(f"{i+1}. {r['model_name']} -> Acc: {r['accuracy']*100:.2f}%, F1: {r['f1']:.4f}, Recall: {r['recall']*100:.2f}%, Spec: {r['specificity']*100:.2f}%, AUC: {r['auc']:.4f} (Thresh: {r['threshold']})")
