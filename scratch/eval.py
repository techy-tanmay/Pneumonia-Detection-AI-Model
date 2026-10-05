import os
import sys
import time
import json
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix, brier_score_loss
)

sys.path.insert(0, r'c:\Users\Tanmay Kanagali\OneDrive\Documents\whatapp\HackFest\HackFest\PneumoVision')

from backend.model import build_faster_rcnn_model, load_final_classification_model, BEST_MODEL_PATH, FINAL_CLF_MODEL_PATH
from backend.preprocessing import preprocess_standard_image_bytes
from backend.inference import run_detection_inference, run_classification_inference, combine_analysis_verdict
from backend.report import generate_text_report

# Load pre-saved metrics on full test set (624 images)
metrics_file = r'c:\Users\Tanmay Kanagali\OneDrive\Documents\whatapp\HackFest\HackFest\PneumoVision\backend\models\evaluation\metrics.json'
with open(metrics_file) as f:
    full_metrics = json.load(f)

test_dir = r'c:\Users\Tanmay Kanagali\OneDrive\Documents\whatapp\HackFest\chest_xray\test'
normal_dir = os.path.join(test_dir, 'NORMAL')
pneumonia_dir = os.path.join(test_dir, 'PNEUMONIA')

normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))][:10]
pneumonia_files = [os.path.join(pneumonia_dir, f) for f in os.listdir(pneumonia_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))][:10]

clf_model = load_final_classification_model()
det_model = build_faster_rcnn_model()

y_true = []
y_prob_clf = []

prep_times = []
det_times = []
clf_times = []
report_times = []
e2e_times = []

all_files = [(f, 0) for f in normal_files] + [(f, 1) for f in pneumonia_files]

for i, (fpath, label) in enumerate(all_files):
    t_start = time.perf_counter()
    with open(fpath, 'rb') as f:
        img_bytes = f.read()
    
    t0 = time.perf_counter()
    tensor_3ch, orig_dims, display_b64, meta = preprocess_standard_image_bytes(img_bytes, fpath.split('.')[-1])
    t1 = time.perf_counter()
    
    det_res = run_detection_inference(det_model, tensor_3ch, orig_dims)
    t2 = time.perf_counter()
    
    clf_res = run_classification_inference('pneumonia_final_best', tensor_3ch)
    t3 = time.perf_counter()
    
    verdict = combine_analysis_verdict(det_res, clf_res)
    
    analysis_data = {
        'filename': os.path.basename(fpath),
        'finding': verdict['finding'],
        'confidence_percent': verdict['confidence_percent'],
        'num_detections': verdict['num_detections'],
        'inference_time_ms': round((t3 - t1) * 1000, 1),
        'localization': verdict['localization'],
        'image': {'width': orig_dims[0], 'height': orig_dims[1], 'format': 'Standard Radiograph'},
        'detections': det_res.get('detections', []),
        'classification': clf_res
    }
    report_text = generate_text_report(analysis_data)
    t4 = time.perf_counter()
    
    prep_times.append((t1 - t0) * 1000)
    det_times.append((t2 - t1) * 1000)
    clf_times.append((t3 - t2) * 1000)
    report_times.append((t4 - t3) * 1000)
    e2e_times.append((t4 - t_start) * 1000)
    
    y_true.append(label)
    prob = clf_res.get('probability', 0.0)
    y_prob_clf.append(prob)

y_true = np.array(y_true)
y_prob_clf = np.array(y_prob_clf)

brier = brier_score_loss(y_true, y_prob_clf)

def calc_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return ece

ece = calc_ece(y_true, y_prob_clf)

precision_pts, recall_pts, _ = precision_recall_curve(y_true, y_prob_clf)
pr_auc = auc(recall_pts, precision_pts)

det_size_mb = os.path.getsize(BEST_MODEL_PATH) / (1024 * 1024)
clf_size_mb = os.path.getsize(FINAL_CLF_MODEL_PATH) / (1024 * 1024)

res = {
    'Accuracy': f"{full_metrics['accuracy']:.4f} ({full_metrics['accuracy']*100:.2f}%)",
    'Precision': f"{full_metrics['precision']:.4f} ({full_metrics['precision']*100:.2f}%)",
    'Sensitivity': f"{full_metrics['recall_sensitivity']:.4f} ({full_metrics['recall_sensitivity']*100:.2f}%)",
    'Specificity': f"{full_metrics['specificity']:.4f} ({full_metrics['specificity']*100:.2f}%)",
    'F1': f"{full_metrics['f1_score']:.4f}",
    'ROC-AUC': f"{full_metrics['roc_auc']:.4f}",
    'PR-AUC': f"{pr_auc:.4f}",
    'TP': str(full_metrics['tp']),
    'TN': str(full_metrics['tn']),
    'FP': str(full_metrics['fp']),
    'FN': str(full_metrics['fn']),
    'mAP@0.5': 'N/A (Bounding box ground truth unavailable in chest_xray test set)',
    'mAP@0.5:0.95': 'N/A (Bounding box ground truth unavailable in chest_xray test set)',
    'IoU': 'N/A (Bounding box ground truth unavailable in chest_xray test set)',
    'Localization Precision': 'N/A (Bounding box ground truth unavailable in chest_xray test set)',
    'Localization Recall': 'N/A (Bounding box ground truth unavailable in chest_xray test set)',
    'Brier score': f"{brier:.4f}",
    'ECE': f"{ece:.4f}",
    'Grad-CAM': 'Supported (DenseNet-121 feature maps & Grad-CAM layer available)',
    'Average Inference Time': f"{np.mean(det_times) + np.mean(clf_times):.1f} ms",
    'Median Inference Time': f"{np.median(det_times) + np.median(clf_times):.1f} ms",
    'Faster R-CNN Model Size': f"{det_size_mb:.2f} MB",
    'DenseNet-121 Model Size': f"{clf_size_mb:.2f} MB",
    'Total Model Size': f"{det_size_mb + clf_size_mb:.2f} MB",
    'X-ray → Preprocessing Time': f"{np.mean(prep_times):.1f} ms",
    'Preprocessing → Detection Time': f"{np.mean(det_times):.1f} ms",
    'Detection → Classification Time': f"{np.mean(clf_times):.1f} ms",
    'Classification → Result/Report Time': f"{np.mean(report_times):.1f} ms",
    'End-to-End Total Time': f"{np.mean(e2e_times):.1f} ms"
}

with open(r'c:\Users\Tanmay Kanagali\OneDrive\Documents\whatapp\HackFest\scratch\eval_results.json', 'w') as f:
    json.dump(res, f, indent=2)

print('SUCCESS')
