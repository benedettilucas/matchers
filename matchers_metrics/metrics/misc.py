import numpy as np

def compute_metrics(pred, gt, threshold=3.0):
    
    # Calcula erro entre os matches estimados e o ground truth
    dists = np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=-1)
    min_dists = dists.min(axis=1)
    indices = dists.argmin(axis=1)
    tp_mask = np.where(min_dists < threshold, indices, -1)
    total_estimativas = pred.shape[0]
    total_ground_truth = gt.shape[0]
    
    tp = np.sum(tp_mask != -1) # true positives: matches estimados corretamente
    fp = total_estimativas - tp # false positives: matches estimados incorretamente
    fn = total_ground_truth - tp # false negatives: matches existentes não correspondidos
    errors = min_dists[min_dists < threshold]

    # Métricas
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    mean_error = np.mean(errors) if len(errors) > 0 else np.nan

    return {
        f"precision@{round(threshold)}px": round(precision.copy().tolist()*100,2),
        f"recall@{round(threshold)}px": round(recall.copy().tolist()*100,2),
        "mean_error": round(mean_error.copy().tolist()*100,2),
        "tp": tp.copy().tolist(),
        "fp": fp.copy().tolist(),
        "fn": fn.copy().tolist(),
    }