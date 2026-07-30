import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc
import logging

def binary_open_set_metrics(y_true_unknown, unknown_scores):
    """
    Computes extended open-set recognition metrics.
    y_true_unknown: array of 0s and 1s, where 1 means UNKNOWN.
    unknown_scores: array of floats, where higher means MORE LIKELY TO BE UNKNOWN.
    
    Returns a dictionary of metrics.
    """
    metrics = {}
    y_true = np.array(y_true_unknown)
    scores = np.array(unknown_scores)
    
    if len(np.unique(y_true)) < 2:
        return {
            'AUROC': None,
            'AUPR_Out': None,
            'FPRat95_TPR': None,
            'TPRat1_FPR': None,
            'detection_error': None,
            'falsealarmsper10kat95tpr': None
        }

    # AUROC
    metrics['AUROC'] = roc_auc_score(y_true, scores)
    
    # AUPR_Out (Precision-Recall curve where positive class = unknown)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    metrics['AUPR_Out'] = auc(recall, precision)
    
    # FPRat95_TPR and TPRat1_FPR
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    
    # Find FPR at TPR >= 0.95
    idx_tpr95 = np.argmax(tpr >= 0.95)
    metrics['FPRat95_TPR'] = fpr[idx_tpr95]
    metrics['falsealarmsper10kat95tpr'] = metrics['FPRat95_TPR'] * 10000.0
    
    # Find TPR at FPR <= 0.01 (1%)
    # we want the largest TPR where FPR is <= 0.01
    valid_idx = np.where(fpr <= 0.01)[0]
    if len(valid_idx) > 0:
        idx_fpr1 = valid_idx[-1]
        metrics['TPRat1_FPR'] = tpr[idx_fpr1]
    else:
        metrics['TPRat1_FPR'] = 0.0
        
    # Detection Error: 0.5 * (1 - TPR) + 0.5 * FPR
    # Typically taken at the threshold that minimizes it
    det_errors = 0.5 * (1.0 - tpr) + 0.5 * fpr
    metrics['detection_error'] = np.min(det_errors)
    
    return metrics

def ensure_unknown_score_direction(known_scores, unknown_scores, score_name="anomaly_score"):
    """
    Ensures that unknown_scores are generally higher than known_scores.
    If the metric (like probability of being normal) is lower for unknowns, it inverts the scores.
    Returns: (adjusted_known_scores, adjusted_unknown_scores, score_was_inverted)
    """
    known = np.array(known_scores)
    unknown = np.array(unknown_scores)
    
    if len(known) == 0 or len(unknown) == 0:
        return known, unknown, False
        
    median_known = np.median(known)
    median_unknown = np.median(unknown)
    
    score_was_inverted = False
    
    if median_unknown < median_known:
        logging.info(f"[{score_name}] Median unknown ({median_unknown:.4f}) < Median known ({median_known:.4f}). Inverting scores...")
        # To invert safely, we can negate
        known = -known
        unknown = -unknown
        score_was_inverted = True
    else:
        logging.info(f"[{score_name}] Score direction OK: Median unknown ({median_unknown:.4f}) > Median known ({median_known:.4f}).")
        
    return known.tolist(), unknown.tolist(), score_was_inverted
