from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np

def calculate_metrics(y_true, y_pred, y_scores=None):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class': {
            'precision': per_class_p.tolist(),
            'recall': per_class_r.tolist(),
            'f1': per_class_f1.tolist(),
            'support': support.tolist()
        }
    }
    
    if y_scores is not None and len(np.unique(y_true)) > 1:
        try:
            # Multi-class AUROC or binary AUROC depending on shape
            if y_scores.ndim > 1 and y_scores.shape[1] > 1:
                auroc = roc_auc_score(y_true, y_scores, multi_class='ovr')
            else:
                auroc = roc_auc_score(y_true, y_scores)
            metrics['auroc'] = auroc
        except Exception as e:
            metrics['auroc'] = float('nan')
            
    return metrics
