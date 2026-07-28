import numpy as np

def compute_oscr(known_scores, unknown_scores, known_labels_correct):
    """
    Computes Open Set Classification Rate (OSCR).
    known_scores: Array of max confidence scores for known test samples.
    unknown_scores: Array of max confidence scores for unknown test samples.
    known_labels_correct: Boolean array if the known sample was classified correctly.
    """
    thresholds = np.unique(np.concatenate([known_scores, unknown_scores]))
    thresholds = np.sort(thresholds)[::-1] # Descending
    
    fprs = []
    ccrs = []
    
    num_unknown = len(unknown_scores)
    num_known = len(known_scores)
    
    if num_unknown == 0 or num_known == 0:
        return 0.0
        
    for tau in thresholds:
        # False Positive Rate: fraction of unknown > tau
        fp = np.sum(unknown_scores >= tau)
        fpr = fp / num_unknown
        
        # Correct Classification Rate: fraction of known > tau AND correctly classified
        tp = np.sum((known_scores >= tau) & known_labels_correct)
        ccr = tp / num_known
        
        fprs.append(fpr)
        ccrs.append(ccr)
        
    # Compute OSCR as Area Under the Curve
    oscr = np.trapezoid(ccrs, fprs)
    return oscr
