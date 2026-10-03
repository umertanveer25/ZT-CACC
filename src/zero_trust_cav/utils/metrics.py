"""
Statistical Validation and Evaluation Metrics Suite.
Calculates paired t-tests, Cohen's d effect sizes, Wilcoxon signed-rank, and classification metrics.
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
    """Computes comprehensive classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc)
    }

def run_statistical_hypothesis_tests(proposed_scores: np.ndarray, baseline_scores: np.ndarray) -> Dict[str, Any]:
    """
    Runs paired Student's t-test, Wilcoxon signed-rank test, and computes Cohen's d effect size.
    """
    t_stat, t_pval = stats.ttest_rel(proposed_scores, baseline_scores)
    diff = proposed_scores - baseline_scores
    cohen_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)
    
    try:
        w_stat, w_pval = stats.wilcoxon(proposed_scores, baseline_scores)
    except Exception:
        w_stat, w_pval = 0.0, 1.0

    return {
        "t_statistic": float(t_stat),
        "t_p_value": float(t_pval),
        "cohens_d": float(cohen_d),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p_value": float(w_pval),
        "is_significant_p01": bool(t_pval < 0.01)
    }
