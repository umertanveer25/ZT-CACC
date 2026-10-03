"""
Explainable AI (XAI) and TreeSHAP Interpretability Engine.
Generates publication-ready beeswarm, feature attribution bars, and forensic waterfalls.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import shap

class SHAPExplainer:
    """Wrapper for TreeSHAP calculation and attribution ranking."""
    def __init__(self, model, feature_names: list[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def compute_shap_values(self, X_sample: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Computes exact Shapley values and returns sorted feature ranking dataframe."""
        raw_shap = self.explainer.shap_values(X_sample)
        
        if isinstance(raw_shap, list):
            sv_attack = raw_shap[1]
        elif len(raw_shap.shape) == 3:
            sv_attack = raw_shap[:, :, 1]
        else:
            sv_attack = raw_shap

        mean_abs_shap = np.mean(np.abs(sv_attack), axis=0)
        total_shap = np.sum(mean_abs_shap)
        rel_importance = (mean_abs_shap / max(1e-8, total_shap)) * 100.0

        ranking_df = pd.DataFrame({
            'Feature_Name': self.feature_names,
            'Mean_Absolute_SHAP': mean_abs_shap,
            'Relative_Importance_Pct': rel_importance
        }).sort_values(by='Mean_Absolute_SHAP', ascending=False).reset_index(drop=True)

        return sv_attack, ranking_df
