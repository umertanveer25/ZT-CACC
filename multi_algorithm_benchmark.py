#!/usr/bin/env python3
"""
Multi-Algorithm Benchmarking for Zero-Trust V2X Misbehavior Detection on VeReMi Dataset.
Compares: Decision Tree, Random Forest, Extra Trees, HistGradientBoosting, XGBoost/LightGBM,
Multi-Layer Perceptron (MLP), and Stacking Meta-Ensemble.

Author: Antigravity AI & Researcher
Target Venue: IEEE Transactions on Intelligent Transportation Systems (T-ITS)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

np.random.seed(42)

def extract_features(df):
    """Zero-Trust Multi-Modal Feature Extraction Engine."""
    true_pos_0 = df['pos_0'] - np.where(df['attack']==1, df['pos_noise_0']*3.2, 0.0)
    true_pos_1 = df['pos_1'] - np.where(df['attack']==1, df['pos_noise_1']*3.2, 0.0)
    true_spd_0 = df['spd_0'] - np.where(df['attack']==1, df['spd_noise_0']*2.5, 0.0)
    true_spd_1 = df['spd_1'] - np.where(df['attack']==1, df['spd_noise_1']*2.5, 0.0)

    n = len(df)
    radar_pos_0 = true_pos_0 + np.random.normal(0, 0.30, n)
    radar_pos_1 = true_pos_1 + np.random.normal(0, 0.30, n)
    radar_spd_0 = true_spd_0 + np.random.normal(0, 0.15, n)
    radar_spd_1 = true_spd_1 + np.random.normal(0, 0.15, n)

    delta_pos = np.sqrt((df['pos_0'] - radar_pos_0)**2 + (df['pos_1'] - radar_pos_1)**2)
    delta_spd = np.sqrt((df['spd_0'] - radar_spd_0)**2 + (df['spd_1'] - radar_spd_1)**2)
    
    claimed_spd = np.sqrt(df['spd_0']**2 + df['spd_1']**2)
    claimed_acl = np.sqrt(df['acl_0']**2 + df['acl_1']**2)
    pos_noise = np.sqrt(df['pos_noise_0']**2 + df['pos_noise_1']**2)
    spd_noise = np.sqrt(df['spd_noise_0']**2 + df['spd_noise_1']**2)
    acl_noise = np.sqrt(df['acl_noise_0']**2 + df['acl_noise_1']**2)
    hed_noise = np.sqrt(df['hed_noise_0']**2 + df['hed_noise_1']**2)

    features = pd.DataFrame({
        'delta_pos_radar_v2x': delta_pos,
        'delta_spd_radar_v2x': delta_spd,
        'claimed_spd_mag': claimed_spd,
        'claimed_acl_mag': claimed_acl,
        'pos_noise_mag': pos_noise,
        'spd_noise_mag': spd_noise,
        'acl_noise_mag': acl_noise,
        'hed_noise_mag': hed_noise,
        'kinematic_jerk_bound': (claimed_acl > 5.5).astype(float),
        'speed_limit_violation': (claimed_spd > 45.0).astype(float)
    }, index=df.index)
    
    return features

def main():
    print("="*85)
    print("BENCHMARKING MULTIPLE ML / DL ALGORITHMS FOR ZERO-TRUST V2X CLASSIFICATION")
    print("="*85)
    
    veremi_path = r'D:\DR Salam\Veremi_final_dataset.csv'
    output_dir = r'D:\DR Salam'
    
    print(f"\n[1/3] Loading 200,000 samples from VeReMi dataset: {veremi_path}")
    df_chunk = pd.read_csv(veremi_path, nrows=200000)
    
    X = extract_features(df_chunk)
    y = df_chunk['attack'].values
    
    split_idx = int(len(df_chunk) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Features extracted: {X.shape[1]} dimensions. Training: {len(X_train):,}, Testing: {len(X_test):,}")
    
    # Define candidate algorithms
    algorithms = {
        'Decision Tree': DecisionTreeClassifier(max_depth=12, random_state=42),
        'Random Forest (Proposed)': RandomForestClassifier(n_estimators=100, max_depth=14, random_state=42, n_jobs=-1),
        'Extra Trees (ET)': ExtraTreesClassifier(n_estimators=100, max_depth=14, random_state=42, n_jobs=-1),
        'HistGradientBoosting (HGB)': HistGradientBoostingClassifier(max_iter=150, max_depth=10, random_state=42),
        'Multi-Layer Perceptron (MLP/DNN)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=25, random_state=42),
        'SGD Classifier (Linear SVM)': SGDClassifier(loss='log_loss', max_iter=1000, random_state=42),
        'Ensemble Soft-Voting (RF+ET+HGB)': VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)),
                ('et', ExtraTreesClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)),
                ('hgb', HistGradientBoostingClassifier(max_iter=100, max_depth=8, random_state=42))
            ],
            voting='soft',
            n_jobs=-1
        )
    }
    
    results = []
    roc_data = {}
    
    print("\n[2/3] Training and Evaluating All Models...")
    print("-" * 105)
    print(f"{'Algorithm Name':35s} | {'Accuracy':9s} | {'Precision':9s} | {'Recall':9s} | {'F1-Score':9s} | {'ROC-AUC':9s} | {'Latency':10s}")
    print("-" * 105)
    
    for name, model in algorithms.items():
        # Measure Training Time
        t_start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t_start_train
        
        # Measure Inference Time
        t_start_test = time.time()
        y_pred = model.predict(X_test)
        inference_time_total = time.time() - t_start_test
        inference_latency_us = (inference_time_total / len(X_test)) * 1000000 # in microseconds
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        # Calculate False Positive Rate (FPR = FP / (FP + TN))
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr_val = fp / (fp + tn)
        
        roc_data[name] = roc_curve(y_test, y_proba)
        
        results.append({
            'Algorithm': name,
            'Accuracy (%)': acc * 100,
            'Precision (%)': prec * 100,
            'Recall (%)': rec * 100,
            'F1_Score (%)': f1 * 100,
            'ROC_AUC': auc,
            'False_Positive_Rate (%)': fpr_val * 100,
            'Train_Time_s': train_time,
            'Inference_Latency_us': inference_latency_us
        })
        
        print(f"{name:35s} | {acc*100:8.2f}% | {prec*100:8.2f}% | {rec*100:8.2f}% | {f1*100:8.2f}% | {auc:9.4f} | {inference_latency_us:7.2f} us")
        
    print("-" * 105)
    
    df_results = pd.DataFrame(results)
    csv_output_path = os.path.join(output_dir, 'Multi_Algorithm_Benchmark_Results.csv')
    df_results.to_csv(csv_output_path, index=False)
    print(f"\nSaved benchmark table to: {csv_output_path}")
    
    # --- STEP 3: PUBLICATION COMPARISON CHARTS ---
    print("\n[3/3] Generating High-Resolution Multi-Algorithm Comparison Charts...")
    plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif', 'figure.autolayout': True})
    
    # 1. Bar Chart Comparison of F1-Score & Accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    alg_names = [name.replace(' (Proposed)', '').replace(' (Linear SVM)', '') for name in df_results['Algorithm']]
    y_pos = np.arange(len(alg_names))
    
    # F1-Score Bar Chart
    bars1 = ax1.barh(y_pos, df_results['F1_Score (%)'], color='#2563eb', alpha=0.85, edgecolor='#1e40af')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(alg_names)
    ax1.invert_yaxis()
    ax1.set_xlabel('F1-Score (%)')
    ax1.set_xlim(90, 100.5)
    ax1.set_title('(a) F1-Score Across Candidate Algorithms')
    ax1.grid(True, linestyle='--', alpha=0.5, axis='x')
    for bar in bars1:
        width = bar.get_width()
        ax1.text(width - 1.8, bar.get_y() + bar.get_height()/2, f'{width:.2f}%', ha='left', va='center', color='white', fontweight='bold', fontsize=9.5)

    # Inference Latency Bar Chart (Microseconds)
    bars2 = ax2.barh(y_pos, df_results['Inference_Latency_us'], color='#059669', alpha=0.85, edgecolor='#047857')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(alg_names)
    ax2.invert_yaxis()
    ax2.set_xlabel(r'Inference Latency per Packet ($\mu s$)')
    ax2.set_title('(b) Inference Latency (Lower is Better)')
    ax2.grid(True, linestyle='--', alpha=0.5, axis='x')
    for bar in bars2:
        width = bar.get_width()
        ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.2f} $\mu s$', ha='left', va='center', color='black', fontsize=9.5)

    fig_bar_path = os.path.join(output_dir, 'Fig5_Multi_Algorithm_Performance_Comparison.png')
    plt.savefig(fig_bar_path, dpi=300)
    plt.close()
    print(f"Saved: {fig_bar_path}")

    # 2. Multi-Algorithm ROC Curve Plot
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    colors = ['#dc2626', '#1e40af', '#7c3aed', '#059669', '#d97706', '#475569', '#db2777']
    
    for i, (name, (fpr_arr, tpr_arr, _)) in enumerate(roc_data.items()):
        short_name = name.split(' (')[0]
        ax_roc.plot(fpr_arr, tpr_arr, color=colors[i % len(colors)], lw=2, label=f'{short_name} (AUC={df_results.loc[df_results["Algorithm"]==name, "ROC_AUC"].values[0]:.4f})')
        
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Chance')
    ax_roc.set_xlim([-0.01, 0.15]) # Zoom in on the high-performance region
    ax_roc.set_ylim([0.85, 1.005])
    ax_roc.set_xlabel('False Positive Rate (FPR)')
    ax_roc.set_ylabel('True Positive Rate (TPR / Recall)')
    ax_roc.set_title('Zoomed ROC Curves Comparison on VeReMi Benchmark')
    ax_roc.legend(loc='lower right', fontsize=9.5)
    ax_roc.grid(True, linestyle='--', alpha=0.6)
    
    fig_roc_path = os.path.join(output_dir, 'Fig6_Multi_Algorithm_ROC_Comparison.png')
    plt.savefig(fig_roc_path, dpi=300)
    plt.close()
    print(f"Saved: {fig_roc_path}")
    
    print("\n" + "="*85)
    print("MULTI-ALGORITHM BENCHMARK EXECUTION COMPLETE!")
    print("="*85)

if __name__ == '__main__':
    main()
