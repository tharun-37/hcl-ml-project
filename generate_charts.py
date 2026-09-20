import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
from config import Config

def generate_all_charts():
    output_dir = os.path.join(Config.STATIC_DIR, 'charts')
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = Config.DATASET_PATH
    df = pd.read_csv(csv_path)
    
    model_pkl_path = Config.MODEL_PATH
    with open(model_pkl_path, 'rb') as f:
        pkg = pickle.load(f)
    
    reg_models = pkg['reg_models']
    reg_metrics = pkg['reg_metrics']
    cls_metrics = pkg['cls_metrics']
    conf_matrices = pkg['conf_matrices']
    rf_reg = reg_models['Random Forest Regressor']
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. Capacity Fade & SOH (%) Degradation Curve
    plt.figure(figsize=(6, 3.8), dpi=150)
    for cell in sorted(df['Cell'].unique()):
        cdf = df[df['Cell'] == cell]
        plt.plot(cdf['Cycle'], cdf['SOH'], marker='.', markersize=3, label=cell, alpha=0.85)
    plt.axhline(80, color='red', linestyle='--', linewidth=1.2, label='80% EOL Threshold')
    plt.title('Capacity Fade & SOH (%) Degradation Across Cells', fontsize=10, fontweight='bold', color='#0f172a')
    plt.xlabel('Cycle Count', fontsize=8)
    plt.ylabel('State of Health (%)', fontsize=8)
    plt.legend(ncol=3, fontsize=6.5, loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'soh_degradation.png'))
    plt.close()
    
    # 2. Actual vs Predicted RUL (Random Forest Regressor)
    feature_cols = pkg['features']
    X = df[feature_cols]
    y_true = df['RUL']
    y_pred = rf_reg.predict(X)
    
    plt.figure(figsize=(6, 3.8), dpi=150)
    plt.scatter(y_true, y_pred, color='#2563eb', alpha=0.6, s=16, edgecolors='none', label='Observed Telemetry')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Ideal 1:1 Fit')
    plt.title('Actual vs Predicted RUL (Cycles)', fontsize=10, fontweight='bold', color='#0f172a')
    plt.xlabel('Actual RUL (Cycles)', fontsize=8)
    plt.ylabel('Predicted RUL (Cycles)', fontsize=8)
    plt.legend(fontsize=7.5, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'))
    plt.close()
    
    # 3. Model Comparison: Sorted Regression R2 Benchmark
    plt.figure(figsize=(6, 3.8), dpi=150)
    r_names = list(reversed(list(reg_metrics.keys())))
    r2_vals = [reg_metrics[m]['r2'] * 100 for m in r_names]
    
    bars = plt.barh(r_names, r2_vals, color='#0ea5e9', height=0.55, edgecolor='#0284c7')
    plt.xlim(0, 115)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 1.2, bar.get_y() + bar.get_height()/2, f'{w:.1f}%', va='center', ha='left', fontsize=8, fontweight='bold', color='#0f172a')
    plt.title('Regression Models R2 Benchmark (%)', fontsize=10, fontweight='bold', color='#0f172a')
    plt.xlabel('R2 Score (%)', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    plt.close()
    
    # 4. Confusion Matrix (Healthy vs Near End-of-Life)
    cm = np.array(conf_matrices['Random Forest Classifier'])
    plt.figure(figsize=(6, 3.8), dpi=150)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Healthy (SOH>=80%)', 'Near EOL (SOH<80%)'],
                yticklabels=['Healthy', 'Near EOL'])
    plt.title('Classification Confusion Matrix (EOL Diagnosis)', fontsize=10, fontweight='bold', color='#0f172a')
    plt.ylabel('True Class', fontsize=8)
    plt.xlabel('Predicted Class', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 5. SHAP Feature Importance
    plt.figure(figsize=(6, 3.8), dpi=150)
    importances = pkg['importances']
    feats = [i['feature'].replace('_', ' ') for i in reversed(importances[:8])]
    vals = [i['importance'] for i in reversed(importances[:8])]
    
    bars = plt.barh(feats, vals, color='#10b981', height=0.55, edgecolor='#059669')
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.5, bar.get_y() + bar.get_height()/2, f'{w:.1f}%', va='center', ha='left', fontsize=7.5, color='#0f172a')
    plt.title('SHAP Feature Importance (Degradation Drivers)', fontsize=10, fontweight='bold', color='#0f172a')
    plt.xlabel('Mean |SHAP| Impact (%)', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_importance.png'))
    plt.close()
    
    print("All Project 8 charts regenerated successfully!")

if __name__ == '__main__':
    generate_all_charts()
