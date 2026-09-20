import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import xgboost as xgb
import shap
from config import Config

def train_and_save():
    csv_path = Config.DATASET_PATH
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {df.shape}")
    
    # Feature Engineering
    df['Voltage_Drop_Rate'] = (df['Max_Voltage_V'] - df['Min_Voltage_V']) / (df['Discharge_Duration_hr'] + 1e-6)
    df['Capacity_Fade_Pct'] = 100.0 - df['SOH']
    df['Internal_Resist_Proxy'] = (df['Max_Voltage_V'] - df['Mean_Voltage_V']) / 0.740
    
    # 0 = Healthy (SOH >= 80%), 1 = Near End-of-Life (SOH < 80%)
    df['EOL_Class'] = (df['SOH'] < 80.0).astype(int)
    
    feature_cols = [
        'Cycle',
        'Discharge_Capacity_mAh',
        'Mean_Voltage_V',
        'Min_Voltage_V',
        'Max_Voltage_V',
        'Std_Voltage_V',
        'Mean_Temp_C',
        'Max_Temp_C',
        'Discharge_Duration_hr',
        'Energy_Wh',
        'Voltage_Drop_Rate',
        'Internal_Resist_Proxy'
    ]
    
    X = df[feature_cols]
    y_rul = df['RUL']
    y_cls = df['EOL_Class']
    
    X_train, X_test, y_train_rul, y_test_rul, y_train_cls, y_test_cls = train_test_split(
        X, y_rul, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    
    # 1. Regression Models (excluding LightGBM and Gradient Boosting)
    reg_models = {
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost Regressor': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'Support Vector Regressor (SVR)': SVR(kernel='rbf', C=100, epsilon=10)
    }
    
    trained_reg_models = {}
    reg_results = []
    
    for name, model in reg_models.items():
        model.fit(X_train, y_train_rul)
        preds = model.predict(X_test)
        
        r2 = float(r2_score(y_test_rul, preds))
        mse = float(mean_squared_error(y_test_rul, preds))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test_rul, preds))
        
        trained_reg_models[name] = model
        reg_results.append({
            'name': name,
            'r2': round(r2, 4),
            'rmse': round(rmse, 2),
            'mae': round(mae, 2)
        })
        print(f"[Regression: {name}] R2: {r2:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")
        
    # Sort regression models descending by R2
    reg_results = sorted(reg_results, key=lambda x: x['r2'], reverse=True)
    sorted_trained_reg_models = {r['name']: trained_reg_models[r['name']] for r in reg_results}
    sorted_reg_metrics = {r['name']: {'r2': r['r2'], 'rmse': r['rmse'], 'mae': r['mae']} for r in reg_results}
    
    # 2. Classification Models (excluding Gradient Boosting)
    cls_models = {
        'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Support Vector Classifier (SVC)': SVC(kernel='rbf', random_state=42)
    }
    
    trained_cls_models = {}
    cls_results = []
    conf_matrices = {}
    
    for name, model in cls_models.items():
        model.fit(X_train, y_train_cls)
        preds = model.predict(X_test)
        
        acc = float(accuracy_score(y_test_cls, preds))
        prec = float(precision_score(y_test_cls, preds, zero_division=0))
        rec = float(recall_score(y_test_cls, preds, zero_division=0))
        f1 = float(f1_score(y_test_cls, preds, zero_division=0))
        cm = confusion_matrix(y_test_cls, preds).tolist()
        
        trained_cls_models[name] = model
        cls_results.append({
            'name': name,
            'accuracy': round(acc * 100, 2),
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'f1': round(f1 * 100, 2)
        })
        conf_matrices[name] = cm
        print(f"[Classification: {name}] Accuracy: {acc*100:.2f}% | F1: {f1*100:.2f}%")
        
    # Sort classification models descending by Accuracy
    cls_results = sorted(cls_results, key=lambda x: x['accuracy'], reverse=True)
    sorted_trained_cls_models = {c['name']: trained_cls_models[c['name']] for c in cls_results}
    sorted_cls_metrics = {c['name']: {'accuracy': c['accuracy'], 'precision': c['precision'], 'recall': c['recall'], 'f1': c['f1']} for c in cls_results}

    # SHAP Explanations on Best Regressor (Random Forest)
    best_reg = sorted_trained_reg_models['Random Forest Regressor']
    explainer = shap.TreeExplainer(best_reg)
    shap_values = explainer.shap_values(X_test)
    
    feature_importances = []
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    total_shap = np.sum(mean_abs_shap) if np.sum(mean_abs_shap) > 0 else 1.0
    
    for feat, imp in zip(feature_cols, mean_abs_shap):
        pct = float((imp / total_shap) * 100.0)
        feature_importances.append({
            'feature': feat,
            'importance': round(pct, 2)
        })
    feature_importances = sorted(feature_importances, key=lambda x: x['importance'], reverse=True)
    
    df.to_csv(Config.DATASET_PATH, index=False)
    
    package = {
        'reg_models': sorted_trained_reg_models,
        'reg_metrics': sorted_reg_metrics,
        'cls_models': sorted_trained_cls_models,
        'cls_metrics': sorted_cls_metrics,
        'conf_matrices': conf_matrices,
        'features': feature_cols,
        'importances': feature_importances,
        'X_train_summary': X_train.describe().to_dict(),
        'X_train_sample': X_train.sample(min(100, len(X_train)), random_state=42),
        'X_test': X_test,
        'y_test_rul': y_test_rul,
        'y_test_cls': y_test_cls
    }
    
    model_pkl_path = Config.MODEL_PATH
    os.makedirs(os.path.dirname(model_pkl_path), exist_ok=True)
    with open(model_pkl_path, 'wb') as f:
        pickle.dump(package, f)
    print(f"Model package saved successfully to {model_pkl_path}")

if __name__ == '__main__':
    train_and_save()
