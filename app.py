from flask import Flask, render_template, request, jsonify, send_file
import pickle
import pandas as pd
import numpy as np
import sqlite3
import io
import os
import sys
import logging
from datetime import datetime
from pydantic import ValidationError
import shap

from config import Config
from validators import BatteryPredictionRequest

def setup_logging(log_dir="logs", log_file="app.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    app_logger = logging.getLogger("BatteryApp")
    app_logger.setLevel(logging.INFO)
    if not app_logger.handlers:
        try:
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            app_logger.addHandler(file_handler)
        except OSError:
            pass
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        app_logger.addHandler(console_handler)
    return app_logger

logger = setup_logging(log_dir=Config.LOG_DIR, log_file=Config.LOG_FILE)

app = Flask(__name__)
app.config.from_object(Config)

def init_db():
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                model_name TEXT,
                cycle REAL,
                capacity REAL,
                mean_voltage REAL,
                mean_temp REAL,
                energy REAL,
                soh REAL,
                predicted_rul INTEGER,
                health_status TEXT,
                strategy_recommendation TEXT
            )
        ''')
        conn.commit()

        cursor.execute('PRAGMA table_info(history)')
        existing = {row[1] for row in cursor.fetchall()}
        wanted = {
            'timestamp': 'TEXT',
            'model_name': 'TEXT',
            'cycle': 'REAL',
            'capacity': 'REAL',
            'mean_voltage': 'REAL',
            'mean_temp': 'REAL',
            'energy': 'REAL',
            'soh': 'REAL',
            'predicted_rul': 'INTEGER',
            'health_status': 'TEXT',
            'strategy_recommendation': 'TEXT',
        }
        for col, col_type in wanted.items():
            if col not in existing:
                cursor.execute(f'ALTER TABLE history ADD COLUMN {col} {col_type}')
        conn.commit()
        conn.close()
        logger.info(f"Database initialized successfully at {Config.DATABASE_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}", exc_info=True)

init_db()

def load_data():
    try:
        with open(Config.MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
        return (
            data.get('reg_models', {}),
            data.get('reg_metrics', {}),
            data.get('cls_models', {}),
            data.get('cls_metrics', {}),
            data.get('importances', []),
            data.get('features', []),
            data.get('X_train_sample', None)
        )
    except Exception as e:
        logger.error(f"Error loading model package: {str(e)}", exc_info=True)
        return {}, {}, {}, {}, [], [], None

def log_prediction(model_name, cycle, capacity, mean_v, mean_t, energy, soh, rul, status, strategy):
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO history (timestamp, model_name, cycle, capacity, mean_voltage, mean_temp, energy, soh, predicted_rul, health_status, strategy_recommendation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), model_name, cycle, capacity, mean_v, mean_t, energy, soh, rul, status, strategy))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log prediction to DB: {str(e)}", exc_info=True)

def get_history(limit=6):
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT timestamp, model_name, cycle, capacity, soh, predicted_rul, health_status, strategy_recommendation FROM history ORDER BY id DESC LIMIT ?', (limit,))
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"Failed to fetch history: {str(e)}", exc_info=True)
        return []

@app.route('/')
def home():
    try:
        reg_models, reg_metrics, cls_models, cls_metrics, importances, _, _ = load_data()
        reg_model_names = list(reg_models.keys())
        history = get_history(6)
        return render_template(
            'index.html',
            models=reg_model_names,
            reg_metrics=reg_metrics,
            cls_metrics=cls_metrics,
            importances=importances,
            history=history
        )
    except Exception as e:
        logger.error(f"Error rendering home: {str(e)}", exc_info=True)
        return render_template('error.html', code=500, title="Internal Server Error", message="Unable to load battery analytics dashboard."), 500

@app.route('/health')
def health():
    status = {'status': 'healthy', 'timestamp': datetime.now().isoformat(), 'checks': {}}
    all_healthy = True
    try:
        reg_models, _, _, _, _, _, _ = load_data()
        if reg_models:
            status['checks']['model_package'] = 'loaded'
        else:
            status['checks']['model_package'] = 'failed'
            all_healthy = False
    except Exception as e:
        status['checks']['model_package'] = f'error: {str(e)}'
        all_healthy = False

    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT count(*) FROM history')
        count = cursor.fetchone()[0]
        conn.close()
        status['checks']['database'] = f'connected ({count} records)'
    except Exception as e:
        status['checks']['database'] = f'error: {str(e)}'
        all_healthy = False

    code = 200 if all_healthy else 503
    if not all_healthy:
        status['status'] = 'unhealthy'
    return jsonify(status), code

@app.route('/predict', methods=['POST'])
def predict():
    try:
        req_data = request.get_json(silent=True)
        if not req_data:
            return jsonify({'success': False, 'error': 'Invalid or missing JSON payload'}), 400

        try:
            validated = BatteryPredictionRequest(**req_data)
        except ValidationError as ve:
            errors = [f"{err['loc'][0]}: {err['msg']}" for err in ve.errors()]
            return jsonify({'success': False, 'error': 'Validation Error: ' + ', '.join(errors)}), 422

        reg_models, _, cls_models, _, _, feature_cols, X_sample = load_data()
        chosen_model_name = validated.model_name if validated.model_name in reg_models else 'Random Forest Regressor'
        reg_model = reg_models.get(chosen_model_name)

        if not reg_model:
            return jsonify({'success': False, 'error': f'Model {chosen_model_name} not available'}), 500

        v_drop_rate = (validated.max_voltage - validated.min_voltage) / (validated.discharge_duration + 1e-6)
        ir_proxy = (validated.max_voltage - validated.mean_voltage) / 0.740

        input_dict = {
            'Cycle': validated.cycle,
            'Discharge_Capacity_mAh': validated.discharge_capacity,
            'Mean_Voltage_V': validated.mean_voltage,
            'Min_Voltage_V': validated.min_voltage,
            'Max_Voltage_V': validated.max_voltage,
            'Std_Voltage_V': validated.std_voltage,
            'Mean_Temp_C': validated.mean_temp,
            'Max_Temp_C': validated.max_temp,
            'Discharge_Duration_hr': validated.discharge_duration,
            'Energy_Wh': validated.energy,
            'Voltage_Drop_Rate': v_drop_rate,
            'Internal_Resist_Proxy': ir_proxy
        }
        
        input_df = pd.DataFrame([input_dict])[feature_cols]
        raw_rul = float(reg_model.predict(input_df)[0])
        predicted_rul = max(0, int(round(raw_rul)))
        
        nominal_cap = 740.0
        soh = round((validated.discharge_capacity / nominal_cap) * 100.0, 2)
        
        # Determine Fleet Status and AGV Maintenance Strategy
        if soh >= 90:
            health_status = 'Optimal (Fresh Cell)'
            health_color = '#10b981'
            strategy = 'Standard Deployment: Full heavy-load AGV & forklift duty cycles.'
        elif soh >= 80:
            health_status = 'Healthy (Normal Degradation)'
            health_color = '#2563eb'
            strategy = 'Operational: Routine monitoring; prioritize for continuous shift operations.'
        elif soh >= 75:
            health_status = 'Degraded (Near End-of-Life)'
            health_color = '#f59e0b'
            strategy = 'Scheduled Rotation: Rotate to light-duty transport; schedule replacement in 200 cycles to avoid mid-shift stall.'
        else:
            health_status = 'Critical (End-of-Life Reached)'
            health_color = '#ef4444'
            strategy = 'Immediate Replacement: Decommission cell immediately; repurpose for secondary stationary storage or recycle.'

        # Explainable AI (SHAP breakdown)
        local_shap = []
        try:
            if hasattr(reg_model, 'estimators_'):
                explainer = shap.TreeExplainer(reg_model)
                sv = explainer.shap_values(input_df)[0]
            else:
                if X_sample is not None:
                    explainer = shap.KernelExplainer(reg_model.predict, X_sample.iloc[:20])
                    sv = explainer.shap_values(input_df)[0]
                else:
                    sv = np.zeros(len(feature_cols))

            for feat, val, s_val in zip(feature_cols, input_df.iloc[0], sv):
                local_shap.append({
                    'feature': feat,
                    'value': round(float(val), 3),
                    'impact': round(float(s_val), 1),
                    'direction': 'increases_rul' if s_val > 0 else 'reduces_rul'
                })
            local_shap = sorted(local_shap, key=lambda x: abs(x['impact']), reverse=True)
        except Exception as e:
            logger.warning(f"SHAP local explanation fallback: {str(e)}")

        log_prediction(
            chosen_model_name,
            validated.cycle,
            validated.discharge_capacity,
            validated.mean_voltage,
            validated.mean_temp,
            validated.energy,
            soh,
            predicted_rul,
            health_status,
            strategy
        )

        return jsonify({
            'success': True,
            'predicted_rul': predicted_rul,
            'soh': soh,
            'health_status': health_status,
            'health_color': health_color,
            'model_used': chosen_model_name,
            'strategy_recommendation': strategy,
            'local_shap': local_shap[:5],
            'estimated_eol_cycle': int(validated.cycle + predicted_rul)
        })

    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': 'Server error: ' + str(e)}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400
        
        df = pd.read_csv(file)
        reg_models, _, _, _, _, feature_cols, _ = load_data()
        model_name = request.form.get('model_name', 'Random Forest Regressor')
        model = reg_models.get(model_name, list(reg_models.values())[0])
        
        # Calculate engineered features if missing
        if 'Voltage_Drop_Rate' not in df.columns:
            df['Voltage_Drop_Rate'] = (df['Max_Voltage_V'] - df['Min_Voltage_V']) / (df['Discharge_Duration_hr'] + 1e-6)
        if 'Internal_Resist_Proxy' not in df.columns:
            df['Internal_Resist_Proxy'] = (df['Max_Voltage_V'] - df['Mean_Voltage_V']) / 0.740
            
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            return jsonify({'success': False, 'error': f'Missing columns: {", ".join(missing)}'}), 400
        
        preds = model.predict(df[feature_cols])
        df['Predicted_RUL_Cycles'] = [max(0, int(round(p))) for p in preds]
        df['Estimated_SOH_Pct'] = [(c / 740.0) * 100.0 for c in df['Discharge_Capacity_mAh']]
        df['Fleet_Health_Status'] = ['Healthy' if s >= 80.0 else 'Near EOL (Replace)' for s in df['Estimated_SOH_Pct']]
        
        output = io.BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='battery_phm_fleet_predictions.csv'
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
