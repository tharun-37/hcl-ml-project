import os

class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    LOG_FILE = 'battery_rul_app.log'
    DATABASE_PATH = os.path.join(BASE_DIR, 'predictions.db')
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')
    DATASET_PATH = os.path.join(BASE_DIR, 'data', 'oxford_battery_processed.csv')
    RAW_DATASET_PATH = os.path.join(BASE_DIR, 'data', 'Oxford_Battery_Degradation_Dataset_1.mat')
    STATIC_DIR = os.path.join(BASE_DIR, 'static')
    SECRET_KEY = os.environ.get('SECRET_KEY', 'battery-degradation-xai-secret-key-2026')
