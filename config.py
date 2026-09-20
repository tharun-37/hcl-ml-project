import os
import tempfile

def _is_writable(path):
    try:
        with open(path, 'a'):
            pass
        os.remove(path)
        return True
    except OSError:
        return False

class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    IS_SERVERLESS = os.environ.get('VERCEL', '0') == '1' or not _is_writable(
        os.path.join(BASE_DIR, '.write_test')
    )
    _WRITABLE_DIR = tempfile.gettempdir() if IS_SERVERLESS else BASE_DIR
    LOG_DIR = os.environ.get('LOG_DIR', os.path.abspath(os.path.join(_WRITABLE_DIR, 'logs')))
    LOG_FILE = 'battery_rul_app.log'
    DATABASE_PATH = os.path.join(_WRITABLE_DIR, 'predictions.db')
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')
    DATASET_PATH = os.path.join(BASE_DIR, 'data', 'oxford_battery_processed.csv')
    RAW_DATASET_PATH = os.path.join(BASE_DIR, 'data', 'Oxford_Battery_Degradation_Dataset_1.mat')
    STATIC_DIR = os.path.join(BASE_DIR, 'static')
    SECRET_KEY = os.environ.get('SECRET_KEY', 'battery-degradation-xai-secret-key-2026')
