import logging
import os

def setup_logging(log_dir="logs", log_file="app.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    logger = logging.getLogger("BatteryApp")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger
