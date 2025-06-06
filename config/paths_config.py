import os

# Root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Processed data directory
PROCESSED_DIR = os.path.join(ROOT_DIR, "artifacts", "processed")
MODEL_DIR = os.path.join(ROOT_DIR, "artifacts", "model")
# Ensure it exists
os.makedirs(PROCESSED_DIR, exist_ok=True)