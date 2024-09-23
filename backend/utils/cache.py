import time
import os
import pickle

CACHE_FILE_PATH = 'cache.pkl'
CACHE_TTL = 60 * 60

def load_cache():
    if os.path.exists(CACHE_FILE_PATH):
        with open(CACHE_FILE_PATH, 'rb') as f:
            cache_data = pickle.load(f)
            if time.time() - cache_data['timestamp'] < CACHE_TTL:
                return cache_data['data']
    return None

def save_cache(data):
    with open(CACHE_FILE_PATH, 'wb') as f:
        pickle.dump({'timestamp': time.time(), 'data': data}, f)
