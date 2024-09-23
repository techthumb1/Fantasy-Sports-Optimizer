import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

class Config:
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.getenv('SECRET_KEY')
    SLEEPER_API_BASE_URL = 'https://api.sleeper.app/v1'
    SLEEPER_API_TIMEOUT = 10
    SLEEPER_API_RETRIES = 5
    SLEEPER_API_BACKOFF_FACTOR = 0.3
    SLEEPER_API_USER_AGENT = 'Sleeper API Client'
    SLEEPER_API_HEADERS = {
        'Content-Type': 'application/json',
        'User-Agent': SLEEPER_API_USER_AGENT
    }


# Set the cache duration for the Sleeper API data (e.g., 1 day)
CACHE_DURATION = timedelta(days=1)
