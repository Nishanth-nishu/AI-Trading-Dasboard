import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # API Keys (will be loaded from environment variables)
    POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # Currency pairs to analyze
    CURRENCY_PAIRS = ["USDJPY", "EURUSD", "GBPUSD", "AUDUSD"]
    
    # API Settings
    API_RATE_LIMIT_DELAY = 1  # seconds between API calls
    
    # Flask settings
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')
