# backend/models/__init__.py
from backend.models.database import db

# Lazy imports to prevent circular dependency
def get_player_model():
    from database_models import Player
    return Player
