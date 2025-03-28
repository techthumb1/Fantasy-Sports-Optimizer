from flask import Flask
from backend.models.database import db

def create_app():
    app = Flask(__name__)
    # Configuration...
    db.init_app(app)

    with app.app_context():
        # Import and register blueprints
        from backend.routes import main_routes
        # app.register_blueprint(main_routes)

        # Import models to ensure they are registered with SQLAlchemy
        from backend.models import database_models

    return app
