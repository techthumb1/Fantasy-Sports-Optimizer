from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy(app)

# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<User {self.username}>'

# Initialize the database
def init_db():
    with app.app_context():
        if not os.path.exists('sports_optimizer.db'):
            db.create_all()
            print("Database created successfully!")
        else:
            print("Database already exists.")

init_db()