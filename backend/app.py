import os
import requests
import pandas as pd
import json
import numpy as np 
import xgboost as xgb
from dotenv import load_dotenv
from flask import Flask, jsonify, request
import tensorflow as tf
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from backend.api.sleeper import SleeperAPI
from backend.api.espn import ESPNAPI
from backend.utils.preprocessing import preprocess_data
from backend.utils.feature_engineering import apply_feature_engineering
from backend.models.xgboost_model import xgboost_model
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from backend.utils.formatter import pretty_print_json




app = Flask(__name__)

# Configuration for SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sports_optimizer.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

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

# Initialize the database
init_db()

# Load environment variables
load_dotenv()

# Define absolute paths for data
teams_file_path = os.path.join(os.path.dirname(__file__), 'teams_data.json')
scores_file_path = os.path.join(os.path.dirname(__file__), 'scores_data.json')

'***********************************************************************'
'''APIs'''
class SleeperAPI:
    SLEEPER_BASE_URL = "https://api.sleeper.app/v1"

    @staticmethod
    def get_trending_players(sport, trend_type="add", lookback_hours=24, limit=25):
        url = f"{SleeperAPI.SLEEPER_BASE_URL}/players/{sport}/trending/{trend_type}"
        params = {"lookback_hours": lookback_hours, "limit": limit}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch trending players: {response.status_code}")
            return None

    @staticmethod
    def get_players(sport):
        url = f"{SleeperAPI.SLEEPER_BASE_URL}/players/{sport}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch players: {response.status_code}")
            return None

    @staticmethod
    def get_player_stats(player_id):
        url = f"{SleeperAPI.SLEEPER_BASE_URL}/player/{player_id}/stats"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch player stats: {response.status_code}")
            return None

    @staticmethod
    def get_user(username_or_id):
        url = f"{SleeperAPI.SLEEPER_BASE_URL}/user/{username_or_id}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch user data: {response.status_code}")
            return None

    @staticmethod
    def get_leagues(user_id, sport, season):
        url = f"{SleeperAPI.SLEEPER_BASE_URL}/user/{user_id}/leagues/{sport}/{season}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch leagues: {response.status_code}")
            return None

    @staticmethod
    def get_league(league_id):
        url = f"{SleeperAPI.SLEEPER_BASE_URL}/league/{league_id}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch league data: {response.status_code}")
            return None

    @staticmethod
    def get_rosters(league_id):
        url = f"{SleeperAPI.SLEEPER_BASE_URL}/league/{league_id}/rosters"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch rosters: {response.status_code}")
            return None

    @staticmethod
    def get_users(league_id):
        url = f"{SleeperAPI.SLEEPER_BASE_URL}/league/{league_id}/users"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch league users: {response.status_code}")
            return None

    @staticmethod
    def get_matchups(league_id, week):
        url = f"{SleeperAPI.SLEEPER_BASE_URL}/league/{league_id}/matchups/{week}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch matchups: {response.status_code}")
            return None

    @staticmethod
    def get_draft(draft_id):
        url = f"{SleeperAPI.SLEEPER_BASE_URL}/draft/{draft_id}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch draft data: {response.status_code}")
            return None

    @staticmethod
    def get_draft_picks(draft_id):
        url = f"{SleeperAPI.SLEEPER_BASE_URL}/draft/{draft_id}/picks"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch draft picks: {response.status_code}")
            return None


class ESPNAPI:
    @staticmethod
    def get_scores(sport, league):
        if sport == "football" and league == "nfl":
            url = "http://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        elif sport == "basketball" and league == "nba":
            url = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        else:
            raise ValueError("Invalid sport or league provided.")

        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get('events', [])
        else:
            print(f"Failed to fetch scores: {response.status_code}")
            return None

    @staticmethod
    def get_teams(sport, league):
        if sport == "football" and league == "nfl":
            url = "http://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
        elif sport == "basketball" and league == "nba":
            url = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
        else:
            raise ValueError("Invalid sport or league provided.")

        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get('sports', [])[0]['leagues'][0]['teams']
        else:
            print(f"Failed to fetch teams: {response.status_code}")
            return None

    @staticmethod
    def get_player_stats(sport, player_id):
        if sport == "football":
            url = f"http://site.api.espn.com/apis/site/v2/sports/football/nfl/players/{player_id}/statistics"
        elif sport == "basketball":
            url = f"http://site.api.espn.com/apis/site/v2/sports/basketball/nba/players/{player_id}/statistics"
        else:
            raise ValueError("Invalid sport provided.")

        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch player stats: {response.status_code}")
            return None


'***********************************************************************'
'''Load and Process Data'''
def load_json_data(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def process_teams_data(teams_data):
    """Process teams data from the JSON structure."""
    teams_list = []
    for team_info in teams_data:
        team = team_info['team']
        teams_list.append({
            'id': team['id'],
            'abbreviation': team['abbreviation'],
            'displayName': team['displayName'],
            'shortDisplayName': team['shortDisplayName'],
            'location': team['location'],
            'color': team.get('color'),
            'alternateColor': team.get('alternateColor'),
            'logos': team['logos'][0]['href'] if team.get('logos') else None,
            'clubhouseLink': team['links'][0]['href'] if team.get('links') else None
        })
    return teams_list


def process_scores_data(scores_data):
    """Process scores data from the JSON structure."""
    scores_list = []
    for event in scores_data:
        competition = event.get('competitions', [])[0] if event.get('competitions') else None
        if competition:
            for competitor in competition.get('competitors', []):
                scores_list.append({
                    'game_id': event['id'],
                    'date': event['date'],
                    'team_id': competitor['team']['id'],
                    'homeAway': competitor['homeAway'],
                    'score': competitor.get('score'),
                    # Adding placeholder values for target fields
                    'fantasy_points': 0,
                    'touchdowns': 0,
                    'yards': 0,
                    'receptions': 0,
                    'fumbles': 0,
                    'interceptions': 0,
                    'field_goal': 0,
                })
    return scores_list


def merge_data(teams_list, scores_list):
    """Merge the processed teams and scores data."""
    merged_data = []
    for score in scores_list:
        team = next((team for team in teams_list if team['id'] == score['team_id']), None)
        if team:
            merged_record = {**score, **team}
            merged_data.append(merged_record)
    
    if merged_data:
        print("Sample Merged Data:", merged_data[0])  # Print first record as a sample

    if not merged_data:
        raise ValueError("No valid data to process after merging.")
    return merged_data


def extract_features_and_targets(merged_data):
    """Extract features and targets from the merged data."""
    # Convert merged_data into a DataFrame for easier processing
    df = pd.DataFrame(merged_data)
    
    # Print the columns to check what is available
    print("DataFrame Columns:", df.columns)
    
    # Define feature columns and target columns
    feature_columns = ['team_id', 'abbreviation', 'location', 'homeAway', 'score', 'game_id', 'date']
    
    # Make sure these target columns exist in your DataFrame
    target_columns = ['fantasy_points', 'touchdowns', 'yards', 'receptions', 'fumbles', 'interceptions', 'field_goal']
    
    # Check if the target columns exist in df
    missing_targets = [col for col in target_columns if col not in df.columns]
    if missing_targets:
        raise KeyError(f"Missing target columns: {missing_targets}")
    
    # Extract features and targets
    X = df[feature_columns].copy()
    y = df[target_columns].copy()
    
    # Handle date feature: Convert to datetime and extract useful features
    X['date'] = pd.to_datetime(X['date'])
    X['year'] = X['date'].dt.year
    X['month'] = X['date'].dt.month
    X['day'] = X['date'].dt.day
    X['day_of_week'] = X['date'].dt.dayofweek
    X.drop('date', axis=1, inplace=True)
    
    # Identify categorical and numerical columns
    categorical_features = ['team_id', 'abbreviation', 'location', 'homeAway', 'game_id']
    numerical_features = ['score', 'year', 'month', 'day', 'day_of_week']
    
    # Define the ColumnTransformer with appropriate encoders
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ]
    )
    
    # Create a preprocessing pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    
    # Fit and transform the features
    X_processed = pipeline.fit_transform(X)
    
    # Convert targets to NumPy array
    y_processed = y.values  # Shape: (n_samples, n_targets)
    
    return X_processed, y_processed


def prepare_training_data(teams_file_path, scores_file_path):
    # Load the data
    teams_data = load_json_data(teams_file_path)
    scores_data = load_json_data(scores_file_path)  

    print(f"Teams Data: {len(teams_data)} records loaded")
    print(f"Scores Data: {len(scores_data)} records loaded")

    if not teams_data or not scores_data:
        raise ValueError("No data loaded from JSON files")

    # Process the data
    teams_list = process_teams_data(teams_data)
    scores_list = process_scores_data(scores_data)

    print(f"Processed Teams: {len(teams_list)} records")
    print(f"Processed Scores: {len(scores_list)} records")

    if not teams_list or not scores_list:
        raise ValueError("No data available after processing")

    # Merge the data
    merged_data = merge_data(teams_list, scores_list)
    print(f"Merged Data: {len(merged_data)} records")

    # Extract features and targets
    X, y = extract_features_and_targets(merged_data)
    print(f"Extracted Features: {X.shape[0]} samples")
    print(f"Extracted Targets: {y.shape[0]} samples")

    if X.size == 0 or y.size == 0:
        raise ValueError("No data available after extracting features and targets")

    return X, y


# Call the function to prepare the data
X, y = prepare_training_data(teams_file_path, scores_file_path)
# Train the model using the actual data
model = xgboost_model(X, y, problem_type="regression")

print(f"Features (X): {X.shape[0]} samples")
print(f"Targets (y): {y.shape[0]} samples")
print(f"y shape: {y.shape}")



'***********************************************************************'
'''Models'''
# Define file paths for models
XGB_MODEL_PATH = 'backend/models/xgb_model.json'
TRANSFER_MODEL_PATH = 'backend/models/transfer_model.h5'

# Load or train models
if not os.path.exists(XGB_MODEL_PATH):
    print("XGBoost model not found, training the model...")
    X, y = prepare_training_data(teams_file_path, scores_file_path)
    xgb_model = xgboost_model(X, y)
    #xgb_model.save_model(XGB_MODEL_PATH)
else:
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(XGB_MODEL_PATH)

if not os.path.exists(TRANSFER_MODEL_PATH):
    print("Transfer learning model not found, training the model...")
    X_train, y_train = prepare_training_data(teams_file_path, scores_file_path)
    transfer_model = train_transfer_model(X_train, y_train)
    transfer_model.save(TRANSFER_MODEL_PATH)
else:
    transfer_model = tf.keras.models.load_model(TRANSFER_MODEL_PATH)

'***********************************************************************'
'''Routes'''
# Define routes
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    df, _, _ = preprocess_data(df)
    
    xgb_proba = xgb_model.predict(df)
    transfer_proba = transfer_model.predict(df.values)

    ensemble_proba = (xgb_proba + transfer_proba) / 2

    # Convert the predictions into a dictionary
    targets = ['fantasy_points', 'touchdowns', 'yards', 'receptions', 'fumbles', 'interceptions', 'field_goal']
    predictions = dict(zip(targets, ensemble_proba[0]))
    
    return jsonify(predictions)

@app.route('/api/predict_weekly', methods=['POST'])
def predict_weekly():
    data = request.json  # Data should include week, player info, etc.
    df = pd.DataFrame([data])
    df, _, _ = preprocess_data(df)
    
    target_columns = ['fantasy_points', 'touchdowns', 'yards', 'receptions', 'fumbles', 'interceptions', 'field_goal']
    predictions = xgb_model.predict(df)
    
    prediction_dict = {target: float(pred) for target, pred in zip(target_columns, predictions)}
    
    return jsonify(prediction_dict)


# Function to prepare training data based on the current week
def get_current_week(season_start_date):
    season_start = datetime.strptime(season_start_date, '%Y-%m-%d')
    current_date = datetime.now()
    days_diff = (current_date - season_start).days
    current_week = days_diff // 7 + 1
    return current_week

# Example start dates
nfl_season_start_date = '2024-09-05'
nba_season_start_date = '2024-10-24'

# Determine current week
current_week = get_current_week(nfl_season_start_date)  # Use NBA start date for NBA data

X, y = prepare_training_data(week=current_week)


'***********************************************************************'
'''Sleeper API Endpoints'''
@app.route('/api/sleeper/user_data', methods=['POST'])
def get_sleeper_user_data():
    data = request.json
    username_or_id = data.get('username_or_id')
    if not username_or_id:
        return jsonify({"error": "Username or ID required"}), 400
    
    user_data = SleeperAPI.get_user(username_or_id)
    
    if user_data:
        df = pd.DataFrame([user_data])
        df, label_encoders, scaler = preprocess_data(df)
        json_response = df.to_dict(orient='records')
        
        return jsonify(json_response)
        
    return jsonify({"error": "User not found"}), 404


@app.route('/api/sleeper/player_stats/<player_id>', methods=['GET'])
def get_sleeper_player_stats(player_id):
    player_data = SleeperAPI.get_player_stats(player_id)
    
    if player_data:
        df = pd.DataFrame([player_data])
        df, label_encoders, scaler = preprocess_data(df)
        df = apply_feature_engineering(df, feature_type='player')
        return jsonify(df.to_dict(orient='records'))
    
    return jsonify({"error": "Player not found"}), 404

@app.route('/api/sleeper/leagues/<user_id>/<sport>/<season>', methods=['GET'])
def get_sleeper_leagues(user_id, sport, season):
    leagues = SleeperAPI.get_leagues(user_id, sport, season)
    
    if leagues:
        df = pd.DataFrame(leagues)
        df, label_encoders, scaler = preprocess_data(df)
        df = apply_feature_engineering(df, feature_type='team')
        return jsonify(df.to_dict(orient='records'))
    
    return jsonify({"error": "Leagues not found"}), 404

@app.route('/api/sleeper/league/<league_id>', methods=['GET'])
def get_sleeper_league(league_id):
    league_data = SleeperAPI.get_league(league_id)
    
    if league_data:
        df = pd.DataFrame([league_data])
        df, label_encoders, scaler = preprocess_data(df)
        df = apply_feature_engineering(df, feature_type='team')
        return jsonify(df.to_dict(orient='records'))
    
    return jsonify({"error": "League not found"}), 404

@app.route('/api/sleeper/user/<username_or_id>', methods=['GET'])
def get_sleeper_user(username_or_id):
    user_data = SleeperAPI.get_user(username_or_id)
    
    if user_data:
        df = pd.DataFrame([user_data])
        df, label_encoders, scaler = preprocess_data(df)
        json_response = df.to_dict(orient='records')
        
        return jsonify(json_response)
        
    return jsonify({"error": "User not found"}), 404

@app.route('/api/sleeper/league/<league_id>/rosters', methods=['GET'])
def get_sleeper_league_rosters(league_id):
    rosters = SleeperAPI.get_rosters(league_id)
    
    if rosters:
        df = pd.DataFrame(rosters)
        df, label_encoders, scaler = preprocess_data(df)
        df = apply_feature_engineering(df, feature_type='team')
        return jsonify(df.to_dict(orient='records'))
    
    return jsonify({"error": "Rosters not found"}), 404

@app.route('/api/sleeper/league/<league_id>/users', methods=['GET'])
def get_sleeper_league_users(league_id):
    users = SleeperAPI.get_users(league_id)
    
    if users:
        df = pd.DataFrame(users)
        df, label_encoders, scaler = preprocess_data(df)
        df = apply_feature_engineering(df, feature_type='player')
        return jsonify(df.to_dict(orient='records'))
    
    return jsonify({"error": "Users not found"}), 404

@app.route('/api/sleeper/league/<league_id>/matchups/<week>', methods=['GET'])
def get_sleeper_league_matchups(league_id, week):
    matchups = SleeperAPI.get_matchups(league_id, week)
    
    if matchups:
        df = pd.DataFrame(matchups)
        df, label_encoders, scaler = preprocess_data(df)
        df = apply_feature_engineering(df, feature_type='team')
        return jsonify(df.to_dict(orient='records'))
    
    return jsonify({"error": "Matchups not found"}), 404

@app.route('/api/sleeper/draft/<draft_id>', methods=['GET'])
def get_sleeper_draft(draft_id):
    draft_data = SleeperAPI.get_draft(draft_id)
    
    if draft_data:
        df = pd.DataFrame([draft_data])
        df, label_encoders, scaler = preprocess_data(df)
        df = apply_feature_engineering(df, feature_type='team')
        return jsonify(df.to_dict(orient='records'))
    
    return jsonify({"error": "Draft not found"}), 404

@app.route('/api/sleeper/draft/<draft_id>/picks', methods=['GET'])
def get_sleeper_draft_picks(draft_id):
    picks = SleeperAPI.get_draft_picks(draft_id)
    
    if picks:
        df = pd.DataFrame(picks)
        df, label_encoders, scaler = preprocess_data(df)
        df = apply_feature_engineering(df, feature_type='team')
        return jsonify(df.to_dict(orient='records'))
    
    return jsonify({"error": "Picks not found"}), 404

@app.route('/api/sleeper/players/<sport>', methods=['GET'])
def get_sleeper_players(sport):
    players = SleeperAPI.get_players(sport)
    
    if players:
        df = pd.DataFrame(players)
        df, label_encoders, scaler = preprocess_data(df)
        df = apply_feature_engineering(df, feature_type='player')
        return jsonify(df.to_dict(orient='records'))
    
    return jsonify({"error": "Players not found"}), 404


'***********************************************************************'
'''ESPN API Endpoints'''
@app.route('/api/espn/college_football/news', methods=['GET'])
def get_college_football_news():
    url = 'http://site.api.espn.com/apis/site/v2/sports/football/college-football/news'
    response = requests.get(url)
    
    if response.status_code == 200:
        news = response.json()
        return jsonify(news)
    return jsonify({"error": "College Football news not found"}), 404

@app.route('/api/espn/college_football/scoreboard', methods=['GET'])
def get_college_football_scores():
    url = 'http://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard'
    params = {
        'calendar': request.args.get('calendar', 'blacklist'),
        'dates': request.args.get('dates')
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        scores = response.json()
        return jsonify(scores)
    return jsonify({"error": "College Football scores not found"}), 404

@app.route('/api/espn/college_football/game/<gameId>', methods=['GET'])
def get_college_football_game_info(gameId):
    url = f'http://site.api.espn.com/apis/site/v2/sports/football/college-football/summary?event={gameId}'
    response = requests.get(url)
    
    if response.status_code == 200:
        game_info = response.json()
        return jsonify(game_info)
    return jsonify({"error": f"Game {gameId} not found"}), 404

@app.route('/api/espn/college_football/teams/<team>', methods=['GET'])
def get_college_football_team_info(team):
    url = f'http://site.api.espn.com/apis/site/v2/sports/football/college-football/teams/{team}'
    response = requests.get(url)
    
    if response.status_code == 200:
        team_info = response.json()
        return jsonify(team_info)
    return jsonify({"error": f"Team {team} not found"}), 404

@app.route('/api/espn/college_football/rankings', methods=['GET'])
def get_college_football_rankings():
    url = 'http://site.api.espn.com/apis/site/v2/sports/football/college-football/rankings'
    response = requests.get(url)
    
    if response.status_code == 200:
        rankings = response.json()
        return jsonify(rankings)
    return jsonify({"error": "Rankings not found"}), 404

# NFL Endpoints
@app.route('/api/espn/nfl/scoreboard', methods=['GET'])
def get_nfl_scores():
    url = 'http://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard'
    response = requests.get(url)
    
    if response.status_code == 200:
        scores = response.json()
        return jsonify(scores)
    return jsonify({"error": "NFL scores not found"}), 404

@app.route('/api/espn/nfl/news', methods=['GET'])
def get_nfl_news():
    url = 'http://site.api.espn.com/apis/site/v2/sports/football/nfl/news'
    response = requests.get(url)
    
    if response.status_code == 200:
        news = response.json()
        return jsonify(news)
    return jsonify({"error": "NFL news not found"}), 404

@app.route('/api/espn/nfl/teams', methods=['GET'])
def get_nfl_teams():
    url = 'http://site.api.espn.com/apis/site/v2/sports/football/nfl/teams'
    response = requests.get(url)
    
    if response.status_code == 200:
        teams = response.json()
        return jsonify(teams)
    return jsonify({"error": "NFL teams not found"}), 404

@app.route('/api/espn/nfl/teams/<team>', methods=['GET'])
def get_nfl_team_info(team):
    url = f'http://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team}'
    response = requests.get(url)
    
    if response.status_code == 200:
        team_info = response.json()
        return jsonify(team_info)
    return jsonify({"error": f"Team {team} not found"}), 404

# NBA Endpoints
@app.route('/api/espn/nba/scoreboard', methods=['GET'])
def get_nba_scores():
    url = 'http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'
    response = requests.get(url)
    
    if response.status_code == 200:
        scores = response.json()
        return jsonify(scores)
    return jsonify({"error": "NBA scores not found"}), 404

@app.route('/api/espn/nba/news', methods=['GET'])
def get_nba_news():
    url = 'http://site.api.espn.com/apis/site/v2/sports/basketball/nba/news'
    response = requests.get(url)
    
    if response.status_code == 200:
        news = response.json()
        return jsonify(news)
    return jsonify({"error": "NBA news not found"}), 404

@app.route('/api/espn/nba/teams', methods=['GET'])
def get_nba_teams():
    url = 'http://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams'
    response = requests.get(url)
    
    if response.status_code == 200:
        teams = response.json()
        return jsonify(teams)
    return jsonify({"error": "NBA teams not found"}), 404

@app.route('/api/espn/nba/teams/<team>', methods=['GET'])
def get_nba_team_info(team):
    url = f'http://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team}'
    response = requests.get(url)
    
    if response.status_code == 200:
        team_info = response.json()
        return jsonify(team_info)
    return jsonify({"error": f"NBA team {team} not found"}), 404


# College Football Endpoints

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
