import os
import requests
import pandas as pd
import json
import numpy as np 
import xgboost as xgb
from dotenv import load_dotenv
from flask import Flask, jsonify, request, Response
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
from backend.models.train_models import train_ensemble_model, train_xgboost_model
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from backend.utils.formatter import pretty_print_json
from backend.models.train_models import train_ensemble_model, train_xgboost_model, save_models, load_models
from backend.api.player_endpoints import PlayerEndpoint

app = Flask(__name__)
app.run(debug=True)

# Initialize Sleeper API client
sleeper_client = SleeperAPI()

# Initialize PlayerEndpoint
player_endpoint = PlayerEndpoint(client=sleeper_client)

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
    """Merge the processed teams and scores data with validation."""
    merged_data = []
    
    print(f"Teams List Size: {len(teams_list)}")
    print(f"Scores List Size: {len(scores_list)}")
    
    # Create a set of valid team ids for quick lookup
    valid_team_ids = {team['id'] for team in teams_list}
    
    for score in scores_list:
        if score['team_id'] in valid_team_ids:
            team = next((team for team in teams_list if team['id'] == score['team_id']), None)
            if team:
                merged_record = {**score, **team}
                merged_data.append(merged_record)
        else:
            print(f"No matching team found for team_id: {score['team_id']}")
    
    if not merged_data:
        raise ValueError("No valid data to process after merging.")
    
    print(f"Merged Data Size: {len(merged_data)}")
    if len(merged_data) > 0:
        print(f"Sample Merged Data: {merged_data[0]}")
    
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


def prepare_training_data(teams_file_path, scores_file_path, week=None):
    teams_data = load_json_data(teams_file_path)
    scores_data = load_json_data(scores_file_path)  

    teams_list = process_teams_data(teams_data)
    scores_list = process_scores_data(scores_data)

    if week is not None:
        scores_list = [score for score in scores_list if score.get('week', {}).get('number') == week]
        print(f"Filtered Scores for Week {week}: {len(scores_list)} records")
    
    # If filtering resulted in no data, skip the filtering for now
    if not scores_list:
        print("No scores found for the selected week. Using all scores.")
        scores_list = process_scores_data(scores_data)  # Reprocess without week filtering

    merged_data = merge_data(teams_list, scores_list)
    X, y = extract_features_and_targets(merged_data)

    return X, y


# Call the function to prepare the data
X, y = prepare_training_data(teams_file_path, scores_file_path)
# Train the model using the actual data
xgb_model = train_xgboost_model(X, y)

# Train and evaluate the ensemble model
ensemble_accuracy = train_ensemble_model(X, y, xgb_model)

print(f"Final Ensemble Model Accuracy: {ensemble_accuracy:.2f}")
print(f"Features (X): {X.shape[0]} samples")
print(f"Targets (y): {y.shape[0]} samples")
print(f"y shape: {y.shape}")

'***********************************************************************'
'''Models'''
# Define file paths for models
import os
import joblib

XGB_MODEL_PATH = 'backend/models/xgb_model_{}.joblib'  # Use joblib for saving individual models

# Load or train models
if not os.path.exists(XGB_MODEL_PATH.format(0)):  # Check for the first model as a proxy
    print("XGBoost models not found, training the models...")
    X, y = prepare_training_data(teams_file_path, scores_file_path)
    xgb_model = train_xgboost_model(X, y)
    
    # Save each individual model in the MultiOutputClassifier
    for idx, estimator in enumerate(xgb_model.estimators_):
        joblib.dump(estimator, XGB_MODEL_PATH.format(idx))
        print(f"Model for target {idx} saved as {XGB_MODEL_PATH.format(idx)}")
else:
    from sklearn.multioutput import MultiOutputClassifier
    from xgboost import XGBClassifier
    
    # Load each individual model and reassemble the MultiOutputClassifier
    estimators = []
    for idx in range(y.shape[1]):
        estimator = joblib.load(XGB_MODEL_PATH.format(idx))
        estimators.append(estimator)
    
    xgb_model = MultiOutputClassifier(XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3))
    xgb_model.estimators_ = estimators
    print("XGBoost models loaded successfully.")

'***********************************************************************'
'''Routes'''
# Define routes
@app.route('/')
def home():
    return "Welcome to the Fantasy Sports Optimizer!"

@app.route('/favicon.ico')
def favicon():
    return "", 204  # You can replace this with a real favicon if you have one

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    df, _, _ = preprocess_data(df)

    # Get predictions from the XGBoost model
    xgb_proba = xgb_model.predict(df)

    # Convert the predictions into a dictionary
    targets = ['fantasy_points', 'touchdowns', 'yards', 'receptions', 'fumbles', 'interceptions', 'field_goal']
    predictions = dict(zip(targets, xgb_proba[0]))  # Assuming the first row of prediction

    return jsonify(predictions)

    
    return jsonify(predictions)

@app.route('/api/predict_weekly', methods=['POST'])
def predict_weekly():
    data = request.json  # Data should include week, player info, etc.
    df = pd.DataFrame([data])
    df, _, _ = preprocess_data(df)

    # Get predictions from the XGBoost model
    predictions = xgb_model.predict(df)

    # Convert the predictions into a dictionary
    target_columns = ['fantasy_points', 'touchdowns', 'yards', 'receptions', 'fumbles', 'interceptions', 'field_goal']
    prediction_dict = {target: float(pred) for target, pred in zip(target_columns, predictions[0])}  # Assuming the first row of prediction

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

# Call the function with the 'week' parameter
X, y = prepare_training_data(teams_file_path, scores_file_path, week=current_week)



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
        pretty_json = json.dumps(user_data, indent=4)
        return Response(pretty_json, mimetype='application/json')
        
    return jsonify({"error": "User not found"}), 404


@app.route('/api/sleeper/player_stats/<player_id>', methods=['GET'])
def get_sleeper_player_stats(player_id):
    player_data = SleeperAPI.get_player_stats(player_id)
    
    if player_data:
        pretty_json = json.dumps(player_data, indent=4)
        return Response(pretty_json, mimetype='application/json')
    
    return jsonify({"error": "Player not found"}), 404


@app.route('/api/sleeper/leagues/<user_id>/<sport>/<season>', methods=['GET'])
def get_sleeper_leagues(user_id, sport, season):
    leagues = SleeperAPI.get_leagues(user_id, sport, season)
    
    if leagues:
        pretty_json = json.dumps(leagues, indent=4)
        return Response(pretty_json, mimetype='application/json')
    
    return jsonify({"error": "Leagues not found"}), 404


@app.route('/api/sleeper/league/<league_id>', methods=['GET'])
def get_sleeper_league(league_id):
    league_data = SleeperAPI.get_league(league_id)
    
    if league_data:
        pretty_json = json.dumps(league_data, indent=4)
        return Response(pretty_json, mimetype='application/json')
    
    return jsonify({"error": "League not found"}), 404


@app.route('/api/sleeper/league/<league_id>/rosters', methods=['GET'])
def get_sleeper_league_rosters(league_id):
    rosters = SleeperAPI.get_rosters(league_id)
    
    if rosters:
        pretty_json = json.dumps(rosters, indent=4)
        return Response(pretty_json, mimetype='application/json')
    
    return jsonify({"error": "Rosters not found"}), 404


@app.route('/api/sleeper/league/<league_id>/users', methods=['GET'])
def get_sleeper_league_users(league_id):
    users = SleeperAPI.get_users(league_id)
    
    if users:
        pretty_json = json.dumps(users, indent=4)
        return Response(pretty_json, mimetype='application/json')
    
    return jsonify({"error": "Users not found"}), 404


@app.route('/api/sleeper/league/<league_id>/matchups/<week>', methods=['GET'])
def get_sleeper_league_matchups(league_id, week):
    matchups = SleeperAPI.get_matchups(league_id, week)
    
    if matchups:
        pretty_json = json.dumps(matchups, indent=4)
        return Response(pretty_json, mimetype='application/json')
    
    return jsonify({"error": "Matchups not found"}), 404


@app.route('/api/sleeper/draft/<draft_id>', methods=['GET'])
def get_sleeper_draft(draft_id):
    draft_data = SleeperAPI.get_draft(draft_id)
    
    if draft_data:
        pretty_json = json.dumps(draft_data, indent=4)
        return Response(pretty_json, mimetype='application/json')
    
    return jsonify({"error": "Draft not found"}), 404


@app.route('/api/sleeper/draft/<draft_id>/picks', methods=['GET'])
def get_sleeper_draft_picks(draft_id):
    picks = SleeperAPI.get_draft_picks(draft_id)
    
    if picks:
        pretty_json = json.dumps(picks, indent=4)
        return Response(pretty_json, mimetype='application/json')
    
    return jsonify({"error": "Picks not found"}), 404


@app.route('/api/sleeper/players/<sport>', methods=['GET'])
def get_sleeper_players(sport):
    players = SleeperAPI.get_players(sport)
    
    if players:
        pretty_json = json.dumps(players, indent=4)
        return Response(pretty_json, mimetype='application/json')
    
    return jsonify({"error": "Players not found"}), 404

@app.route('/api/sleeper/players', methods=['GET'])
def get_all_players():
    sport = request.args.get('sport', 'nfl')
    players = player_endpoint.get_all_players(sport)
    
    # Convert PlayerModel to dictionary and pretty print
    response = [player.to_dict() for player in players]
    pretty_response = json.dumps(response, indent=4)
    
    return app.response_class(pretty_response, content_type='application/json')

@app.route('/api/sleeper/trending', methods=['GET'])
def get_trending_players():
    sport = request.args.get('sport', 'nfl')
    trend_type = request.args.get('trend_type', 'add')
    lookback_hours = int(request.args.get('lookback_hours', 24))
    limit = int(request.args.get('limit', 25))
    
    players = player_endpoint.get_trending_players(sport, trend_type, lookback_hours, limit)
    
    # Convert PlayerModel to dictionary and pretty print
    response = [player.to_dict() for player in players]
    pretty_response = json.dumps(response, indent=4)
    
    return app.response_class(pretty_response, content_type='application/json')

@app.route('/api/sleeper/players/team/<team_abbr>', methods=['GET'])
def get_players_by_team(team_abbr):
    players = player_endpoint.get_players_by_team(team_abbr)
    
    # Convert PlayerModel to dictionary and pretty print
    response = [player.to_dict() for player in players]
    pretty_response = json.dumps(response, indent=4)
    
    return app.response_class(pretty_response, content_type='application/json')













'''ESPN API Endpoints'''
'''***********************************************************************'''
@app.route('/api/espn/college_football/news', methods=['GET'])
def get_college_football_news():
    url = 'http://site.api.espn.com/apis/site/v2/sports/football/college-football/news'
    response = requests.get(url)
    
    if response.status_code == 200:
        news = response.json()
        pretty_json = json.dumps(news, indent=4)
        return Response(pretty_json, mimetype='application/json')
    
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
        pretty_json = json.dumps(scores, indent=4)
        return Response(pretty_json, mimetype='application/json')
    
    return jsonify({"error": "College Football scores not found"}), 404


@app.route('/api/espn/college_football/game/<gameId>', methods=['GET'])
def get_college_football_game_info(gameId):
    url = f'http://site.api.espn.com/apis/site/v2/sports/football/college-football/summary?event={gameId}'
    response = requests.get(url)
    
    if response.status_code == 200:
        game_info = response.json()
        pretty_json = json.dumps(game_info, indent=4)
        return Response(pretty_json, mimetype='application/json')
    
    return jsonify({"error": f"Game {gameId} not found"}), 404


@app.route('/api/espn/college_football/teams/<team>', methods=['GET'])
def get_college_football_team_info(team):
    url = f'http://site.api.espn.com/apis/site/v2/sports/football/college-football/teams/{team}'
    response = requests.get(url)
    
    if response.status_code == 200:
        team_info = response.json()
        pretty_json = json.dumps(team_info, indent=4)
        return Response(pretty_json, mimetype='application/json')
    
    return jsonify({"error": f"Team {team} not found"}), 404


@app.route('/api/espn/college_football/rankings', methods=['GET'])
def get_college_football_rankings():
    url = 'http://site.api.espn.com/apis/site/v2/sports/football/college-football/rankings'
    response = requests.get(url)
    
    if response.status_code == 200:
        rankings = response.json()
        pretty_json = json.dumps(rankings, indent=4)
        return Response(pretty_json, mimetype='application/json')
    
    return jsonify({"error": "Rankings not found"}), 404


@app.route('/api/espn/nfl/scoreboard', methods=['GET'])
def get_nfl_scores():
    nfl_scores = ESPNAPI.get_scores("football", "nfl")
    
    if nfl_scores:
        pretty_json = json.dumps(nfl_scores, indent=4)
        return Response(pretty_json, mimetype='application/json')
    
    return jsonify({"error": "NFL scores not found"}), 404


@app.route('/api/espn/nfl/news', methods=['GET'])
def get_nfl_news():
    url = 'http://site.api.espn.com/apis/site/v2/sports/football/nfl/news'
    response = requests.get(url)
    
    if response.status_code == 200:
        news = response.json()
        pretty_json = json.dumps(news, indent=4)
        return Response(pretty_json, mimetype='application/json')
    
    return jsonify({"error": "NFL news not found"}), 404


@app.route('/api/espn/nfl/teams', methods=['GET'])
def get_nfl_teams():
    url = 'http://site.api.espn.com/apis/site/v2/sports/football/nfl/teams'
    response = requests.get(url)
    
    if response.status_code == 200:
        teams = response.json()
        pretty_json = json.dumps(teams, indent=4)
        return Response(pretty_json, mimetype='application/json')
    
    return jsonify({"error": "NFL teams not found"}), 404


@app.route('/api/espn/nfl/teams/<team>', methods=['GET'])
def get_nfl_team_info(team):
    url = f'http://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team}'
    response = requests.get(url)
    
    if response.status_code == 200:
        team_info = response.json()
        pretty_json = json.dumps(team_info, indent=4)
        return Response(pretty_json, mimetype='application/json')
    
    return jsonify({"error": f"Team {team} not found"}), 404


@app.route('/api/espn/nba/scoreboard', methods=['GET'])
def get_nba_scores():
    url = 'http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'
    response = requests.get(url)
    
    if response.status_code == 200:
        scores = response.json()
        pretty_json = json.dumps(scores, indent=4)
        return Response(pretty_json, mimetype='application/json')
    
    return jsonify({"error": "NBA scores not found"}), 404


@app.route('/api/espn/nba/news', methods=['GET'])
def get_nba_news():
    url = 'http://site.api.espn.com/apis/site/v2/sports/basketball/nba/news'
    response = requests.get(url)
    
    if response.status_code == 200:
        news = response.json()
        pretty_json = json.dumps(news, indent=4)
        return Response(pretty_json, mimetype='application/json')
    
    return jsonify({"error": "NBA news not found"}), 404


@app.route('/api/espn/nba/teams', methods=['GET'])
def get_nba_teams():
    url = 'http://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams'
    response = requests.get(url)
    
    if response.status_code == 200:
        teams = response.json()
        pretty_json = json.dumps(teams, indent=4)
        return Response(pretty_json, mimetype='application/json')
    
    return jsonify({"error": "NBA teams not found"}), 404


@app.route('/api/espn/nba/teams/<team>', methods=['GET'])
def get_nba_team_info(team):
    url = f'http://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team}'
    response = requests.get(url)
    
    if response.status_code == 200:
        team_info = response.json()
        pretty_json = json.dumps(team_info, indent=4)
        return Response(pretty_json, mimetype='application/json')
    
    return jsonify({"error": f"NBA team {team} not found"}), 404


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
