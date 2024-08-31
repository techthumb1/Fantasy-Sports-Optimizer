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
from backend.utils.model_training import train_xgboost_model, train_transfer_model
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
'''Load data'''
# Function to load teams data from JSON
#def load_json_data(file_path):
#    with open(file_path, 'r') as f:
#        data = json.load(f)
#    return data

def process_teams_data(teams_data):
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
    teams_df = pd.DataFrame(teams_list)
    return teams_df

def process_scores_data(scores_data):
    scores_list = []
    for event in scores_data:
        competition = event['competitions'][0] if event.get('competitions') else None
        if competition:
            for competitor in competition['competitors']:
                scores_list.append({
                    'game_id': event['id'],
                    'date': event['date'],
                    'team_id': competitor['team']['id'],
                    'homeAway': competitor['homeAway'],
                    'score': competitor.get('score'),
                    'winner': competitor.get('winner')
                })
    scores_df = pd.DataFrame(scores_list)
    return scores_df


# Function to merge the loaded data
def merge_data(teams_df, scores_df):
    # Merge on the correct key, adjust the key names as necessary based on your data structure
    merged_df = pd.merge(scores_df, teams_df, left_on='competitions[0].competitors[0].team.id', right_on='id', how='left')
    
    if merged_df.empty:
        raise ValueError("No valid data to process after merging.")
    
    return merged_df

'***********************************************************************'  
'''Preprocess data'''
# Prepare training data based on the current week        
def get_current_week(season_start_date):
    season_start = datetime.strptime(season_start_date, '%Y-%m-%d')
    current_date = datetime.now()
    days_diff = (current_date - season_start).days
    current_week = days_diff // 7 + 1
    return current_week

# Fetch data from APIs and prepare it for training
#def fetch_data_from_apis(sport="nfl"):
#    # Determine the season start date
#    if sport == "nfl":
#        season_start_date = '2024-09-05'  # Example NFL season start date
#    elif sport == "nba":
#        season_start_date = '2024-10-24'  # Example NBA season start date
#    else:
#        raise ValueError("Unsupported sport")
#
#    # Calculate the current week
#    current_week = get_current_week(season_start_date)
#    
#    # Fetch trending players from Sleeper API
#    trending_players = SleeperAPI.get_trending_players(sport=sport, trend_type="add")
#    
#    if not trending_players:
#        raise ValueError("No data fetched from Sleeper API")
#
#    df = pd.DataFrame(trending_players)
#
#    # Fetch weekly player stats from ESPN API
#    weekly_stats = []
#    for player in trending_players:
#        player_id = player['player_id']
#        player_stats = ESPNAPI.get_player_stats(sport, player_id)
#        if player_stats:
#            weekly_stats.append(player_stats)
#    
#    if not weekly_stats:
#        raise ValueError("No weekly stats fetched from ESPN API")
#
#    stats_df = pd.DataFrame(weekly_stats)
#
#    # Merge the Sleeper data with ESPN data
#    df = df.merge(stats_df, on='player_id', how='left')
#
#    if df.empty:
#        raise ValueError("No valid data to process after merging.")
#
#    return df




def load_json_data(file_path):
    """Load JSON data from a file and convert it to a DataFrame."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    return data

#def prepare_training_data(teams_file_path, scores_file_path):
#    # Load the teams and scores data from JSON files
#    teams_data = load_json_data(teams_file_path)
#    scores_data = load_json_data(scores_file_path)
#
#    # Normalize the teams data
#    teams_df = pd.json_normalize(teams_data['sports'][0]['leagues'][0]['teams'], sep='_')
#
#    # Normalize the scores data to extract the team information
#    scores_df = pd.json_normalize(
#        scores_data,
#        record_path=['competitions', 'competitors'],
#        meta=['id', 'date', 'week.number'],
#        meta_prefix='match_',
#        record_prefix='team_',
#        sep='_'
#    )
#
#    # Print the columns to identify the correct keys
#    print("Teams Data Columns:", teams_df.columns.tolist())
#    print("Scores Data Columns:", scores_df.columns.tolist())
#
#    # Adjust the merge keys based on the actual column names
#    teams_key = 'team_id'  # Adjust based on the normalized structure
#    scores_key = 'team_id'  # Adjust based on the normalized structure
#
#    if teams_key not in teams_df.columns:
#        raise KeyError(f"Key '{teams_key}' not found in teams_df")
#    if scores_key not in scores_df.columns:
#        raise KeyError(f"Key '{scores_key}' not found in scores_df")
#
#    # Merge the DataFrames on the identified keys
#    df = pd.merge(scores_df, teams_df, left_on=scores_key, right_on=teams_key, how='left')
#
#    if df.empty:
#        raise ValueError("No valid data to process after merging teams and scores data.")
#
#    # Define feature columns (X) and target columns (y)
#    feature_columns = [col for col in df.columns if col not in ['fantasy_points', 'touchdowns', 'yards', 'receptions', 'fumbles', 'interceptions', 'field_goal']]
#    target_columns = ['fantasy_points', 'touchdowns', 'yards', 'receptions', 'fumbles', 'interceptions', 'field_goal']
#
#    # Check if all target columns are present
#    missing_columns = [col for col in target_columns if col not in df.columns]
#    if missing_columns:
#        # Handle missing columns by initializing them with zero
#        for col in missing_columns:
#            df[col] = 0
#
#    # Separate the features (X) and targets (y)
#    X = df[feature_columns].copy()  # Features
#
## Process the teams and scores data
#teams_file_path = os.path.join(os.path.dirname(__file__), 'teams_data.json')
#scores_file_path = os.path.join(os.path.dirname(__file__), 'scores_data.json')
#
## Call the function with the correct arguments
#X, y = prepare_training_data(teams_file_path, scores_file_path)

import pandas as pd

def prepare_training_data(teams_file_path, scores_file_path):
    # Load the JSON data from files
    teams_data = load_json_data(teams_file_path)
    scores_data = load_json_data(scores_file_path)

    # Extract and normalize the teams data
    teams_df = pd.json_normalize(
        teams_data,
        sep='_',
        record_path=['team'],
        meta=[
            ['team', 'id'],
            ['team', 'location'],
            ['team', 'displayName'],
            ['team', 'abbreviation'],
            ['team', 'shortDisplayName']
        ]
    )

    # Extract and normalize the scores data
    scores_df = pd.json_normalize(
        scores_data,
        sep='_',
        record_path=['competitions', 'competitors'],
        meta=[
            ['competitions', 'id'],
            ['competitions', 'date'],
            ['competitions', 'season', 'year'],
            ['competitions', 'season', 'type'],
            ['competitions', 'week', 'number']
        ]
    )

    # Merge the teams and scores data on the common team ID
    df = pd.merge(teams_df, scores_df, left_on='team_id', right_on='id', how='left')

    # Check if merged data is empty
    if df.empty:
        raise ValueError("No valid data to process after merging.")

    # List of target columns for multi-target prediction
    target_columns = ['fantasy_points', 'touchdowns', 'yards', 'receptions', 'fumbles', 'interceptions', 'field_goal']

    # Ensure all target columns are present in the DataFrame
    missing_columns = [col for col in target_columns if col not in df.columns]
    if missing_columns:
        # Handle missing columns by initializing them with zero or NaN
        for col in missing_columns:
            df[col] = 0  # Or np.nan if you prefer to handle it differently

    # Proceed with preprocessing
    df, label_encoders, scaler = preprocess_data(df)
    
    # Separate features (X) and targets (y)
    X = df.drop(columns=target_columns)
    y = df[target_columns]
    
    return X, y

## Process the teams and scores data
teams_file_path = os.path.join(os.path.dirname(__file__), 'teams_data.json')
scores_file_path = os.path.join(os.path.dirname(__file__), 'scores_data.json')

# Call the function with the correct arguments
X, y = prepare_training_data(teams_file_path, scores_file_path)


'***********************************************************************'  
'''Models'''
# Define file paths for models
XGB_MODEL_PATH = 'backend/models/xgb_model.json'
TRANSFER_MODEL_PATH = 'backend/models/transfer_model.h5'

# Load or train models
if not os.path.exists(XGB_MODEL_PATH):
    print("XGBoost model not found, training the model...")
    X, y = prepare_training_data()
    xgb_model = train_xgboost_model(X, y)
    xgb_model.save_model(XGB_MODEL_PATH)
else:
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(XGB_MODEL_PATH)

if not os.path.exists(TRANSFER_MODEL_PATH):
    print("Transfer learning model not found, training the model...")
    X_train, y_train = prepare_training_data()
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
        
        pretty_print_json(json_response)
        
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
        
        pretty_print_json(json_response)
        
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
#@app.route('/api/espn/college_football/news', methods=['GET'])
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
