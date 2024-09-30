import os
import requests
import pandas as pd
import json
import joblib
import logging
from dotenv import load_dotenv
from flask import Flask, jsonify, request, Response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from backend.api.sleeper import SleeperAPI
from backend.api.espn import ESPNAPI
from backend.utils.preprocessing import preprocess_data
from backend.models.train_models import train_ensemble_model, train_xgboost_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from backend.api.player_endpoints import PlayerEndpoint
from backend.utils.data_fetching import fetch_live_teams_data, fetch_live_scores_data
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import atexit
from backend.routes import sleeper_routes
from backend.utils.data_updates import update_nfl_data, update_nba_data, update_college_data
from backend.utils.cache import load_cache, save_cache
from backend.utils.data_fetching import fetch_live_nfl_teams_data, fetch_live_nfl_scores_data, fetch_live_nba_teams_data, fetch_live_nba_scores_data, fetch_live_cfb_teams_data, fetch_live_cfb_scores_data
from backend.utils.data_preprocessing import process_nfl_teams_data, process_nfl_scores_data, process_cfb_teams_data, process_cfb_scores_data, process_nba_teams_data, process_nba_scores_data



# Initialize the Flask app
app = Flask(__name__)

# Configuration for SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sports_optimizer.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Register Blueprints
app.register_blueprint(sleeper_routes)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Load environment variables
load_dotenv()

# Initialize the scheduler
scheduler = BackgroundScheduler()

# Initialize Sleeper API client
sleeper_client = SleeperAPI()
player_endpoint = PlayerEndpoint(client=sleeper_client)
espn_client = ESPNAPI()

# Data Loading and Processing Functions
def load_json_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Define absolute paths for data
#teams_file_path = os.path.join(os.path.dirname(__file__), 'teams_data.json')
#scores_file_path = os.path.join(os.path.dirname(__file__), 'scores_data.json')

def get_live_teams_and_scores_data():
    teams_data = fetch_live_teams_data()
    scores_data = fetch_live_scores_data()
    return teams_data, scores_data

def verify_and_print_data():
    try:
        teams_data, scores_data = get_live_teams_and_scores_data()
        teams_list = process_teams_data(teams_data)
        scores_list = process_scores_data(scores_data)
        merged_data = merge_data(teams_list, scores_list)

        # Convert to DataFrame
        df = pd.DataFrame(merged_data)

        # Print the DataFrame structure
        print("DataFrame Head:")
        print(df.head())
        print("\nDataFrame Columns:")
        print(df.columns.tolist())

    except Exception as e:
        print(f"Error while verifying data: {e}")

# Call the function to print the data
verify_and_print_data()
breakpoint()


def process_teams_data(teams_data):
    teams_list = []
    for team_info in teams_data.get('sports', [])[0].get('leagues', [])[0].get('teams', []):
        team = team_info.get('team', {})
        teams_list.append({
            'id': team.get('id'),
            'abbreviation': team.get('abbreviation'),
            'displayName': team.get('displayName'),
            'shortDisplayName': team.get('shortDisplayName'),
            'location': team.get('location'),
            'color': team.get('color'),
            'alternateColor': team.get('alternateColor'),
            'logos': team.get('logos', [{}])[0].get('href'),
            'clubhouseLink': next((link.get('href') for link in team.get('links', []) if 'clubhouse' in link.get('rel', [])), None)
        })
    return teams_list


def process_scores_data(scores_data):
    scores_list = []
    for event in scores_data.get('events', []):
        competition = event.get('competitions', [])[0]
        for competitor in competition.get('competitors', []):
            scores_list.append({
                'game_id': event.get('id'),
                'date': event.get('date'),
                'team_id': competitor['team']['id'],
                'homeAway': competitor.get('homeAway'),
                'score': competitor.get('score'),
                # Add more fields as needed
            })
    return scores_list

teams_data, scores_data = get_live_teams_and_scores_data()
teams_list = process_teams_data(teams_data)
scores_list = process_scores_data(scores_data)

def merge_data(teams_list, scores_list):
    valid_team_ids = {team['id'] for team in teams_list}
    merged_data = [
        {**score, **team}
        for score in scores_list if score['team_id'] in valid_team_ids
        for team in [next((team for team in teams_list if team['id'] == score['team_id']), None)] if team
    ]
    if not merged_data:
        raise ValueError("No valid data to process after merging.")
    return merged_data

merged_data = merge_data(teams_list, scores_list)

df = pd.DataFrame(merged_data)
print(df.head())
print("DataFrame Columns:", df.columns.tolist())


def extract_features_and_targets(merged_data):
    df = pd.DataFrame(merged_data)
    df.dropna(inplace=True)
    
    # Define initial feature and target columns
    feature_columns = ['team_id', 'abbreviation', 'location', 'homeAway', 'score', 'game_id', 'date']
    target_columns = ['fantasy_points', 'touchdowns', 'yards', 'receptions', 'fumbles', 'interceptions', 'field_goal']
    
    # Check which target columns are available
    available_target_columns = [col for col in target_columns if col in df.columns]
    missing_targets = [col for col in target_columns if col not in df.columns]
    
    if missing_targets:
        logger.warning(f"Missing target columns: {missing_targets}")
        if not available_target_columns:
            raise KeyError("No target columns are available in the data.")
    
    # Proceed with available target columns
    X = df[feature_columns].copy()
    y = df[available_target_columns].copy()
    
    # Process date column
    X['date'] = pd.to_datetime(X['date'])
    X['year'] = X['date'].dt.year
    X['month'] = X['date'].dt.month
    X['day'] = X['date'].dt.day
    X['day_of_week'] = X['date'].dt.dayofweek
    X.drop('date', axis=1, inplace=True)
    
    # Define categorical and numerical features
    categorical_features = ['team_id', 'abbreviation', 'location', 'homeAway', 'game_id']
    numerical_features = ['score', 'year', 'month', 'day', 'day_of_week']
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ]
    )
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    X_processed = pipeline.fit_transform(X)
    y_processed = y.values
    
    return X_processed, y_processed




def get_live_teams_and_scores_data():
    try:
        cached_data = load_cache()
        if cached_data:
            teams_data = cached_data['teams_data']
            scores_data = cached_data['scores_data']
        else:
            teams_data = fetch_live_teams_data()
            scores_data = fetch_live_scores_data()
            save_cache({'teams_data': teams_data, 'scores_data': scores_data})
        return teams_data, scores_data
    except Exception as e:
        logger.error(f"Error fetching live data: {e}")
        raise



def prepare_training_data(week=None):
    teams_data, scores_data = get_live_teams_and_scores_data()
    teams_list = process_teams_data(teams_data)
    scores_list = process_scores_data(scores_data)

    if week is not None:
        scores_list = [score for score in scores_list if score.get('week', {}).get('number') == week]

    if not scores_list:
        scores_list = process_scores_data(scores_data)

    merged_data = merge_data(teams_list, scores_list)
    return extract_features_and_targets(merged_data)

# Save the models
def retrain_models():
    X, y = prepare_training_data()
    xgb_model = train_xgboost_model(X, y)
    joblib.dump(xgb_model, 'backend/models/xgb_model_full.joblib')
    try:
        X, y = prepare_training_data()
    except Exception as e:
        logger.error(f"Error preparing training data: {e}")

# Prepare and Train Models
X, y = prepare_training_data()
xgb_model = train_xgboost_model(X, y)
ensemble_accuracy = train_ensemble_model(X, y, xgb_model)
print(f"Final Ensemble Model Accuracy: {ensemble_accuracy:.2f}")

# Model Saving and Loading
XGB_MODEL_PATH = 'backend/models/xgb_model_{}.joblib'

if not os.path.exists(XGB_MODEL_PATH.format(0)):
    print("XGBoost models not found, training the models...")
    xgb_model = train_xgboost_model(X, y)
    for idx, estimator in enumerate(xgb_model.estimators_):
        joblib.dump(estimator, XGB_MODEL_PATH.format(idx))
else:
    from sklearn.multioutput import MultiOutputClassifier
    from xgboost import XGBClassifier
    estimators = [joblib.load(XGB_MODEL_PATH.format(idx)) for idx in range(y.shape[1])]
    xgb_model = MultiOutputClassifier(XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3))
    xgb_model.estimators_ = estimators

'*************************************************************'
# Define scheduled jobs
def schedule_jobs():
    # Schedule college data updates every Friday at 5:00 AM
    scheduler.add_job(
        update_college_data,
        trigger=CronTrigger(day_of_week='fri', hour=5, minute=0),
        id='update_college_data',
        replace_existing=True
    )

    # Schedule NBA data updates twice weekly on Mondays and Thursdays at 6:00 AM
    scheduler.add_job(
        update_nba_data,
        trigger=CronTrigger(day_of_week='mon,thu', hour=6, minute=0),
        id='update_nba_data',
        replace_existing=True
    )

    # Schedule NFL data updates every Saturday at 5:00 AM
    scheduler.add_job(
        update_nfl_data,
        trigger=CronTrigger(day_of_week='sat', hour=5, minute=0),
        id='update_nfl_data',
        replace_existing=True
    )

    # Schedule retrain_models to run every Friday at 6:00 AM
    scheduler.add_job(
        retrain_models,
        trigger=CronTrigger(day_of_week='fri', hour=6, minute=0),
        id='retrain_models_friday',
        replace_existing=True
    )

    # Start the scheduler
    scheduler.start()

    # Shut down the scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())

# Ensure that the scheduler is initialized before scheduling jobs
# Initialize the scheduler (if not already done)
# scheduler = BackgroundScheduler()

# Define your update functions before scheduling them
def update_college_data():
    try:
        # Fetch live data
        teams_data = fetch_live_cfb_teams_data()
        scores_data = fetch_live_cfb_scores_data()

        # Process data
        teams_list = process_cfb_teams_data(teams_data)
        scores_list = process_cfb_scores_data(scores_data)

        # Merge data
        merged_data = merge_data(teams_list, scores_list)

        # Save data
        save_data(merged_data, 'cfb_data.json')

        logger.info("College Football data updated successfully.")
    except Exception as e:
        logger.error(f"Error updating College Football data: {e}")


def update_nba_data():
    try:
        # Fetch live data
        teams_data = fetch_live_nba_teams_data()
        scores_data = fetch_live_nba_scores_data()

        # Process data
        teams_list = process_nba_teams_data(teams_data)
        scores_list = process_nba_scores_data(scores_data)

        # Merge data
        merged_data = merge_data(teams_list, scores_list)

        # Save data
        save_data(merged_data, 'nba_data.json')

        logger.info("NBA data updated successfully.")
    except Exception as e:
        logger.error(f"Error updating NBA data: {e}")


def update_nfl_data():
    try:
        # Fetch live data
        teams_data = fetch_live_nfl_teams_data()
        scores_data = fetch_live_nfl_scores_data()

        # Process data
        teams_list = process_nfl_teams_data(teams_data)
        scores_list = process_nfl_scores_data(scores_data)

        # Merge data
        merged_data = merge_data(teams_list, scores_list)

        # Save data
        save_data(merged_data, 'nfl_data.json')

        logger.info("NFL data updated successfully.")
    except Exception as e:
        logger.error(f"Error updating NFL data: {e}")


def retrain_models():
    try:
        # Retrain models for each sport
        retrain_model_for_sport('nfl')
        retrain_model_for_sport('cfb')
        retrain_model_for_sport('nba')

        logger.info("Models retrained and saved successfully.")
    except Exception as e:
        logger.error(f"Error retraining models: {e}")


# Call the schedule_jobs function when the app starts
schedule_jobs()

# Schedule college data updates every Friday at 5:00 AM
scheduler.add_job(
    update_college_data,
    trigger=CronTrigger(day_of_week='fri', hour=5, minute=0),
    id='update_college_data',
    replace_existing=True
)

# Schedule NBA data updates twice weekly
scheduler.add_job(
    update_nba_data,
    trigger=CronTrigger(day_of_week='mon,thu', hour=6, minute=0),
    id='update_nba_data',
    replace_existing=True
)

# Schedule NFL data updates every Saturday at 5:00 AM
scheduler.add_job(
    update_nfl_data,
    trigger=CronTrigger(day_of_week='sat', hour=5, minute=0),
    id='update_nfl_data',
    replace_existing=True
)

'*************************************************************'
# Define Routes
@app.route('/')
def home():
    return "Welcome to the Fantasy Sports Optimizer!"

@app.route('/favicon.ico')
def favicon():
    return "", 204

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    df, _, _ = preprocess_data(df)
    xgb_proba = xgb_model.predict(df)
    targets = ['fantasy_points', 'touchdowns', 'yards', 'receptions', 'fumbles', 'interceptions', 'field_goal']
    return jsonify(dict(zip(targets, xgb_proba[0])))

@app.route('/api/predict_weekly', methods=['POST'])
def predict_weekly():
    data = request.json
    df = pd.DataFrame([data])
    df, _, _ = preprocess_data(df)
    predictions = xgb_model.predict(df)
    target_columns = ['fantasy_points', 'touchdowns', 'yards', 'receptions', 'fumbles', 'interceptions', 'field_goal']
    return jsonify({target: float(pred) for target, pred in zip(target_columns, predictions[0])})


'*************************************************************'
def integrate_data(sleeper_data, espn_data):
    """
    Integrate Sleeper and ESPN data into a single DataFrame and preprocess it.
    """
    sleeper_df = pd.DataFrame(sleeper_data)
    espn_df = pd.DataFrame(espn_data)
    
    # Merge the data on the 'player_id' key
    if 'player_id' not in sleeper_df.columns or 'player_id' not in espn_df.columns:
        raise ValueError("Both datasets must contain the 'player_id' column for merging.")
    
    integrated_df = pd.merge(sleeper_df, espn_df, on='player_id', how='inner')
    
    # Add any additional preprocessing as needed
    return integrated_df


def search_players():
    player_name = request.args.get('player_name', default=None)
    team_abbr = request.args.get('team_abbr', default=None)
    stat_filter = request.args.get('stat_filter', default=None)
    output_format = request.args.get('format', default='json')
    sort_by = request.args.get('sort_by', default=None)

    # Fetch player data from Sleeper API
    sleeper_players = player_endpoint.get_all_players(sport='nfl', clean=True)
    #sleeper_players_df = pd.DataFrame(players)

    # Fetch player data from ESPN API
    espn_players = ESPNAPI.get_players('football', 'nfl')
    #espn_players_df = pd.DataFrame(espn_players)

 # Integrate the data
    integrated_df = integrate_data(sleeper_players, espn_players)

    # Apply filters based on query parameters
    if player_name:
        integrated_df = integrated_df[integrated_df['displayName'].str.contains(player_name, case=False)]

    if team_abbr:
        integrated_df = integrated_df[integrated_df['teamAbbr'] == team_abbr.upper()]

    if stat_filter:
        integrated_df = integrated_df[integrated_df[stat_filter] > 0]

    if sort_by:
        integrated_df = integrated_df.sort_values(by=sort_by, ascending=False)

    # Return data as JSON or CSV
    if output_format == 'csv':
        return integrated_df.to_csv(index=False), 200, {'Content-Type': 'text/csv'}
    else:
        return jsonify(integrated_df.to_dict(orient='records'))


'*************************************************************'
class ESPNEndpoint:
    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

    def __init__(self):
        pass

    def get_all_players(self, sport='nfl'):
        """
        Fetch all players from ESPN API.
        """
        url = f"{self.BASE_URL}/players"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch data from ESPN API: {response.status_code}")
        players = response.json()
        return players
    

    

    # Add other ESPN-specific endpoints as needed


'*************************************************************'
# ESPN API Endpoints
@app.route('/api/espn/college_football/news', methods=['GET'])
def get_college_football_news():
    url = 'http://site.api.espn.com/apis/site/v2/sports/football/college-football/news'
    response = requests.get(url)
    if response.status_code == 200:
        pretty_json = json.dumps(response.json(), indent=4)
        return Response(pretty_json, mimetype='application/json')
    return jsonify({"error": "College Football news not found"}), 404

@app.route('/api/espn/college_football/scoreboard', methods=['GET'])
def get_college_football_scores():
    url = 'http://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard'
    params = {'calendar': request.args.get('calendar', 'blacklist'), 'dates': request.args.get('dates')}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        pretty_json = json.dumps(response.json(), indent=4)
        return Response(pretty_json, mimetype='application/json')
    return jsonify({"error": "College Football scores not found"}), 404

@app.route('/api/espn/college_football/game/<gameId>', methods=['GET'])
def get_college_football_game_info(gameId):
    url = f'http://site.api.espn.com/apis/site/v2/sports/football/college-football/summary?event={gameId}'
    response = requests.get(url)
    if response.status_code == 200:
        pretty_json = json.dumps(response.json(), indent=4)
        return Response(pretty_json, mimetype='application/json')
    return jsonify({"error": f"Game {gameId} not found"}), 404

@app.route('/api/espn/college_football/teams/<team>', methods=['GET'])
def get_college_football_team_info(team):
    url = f'http://site.api.espn.com/apis/site/v2/sports/football/college-football/teams/{team}'
    response = requests.get(url)
    if response.status_code == 200:
        pretty_json = json.dumps(response.json(), indent=4)
        return Response(pretty_json, mimetype='application/json')
    return jsonify({"error": f"Team {team} not found"}), 404

@app.route('/api/espn/college_football/rankings', methods=['GET'])
def get_college_football_rankings():
    url = 'http://site.api.espn.com/apis/site/v2/sports/football/college-football/rankings'
    response = requests.get(url)
    if response.status_code == 200:
        pretty_json = json.dumps(response.json(), indent=4)
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
        pretty_json = json.dumps(response.json(), indent=4)
        return Response(pretty_json, mimetype='application/json')
    return jsonify({"error": "NFL news not found"}), 404

@app.route('/api/espn/nfl/teams', methods=['GET'])
def get_nfl_teams():
    url = 'http://site.api.espn.com/apis/site/v2/sports/football/nfl/teams'
    response = requests.get(url)
    if response.status_code == 200:
        pretty_json = json.dumps(response.json(), indent=4)
        return Response(pretty_json, mimetype='application/json')
    return jsonify({"error": "NFL teams not found"}), 404

@app.route('/api/espn/nfl/teams/<team>', methods=['GET'])
def get_nfl_team_info(team):
    url = f'http://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team}'
    response = requests.get(url)
    if response.status_code == 200:
        pretty_json = json.dumps(response.json(), indent=4)
        return Response(pretty_json, mimetype='application/json')
    return jsonify({"error": f"Team {team} not found"}), 404

@app.route('/api/espn/nba/scoreboard', methods=['GET'])
def get_nba_scores():
    url = 'http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'
    response = requests.get(url)
    if response.status_code == 200:
        pretty_json = json.dumps(response.json(), indent=4)
        return Response(pretty_json, mimetype='application/json')
    return jsonify({"error": "NBA scores not found"}), 404

@app.route('/api/espn/nba/news', methods=['GET'])
def get_nba_news():
    url = 'http://site.api.espn.com/apis/site/v2/sports/basketball/nba/news'
    response = requests.get(url)
    if response.status_code == 200:
        pretty_json = json.dumps(response.json(), indent=4)
        return Response(pretty_json, mimetype='application/json')
    return jsonify({"error": "NBA news not found"}), 404

@app.route('/api/espn/nba/teams', methods=['GET'])
def get_nba_teams():
    url = 'http://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams'
    response = requests.get(url)
    if response.status_code == 200:
        pretty_json = json.dumps(response.json(), indent=4)
        return Response(pretty_json, mimetype='application/json')
    return jsonify({"error": "NBA teams not found"}), 404

@app.route('/api/espn/nba/teams/<team>', methods=['GET'])
def get_nba_team_info(team):
    url = f'http://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team}'
    response = requests.get(url)
    if response.status_code == 200:
        pretty_json = json.dumps(response.json(), indent=4)
        return Response(pretty_json, mimetype='application/json')
    return jsonify({"error": f"NBA team {team} not found"}), 404

if __name__ == '__main__':
    schedule_jobs()
    app.run(debug=True, host='0.0.0.0')
