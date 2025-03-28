# Imports
import os
import requests
import json
import logging
import pandas as pd
import joblib
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from backend.init_db import init_db
from backend.routes import register_routes
from datetime import datetime
from backend.models.train_models import save_models
from backend.models.train_models import extract_features_and_targets
from apscheduler.schedulers.background import BackgroundScheduler

from datetime import datetime

from apscheduler.triggers.cron import CronTrigger
import atexit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Import custom modules
from backend.api.sleeper import SleeperAPI
from backend.api.espn import ESPNAPI
from backend.api.player_endpoints import PlayerEndpoint

from backend.utils.data_storage import store_player_game_stats
from backend.utils.data_fetching import (
    fetch_live_teams_data, fetch_live_scores_data,
    fetch_live_nfl_teams_data, fetch_live_nfl_scores_data,
    fetch_live_nba_teams_data, fetch_live_nba_scores_data,
    fetch_live_cfb_teams_data, fetch_live_cfb_scores_data,
    fetch_player_game_stats
)
from backend.utils.data_preprocessing import (
    process_teams_data, process_scores_data, merge_data,
    process_nfl_teams_data, process_nfl_scores_data,
    process_nba_teams_data, process_nba_scores_data,
    process_cfb_teams_data, process_cfb_scores_data,
    process_player_game_stats
)

from backend.services.db_services import (
    update_teams_data, update_schedules, update_box_scores, update_live_box_scores
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sports_optimizer.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
init_db(app)

# Register the routes
register_routes(app)

# Load environment variables
load_dotenv()

# Initialize the scheduler
scheduler = BackgroundScheduler()

# Define save_data function
def save_data(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)

# Define load_cache function
def load_cache():
    if os.path.exists('cache.json'):
        with open('cache.json', 'r') as f:
            return json.load(f)
    return None

# Define save_cache function
def save_cache(data):
    with open('cache.json', 'w') as f:
        json.dump(data, f)

# Initialize Sleeper API client
sleeper_client = SleeperAPI()
player_endpoint = PlayerEndpoint(client=sleeper_client)
espn_client = ESPNAPI()


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

# Model Training Functions
def train_xgboost_model(X, y):
    xgb_model = xgb_model.fit(X, y)
    return xgb_model

def train_dense_nn_model(X, y):
    dense_nn_model = train_dense_nn_model(X, y)
    return dense_nn_model

def train_ensemble_model(X, y, xgb_model):
    # Train ensemble model
    ensemble_model = train_ensemble_model(X, y, xgb_model)
    return ensemble_model

#def save_models(xgb_model, nn_model):
#    joblib.dump(xgb_model, 'xgb_model_full.joblib')
#    nn_model.save('dense_nn_model.h5')

# Model Retraining Function
def retrain_models():
    from backend.models.train_models import train_xgboost_model, train_dense_nn_model
    X, y = prepare_training_data()
    xgb_model = train_xgboost_model(X, y)
    nn_model = train_dense_nn_model(X, y)
    save_models(xgb_model, nn_model)
    logging.info("Models retrained successfully.")


# Update Functions
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





def update_player_game_stats():
    stats_data = fetch_player_game_stats(CURRENT_SEASON, CURRENT_WEEK)
    stats_list = process_player_game_stats(stats_data)
    store_player_game_stats(stats_list)
    logger.info("Player game stats updated.")

def background_task():
        with app.app_context():
            # Example of a database query or another task you want to execute periodically
            result = db.session.query(YourModel).filter(YourModel.some_field == some_value).all()
            # Perform the required operation (logging, saving to db, etc.)
            for row in result:
                # Do something with each row (e.g., print/log data)
                print(f"Running task for: {row}")
            print(f"Task executed at {datetime.now()}")

def schedule_jobs():
    # Schedule college data updates every Friday at 5:00 AM
    scheduler.add_job(
        update_college_data,
        trigger=CronTrigger(day_of_week='fri', hour=5, minute=0),
        id='update_college_data',
        replace_existing=True
    )

    scheduler.add_job(
        background_task,  # The function to run
        'interval',  # The trigger type (can also be 'cron', 'date', etc.)
        minutes=10,  # Set how often it should run (e.g., every 10 minutes)
        id='background_task',  # Optional ID to reference the task
        replace_existing=True  # Replace the existing job if it already exists
    )
 
    scheduler.add_job(
        update_player_game_stats,
        trigger=CronTrigger(hour='*/1'),  # Adjust frequency as needed
        id='update_player_game_stats',
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

    scheduler.add_job(
        update_teams_data,
        trigger=CronTrigger(hour=2, minute=0),
        id='update_teams_data',
        replace_existing=True
    )   


    scheduler.add_job(
        update_schedules,
        trigger=CronTrigger(hour=3, minute=0),
        args=[CURRENT_SEASON],
        id='update_schedules',
        replace_existing=True
    )
    
    scheduler.add_job(
        update_box_scores,
        trigger=CronTrigger(minute='*/1'),
        args=[CURRENT_SEASON, CURRENT_WEEK],
        id='update_box_scores',
        replace_existing=True
    )
    
    scheduler.add_job(
        update_live_box_scores,
        trigger='interval',
        seconds=60,
        id='update_live_box_scores',
        replace_existing=True
    )


    CURRENT_SEASON = 2024
    CURRENT_WEEK = 1
    
    # Scheduler setup
    scheduler = BackgroundScheduler()

    # Start the scheduler
    scheduler.start()

    # Shut down the scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())

# Define Routes
@app.route('/')
def home():
    return "Welcome to the Fantasy Sports Optimizer!"

@app.route('/favicon.ico')
def favicon():
    return "", 204

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    X, _ = extract_features_and_targets(df)
    xgb_model = joblib.load('backend/models/xgb_model_full.joblib')
    prediction = xgb_model.predict(X)
    result = {'FantasyPointsHalfPPR': prediction[0][0]}
    return jsonify(result)


#@app.route('/api/predict_weekly', methods=['POST'])
#def predict_weekly():
#    data = request.json
#    df = pd.DataFrame([data])
#    df, _, _ = preprocess_data(df)
#    predictions = xgb_model.predict(df)
#    target_columns = ['fantasy_points', 'touchdowns', 'yards', 'receptions', 'fumbles', 'interceptions', 'field_goal']
#    return jsonify({target: float(pred) for target, pred in zip(target_columns, predictions[0])})


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
    
if __name__ == '__main__':
    schedule_jobs()
    app.run(debug=True, host='0.0.0.0')
