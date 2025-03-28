from flask import Blueprint, Response, jsonify, request
import json
import requests
import logging
from backend.services.player_service import get_player_data
from backend.api.espn import ESPNAPI
from backend.api.sleeper import SleeperAPI
from backend.api.sportsdataio import SportsDataIOAPI


SPORTSDATAIO_BASE_URL = 'https://api.sportsdata.io/v3'
SLEEPER_BASE_URL = 'https://api.sleeper.app/v1'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define blueprints for different routes
sleeper_routes = Blueprint('sleeper_routes', __name__)
sportsdataio_routes = Blueprint('sportsdataio_routes', __name__)
espn_routes = Blueprint('espn_routes', __name__)

# Initialize API clients
sleeper_client = SleeperAPI()
sportsdataio_client = SportsDataIOAPI()
espn_client = ESPNAPI()

# Register routes function
def register_routes(app):
    # Register Sleeper routes
    app.register_blueprint(sleeper_routes)

    # Register SportsDataIO routes
    app.register_blueprint(sportsdataio_routes)

    # Register ESPN routes
    app.register_blueprint(espn_routes)

    @app.route('/api/players', methods=['GET'])
    def get_players():
        """Endpoint to get player data from a combined data source."""
        try:
            player_data = get_player_data()
            return jsonify(player_data)
        except Exception as e:
            logger.error(f"Error fetching player data: {e}")
            return jsonify({"error": "Unable to fetch player data"}), 500

# ESPN Routes
def register_espn_routes(app):
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
    
    @app.route('/api/espn/nfl/scoreboard', methods=['GET'])
    def get_nfl_scores():
        nfl_scores = ESPNAPI.get_scores("football", "nfl")
        if nfl_scores:
            pretty_json = json.dumps(nfl_scores, indent=4)
            return Response(pretty_json, mimetype='application/json')
        return jsonify({"error": "NFL scores not found"}), 404

    @app.route('/api/espn/nfl/player/<player_id>', methods=['GET'])
    def get_nfl_player_stats(player_id):
        player_stats = ESPNAPI.get_player_stats("football", player_id)
        if player_stats:
            pretty_json = json.dumps(player_stats, indent=4)
            return Response(pretty_json, mimetype='application/json')
        return jsonify({"error": "Player stats not found"}), 404
    
    @app.route('/api/espn/nfl/team/<team_id>', methods=['GET'])
    def get_nfl_team_stats(team_id):
        team_stats = ESPNAPI.get_team_stats("football", team_id)
        if team_stats:
            pretty_json = json.dumps(team_stats, indent=4)
            return Response(pretty_json, mimetype='application/json')
        return jsonify({"error": "Team stats not found"}), 404
    
    @app.route('/api/espn/nfl/teams', methods=['GET'])
    def get_nfl_teams():
        nfl_teams = ESPNAPI.get_teams("football", "nfl")
        if nfl_teams:
            pretty_json = json.dumps(nfl_teams, indent=4)
            return Response(pretty_json, mimetype='application/json')
        return jsonify({"error": "NFL teams not found"}), 404
    
    @app.route('/api/espn/nfl/news', methods=['GET'])
    def get_nfl_news():
        url = 'http://site.api.espn.com/apis/site/v2/sports/football/nfl/news'
        response = requests.get(url)
        if response.status_code == 200:
            pretty_json = json.dumps(response.json(), indent=4)
            return Response(pretty_json, mimetype='application/json')
        return jsonify({"error": "NFL news not found"}), 404
    
    @app.route('/api/espn/nfl/rankings', methods=['GET'])
    def get_nfl_rankings():
        url = 'http://site.api.espn.com/apis/site/v2/sports/football/nfl/rankings'
        response = requests.get(url)
        if response.status_code == 200:
            pretty_json = json.dumps(response.json(), indent=4)
            return Response(pretty_json, mimetype='application/json')
        return jsonify({"error": "NFL rankings not found"}), 404
    
    @app.route('/api/espn/nfl/schedule', methods=['GET'])
    def get_nfl_schedule():
        url = 'http://site.api.espn.com/apis/site/v2/sports/football/nfl/schedule'
        response = requests.get(url)
        if response.status_code == 200:
            pretty_json = json.dumps(response.json(), indent=4)
            return Response(pretty_json, mimetype='application/json')
        return jsonify({"error": "NFL schedule not found"}), 404

# Sleeper Routes
def register_sleeper_routes(app):
    @app.route('/api/sleeper/user/<username_or_id>', methods=['GET'])
    def get_sleeper_user(username_or_id):
        url = f"{SLEEPER_BASE_URL}/user/{username_or_id}"
        response = requests.get(url)
        if response.status_code == 200:
            pretty_json = json.dumps(response.json(), indent=4)
            return Response(pretty_json, mimetype='application/json')
        return jsonify({"error": "User not found"}), 404

    @app.route('/api/sleeper/leagues/<user_id>/<sport>/<season>', methods=['GET'])
    def get_sleeper_leagues(user_id, sport, season):
        url = f"{SLEEPER_BASE_URL}/user/{user_id}/leagues/{sport}/{season}"
        response = requests.get(url)
        if response.status_code == 200:
            pretty_json = json.dumps(response.json(), indent=4)
            return Response(pretty_json, mimetype='application/json')
        return jsonify({"error": "Leagues not found"}), 404

    @app.route('/api/sleeper/draft/<draft_id>', methods=['GET'])
    def get_sleeper_draft(draft_id):
        url = f"{SLEEPER_BASE_URL}/draft/{draft_id}"
        response = requests.get(url)
        if response.status_code == 200:
            pretty_json = json.dumps(response.json(), indent=4)
            return Response(pretty_json, mimetype='application/json')
        return jsonify({"error": "Draft not found"}), 404
    

# SportsDataIO Routes
def register_sportsdataio_routes(app):
    @app.route('/api/sportsdataio/nfl/stats', methods=['GET'])
    def get_nfl_stats():
        season = request.args.get('season', '2024')
        week = request.args.get('week', '1')
        api_key = 'SPROTSDATAIO_API_KEY'  # Replace with actual key

        url = f"{SPORTSDATAIO_BASE_URL}/nfl/stats/{season}/{week}?key={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            pretty_json = json.dumps(response.json(), indent=4)
            return Response(pretty_json, mimetype='application/json')
        return jsonify({"error": "NFL stats not found"}), 404

    @app.route('/api/sportsdataio/nfl/player/<player_id>', methods=['GET'])
    def get_nfl_player_stats(player_id):
        api_key = 'SPROTSDATAIO_API_KEY'  # Replace with actual key

        url = f"{SPORTSDATAIO_BASE_URL}/nfl/stats/player/{player_id}?key={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            pretty_json = json.dumps(response.json(), indent=4)
            return Response(pretty_json, mimetype='application/json')
        return jsonify({"error": "Player stats not found"}), 404

    @app.route('/api/sportsdataio/nfl/teams', methods=['GET'])
    def get_nfl_teams():
        api_key = 'SPROTSDATAIO_API_KEY'  # Replace with actual key

        url = f"{SPORTSDATAIO_BASE_URL}/nfl/teams?key={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            pretty_json = json.dumps(response.json(), indent=4)
            return Response(pretty_json, mimetype='application/json')
        return jsonify({"error": "NFL teams not found"}), 404
    
    # Add more routes for fetching other data such as box scores, odds, etc.
    @app.route('/api/sportsdataio/nfl/box_scores', methods=['GET'])
    def get_nfl_box_scores():
        season = request.args.get('season', '2024')
        week = request.args.get('week', '1')
        api_key = "SPROTSDATAIO_API_KEY"

        url = f"{SPORTSDATAIO_BASE_URL}/nfl/box_scores/{season}/{week}?key={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            pretty_json = json.dumps(response.json(), indent=4)
            return Response(pretty_json, mimetype='application/json')
        return jsonify({"error": "NFL box scores not found"}), 404
    
    @app.route('/api/sportsdataio/nfl/odds', methods=['GET'])
    def get_nfl_odds():
        season = request.args.get('season', '2024')
        week = request.args.get('week', '1')
        api_key = "SPROTSDATAIO_API_KEY"

        url = f"{SPORTSDATAIO_BASE_URL}/nfl/odds/{season}/{week}?key={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            pretty_json = json.dumps(response.json(), indent=4)
            return Response(pretty_json, mimetype='application/json')
        return jsonify({"error": "NFL odds not found"}), 404
    
    
    # Add more routes for fetching other data such as box scores, odds, etc.

