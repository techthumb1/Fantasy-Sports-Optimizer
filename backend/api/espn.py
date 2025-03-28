from flask import Blueprint, jsonify, request
import requests

espn_blueprint = Blueprint('espn', __name__)

BASE_URL = "http://site.api.espn.com/apis/site/v2/sports/football/nfl"

@espn_blueprint.route('/scoreboard', methods=['GET'])
def get_scoreboard():
    response = requests.get(f"{BASE_URL}/scoreboard")
    if response.status_code == 200:
        return jsonify(response.json()), 200
    return jsonify({"error": "Failed to fetch scoreboard"}), response.status_code

@espn_blueprint.route('/teams/<team>', methods=['GET'])
def get_team_info(team):
    response = requests.get(f"{BASE_URL}/teams/{team}")
    if response.status_code == 200:
        return jsonify(response.json()), 200
    return jsonify({"error": f"Failed to fetch data for team {team}"}), response.status_code

@espn_blueprint.route('/standings', methods=['GET'])
def get_standings():
    response = requests.get(f"{BASE_URL}/standings")
    if response.status_code == 200:
        return jsonify(response.json()), 200
    return jsonify({"error": "Failed to fetch standings"}), response.status_code

# Fetch the NFL scoreboard
# stats = ESPNAPI.get_player_stats("basketball", "player_id_here")
# team_stats = ESPNAPI.get_team_stats("football", "team_id_here")
