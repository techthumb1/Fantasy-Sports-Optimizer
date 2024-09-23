from flask import Blueprint, jsonify
import requests

sleeper_routes = Blueprint('sleeper_routes', __name__)

SLEEPER_BASE_URL = "https://api.sleeper.app/v1"

@sleeper_routes.route('/api/sleeper/user/<username_or_id>', methods=['GET'])
def get_sleeper_user(username_or_id):
    url = f"{SLEEPER_BASE_URL}/user/{username_or_id}"
    response = requests.get(url, timeout=5)
    if response.status_code == 200:
        return jsonify(response.json())
    return jsonify({"error": "User not found"}), 404

@sleeper_routes.route('/api/sleeper/leagues/<user_id>/<sport>/<season>', methods=['GET'])
def get_sleeper_leagues(user_id, sport, season):
    url = f"{SLEEPER_BASE_URL}/user/{user_id}/leagues/{sport}/{season}"
    response = requests.get(url, timeout=5)
    if response.status_code == 200:
        return jsonify(response.json())
    return jsonify({"error": "Leagues not found"}), 404

@sleeper_routes.route('/api/sleeper/league/<league_id>', methods=['GET'])
def get_sleeper_league(league_id):
    url = f"{SLEEPER_BASE_URL}/league/{league_id}"
    response = requests.get(url, timeout=5)
    if response.status_code == 200:
        return jsonify(response.json())
    return jsonify({"error": "League not found"}), 404

@sleeper_routes.route('/api/sleeper/drafts/<user_id>/<sport>/<season>', methods=['GET'])
def get_sleeper_drafts(user_id, sport, season):
    url = f"{SLEEPER_BASE_URL}/user/{user_id}/drafts/{sport}/{season}"
    response = requests.get(url, timeout=5)
    if response.status_code == 200:
        return jsonify(response.json())
    return jsonify({"error": "Drafts not found"}), 404

@sleeper_routes.route('/api/sleeper/draft/<draft_id>', methods=['GET'])
def get_sleeper_draft(draft_id):
    url = f"{SLEEPER_BASE_URL}/draft/{draft_id}"
    response = requests.get(url, timeout=5)
    if response.status_code == 200:
        return jsonify(response.json())
    return jsonify({"error": "Draft not found"}), 404

@sleeper_routes.route('/api/sleeper/players/<sport>', methods=['GET'])
def get_sleeper_players(sport):
    url = f"{SLEEPER_BASE_URL}/players/{sport}"
    response = requests.get(url, timeout=5)
    if response.status_code == 200:
        return jsonify(response.json())
    return jsonify({"error": "Players not found"}), 404

