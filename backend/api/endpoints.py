import json
from flask import request, jsonify, Response
from backend import app
from backend.api import SleeperAPI
from backend.api import player_endpoint

# Sleeper API Endpoints
@app.route('/api/sleeper/user_data', methods=['POST'])
def get_sleeper_user_data():
    username_or_id = request.json.get('username_or_id')
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
def get_sleeper_players_sport(sport):
    players = SleeperAPI.get_players(sport)
    if players:
        pretty_json = json.dumps(players, indent=4)
        return Response(pretty_json, mimetype='application/json')
    return jsonify({"error": "Players not found"}), 404

#@app.route('/api/sleeper/players', methods=['GET'])
#def get_all_players():
#    sport = request.args.get('sport', 'nfl')
#    players = player_endpoint.get_all_players(sport)
#    response = [player.to_dict() for player in players]
#    pretty_response = json.dumps(response, indent=4)
#    return app.response_class(pretty_response, content_type='application/json')

@app.route('/api/sleeper/players', methods=['GET'])
def get_sleeper_players():
    sport = request.args.get('sport', 'nfl')  # Default to 'nfl' if not provided
    try:
        players = SleeperAPI.get_players(sport)
        return jsonify(players)  # Make sure this returns valid JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/sleeper/trending', methods=['GET'])
def get_trending_players():
    sport = request.args.get('sport', 'nfl')
    trend_type = request.args.get('trend_type', 'add')
    lookback_hours = int(request.args.get('lookback_hours', 24))
    limit = int(request.args.get('limit', 25))
    players = player_endpoint.get_trending_players(sport, trend_type, lookback_hours, limit)
    response = [player.to_dict() for player in players]
    pretty_response = json.dumps(response, indent=4)
    return app.response_class(pretty_response, content_type='application/json')

@app.route('/api/sleeper/players/team/<team_abbr>', methods=['GET'])
def get_players_by_team(team_abbr):
    players = player_endpoint.get_players_by_team(team_abbr)
    response = [player.to_dict() for player in players]
    pretty_response = json.dumps(response, indent=4)
    return app.response_class(pretty_response, content_type='application/json')
