import os
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from api.sleeper import SleeperAPI
from api.espn import ESPNAPI

app = Flask(__name__)

load_dotenv()

SLEEPER_API_KEY = os.getenv('SLEEPER_API_KEY')
ESPN_API_KEY = os.getenv('ESPN_API_KEY')


@app.route('/api/sleeper/player_stats/<player_id>', methods=['GET'])
def get_sleeper_player_stats(player_id):
    stats = SleeperAPI.get_player_stats(player_id)
    return jsonify(stats) if stats else jsonify({"error": "Player not found"}), 404

@app.route('/api/espn/player_stats/<sport>/<player_id>', methods=['GET'])
def get_espn_player_stats(sport, player_id):
    stats = ESPNAPI.get_player_stats(sport, player_id)
    return jsonify(stats) if stats else jsonify({"error": "Player not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
