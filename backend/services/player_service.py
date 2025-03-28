from backend.api.sleeper import SleeperAPI
from backend.utils.formatter import pretty_print_json

class PlayerService:
    @staticmethod
    def get_user_data(username_or_id):
        user_data = SleeperAPI.get_user(username_or_id)
        if user_data:
            return pretty_print_json(user_data)
        return {"error": "User not found"}, 404

    @staticmethod
    def search_players():
        # Add the search logic
        pass

# backend/services/player_service.py
def get_player_data(player_id):
    from database_models import Player  # Lazy import
    # Fetch the player data
    player = Player.query.filter_by(player_id=player_id).first()
    if player:
        return player.to_dict()
    return {"error": "Player not found"}
