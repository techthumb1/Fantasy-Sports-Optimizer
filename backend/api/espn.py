import requests

class ESPNAPI:
    BASE_URL = "https://site.api.espn.com/apis/v2"

    @staticmethod
    def get_player_stats(sport, player_id):
        url = f"{ESPNAPI.BASE_URL}/sports/{sport}/athletes/{player_id}/stats"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return None

    @staticmethod
    def get_team_stats(sport, team_id):
        url = f"{ESPNAPI.BASE_URL}/sports/{sport}/teams/{team_id}/statistics"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return None

# Example usage:
# stats = ESPNAPI.get_player_stats("basketball", "player_id_here")
# team_stats = ESPNAPI.get_team_stats("football", "team_id_here")
