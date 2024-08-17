import requests

class SleeperAPI:
    BASE_URL = "https://api.sleeper.app/v1"

    @staticmethod
    def get_player_stats(player_id):
        url = f"{SleeperAPI.BASE_URL}/player/{player_id}/stats"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return None

    @staticmethod
    def get_league_standings(league_id):
        url = f"{SleeperAPI.BASE_URL}/league/{league_id}/standings"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return None

# Example usage:
# stats = SleeperAPI.get_player_stats("player_id_here")
# standings = SleeperAPI.get_league_standings("league_id_here")
