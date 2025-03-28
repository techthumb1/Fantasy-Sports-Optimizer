import requests
import logging

class SleeperAPI:
    BASE_URL = "https://api.sleeper.app/v1"

    @classmethod
    def get_user(cls, username_or_id):
        response = requests.get(f"{cls.BASE_URL}/user/{username_or_id}")
        if response.status_code == 200:
            return response.json()
        logging.error(f"Failed to fetch user {username_or_id}. Status code: {response.status_code}")
        return None
    
    @classmethod
    def get_leagues(cls, user_id, sport, season):
        response = requests.get(f"{cls.BASE_URL}/user/{user_id}/leagues/{sport}/{season}")
        if response.status_code == 200:
            return response.json()
        return None
    
    @classmethod
    def get_league(cls, league_id):
        response = requests.get(f"{cls.BASE_URL}/league/{league_id}")
        if response.status_code == 200:
            return response.json()
        return None
    
    @classmethod
    def get_rosters(cls, league_id):
        response = requests.get(f"{cls.BASE_URL}/league/{league_id}/rosters")
        if response.status_code == 200:
            return response.json()
        return None
    
    @classmethod
    def get_users(cls, league_id):
        response = requests.get(f"{cls.BASE_URL}/league/{league_id}/users")
        if response.status_code == 200:
            return response.json()
        return None
    
    @classmethod
    def get_matchups(cls, league_id, week):
        response = requests.get(f"{cls.BASE_URL}/league/{league_id}/matchups/{week}")
        if response.status_code == 200:
            return response.json()
        return None
    
    @classmethod
    def get_draft(cls, draft_id):
        response = requests.get(f"{cls.BASE_URL}/draft/{draft_id}")
        if response.status_code == 200:
            return response.json()
        return None
    
    @classmethod
    def get_draft_picks(cls, draft_id):
        response = requests.get(f"{cls.BASE_URL}/draft/{draft_id}/picks")
        if response.status_code == 200:
            return response.json()
        return None

    @classmethod
    def handle_response(cls, response):
        if response.status_code == 200:
            return response.json()
        logging.error(f"API call failed: {response.url} | Status Code: {response.status_code}")
        return None

    @classmethod
    def get_user(cls, username_or_id):
        response = requests.get(f"{cls.BASE_URL}/user/{username_or_id}")
        return cls.handle_response(response)

    @classmethod
    def get_players(cls, sport):
        response = requests.get(f"{cls.BASE_URL}/players/{sport}")
        if response.status_code == 200:
            return response.json()
        return None