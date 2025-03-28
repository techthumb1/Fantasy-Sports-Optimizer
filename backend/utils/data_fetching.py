import requests
import logging
import time
import os

API_KEY = os.getenv('SPORTSDATAIO_API_KEY')
BASE_URL = 'https://api.sportsdata.io/v3/nfl'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_live_scores_data():
    try:
        url = 'http://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard'
        response = requests.get(url)
        response.raise_for_status()
        logger.info('Successfully fetched live scores data.')
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching live scores data: {e}")
        print(f"Error fetching live scores data: {e}")
  
        return None

def fetch_live_teams_data():
    try:
        url = 'http://site.api.espn.com/apis/site/v2/sports/football/nfl/teams'
        response = requests.get(url)
        response.raise_for_status()
        logger.info('Successfully fetched live teams data.')
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching live teams data: {e}")
        print(f"Error fetching live teams data: {e}")

        return None

def fetch_with_retries(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Failed to fetch data after {max_retries} attempts.")
                raise

def fetch_player_game_stats(season, week):
    url = f"{BASE_URL}/stats/json/PlayerGameStatsByWeek/{season}/{week}"
    headers = {'Ocp-Apim-Subscription-Key': API_KEY}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def fetch_betting_player_props(game_id):
    url = f"{BASE_URL}/odds/json/BettingPlayerPropsByGameID/{game_id}"
    headers = {'Ocp-Apim-Subscription-Key': API_KEY}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def fetch_teams():
    url = f"{BASE_URL}/scores/json/Teams"
    headers = {'Ocp-Apim-Subscription-Key': API_KEY}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    logger.info("Successfully fetched teams data.")
    return response.json()

def fetch_nfl_schedules(season):
    url = f"{BASE_URL}/scores/json/Games/{season}"
    headers = {'Ocp-Apim-Subscription-Key': API_KEY}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    logger.info(f"Successfully fetched NFL schedules for season {season}.")
    return response.json()

def fetch_nfl_box_scores(season, week):
    url = f"{BASE_URL}/stats/json/BoxScoresByWeek/{season}/{week}"
    headers = {'Ocp-Apim-Subscription-Key': API_KEY}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    logger.info(f"Successfully fetched NFL box scores for season {season}, week {week}.")
    return response.json()

def fetch_live_box_scores_delta(minutes):
    url = f"{BASE_URL}/stats/json/LiveBoxScoresDelta/{minutes}"
    headers = {'Ocp-Apim-Subscription-Key': API_KEY}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    logger.info(f"Successfully fetched live box scores delta for the past {minutes} minutes.")
    return response.json()

def fetch_live_cfb_teams_data():
    url = 'http://site.api.espn.com/apis/site/v2/sports/football/college-football/teams'
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def fetch_live_cfb_scores_data():
    url = 'http://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard'
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def fetch_live_nba_teams_data():
    url = 'http://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams'
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def fetch_live_nba_scores_data():
    url = 'http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def fetch_live_nfl_teams_data():
    url = 'http://site.api.espn.com/apis/site/v2/sports/football/nfl/teams'
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def fetch_live_nfl_scores_data():
    url = 'http://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard'
    response = requests.get(url)
    response.raise_for_status()
    return response.json()
