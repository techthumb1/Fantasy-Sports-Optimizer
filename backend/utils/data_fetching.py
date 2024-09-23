import requests
import logging
import time

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

import requests

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
