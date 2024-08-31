import os
import json
import requests
from dotenv import load_dotenv



#class ESPNAPI:
#
#    @staticmethod
#    def get_teams(sport, league):
#        if sport == "football" and league == "nfl":
#            url = "http://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
#        elif sport == "basketball" and league == "nba":
#            url = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
#        else:
#            raise ValueError("Invalid sport or league provided.")
#
#        response = requests.get(url)
#        
#        if response.status_code == 200:
#            return response.json().get('sports', [])[0]['leagues'][0]['teams']
#        else:
#            print(f"Failed to fetch teams: {response.status_code}")
#            return None
#
#    @staticmethod
#    def get_scores(sport, league):
#        if sport == "football" and league == "nfl":
#            url = "http://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
#        elif sport == "basketball" and league == "nba":
#            url = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
#        else:
#            raise ValueError("Invalid sport or league provided.")
#
#        response = requests.get(url)
#        
#        if response.status_code == 200:
#            return response.json().get('events', [])
#        else:
#            print(f"Failed to fetch scores: {response.status_code}")
#            return None
#
#
#
## Load environment variables
#load_dotenv()
#
#def fetch_and_save_json():
#    sport = "football"  # or "basketball"
#    league = "nfl"  # or "nba"
#
#    try:
#        print("Fetching teams data from ESPN API...")
#        teams = ESPNAPI.get_teams(sport=sport, league=league)
#        
#        if teams:
#            with open('teams_data.json', 'w') as f:
#                json.dump(teams, f, indent=4)
#            print("Teams data saved to teams_data.json")
#        else:
#            print("No teams data fetched or empty response.")
#    
#    except Exception as e:
#        print(f"An error occurred while fetching teams data: {e}")
#
#    try:
#        print("Fetching scores data from ESPN API...")
#        scores = ESPNAPI.get_scores(sport=sport, league=league)
#        
#        if scores:
#            with open('scores_data.json', 'w') as f:
#                json.dump(scores, f, indent=4)
#            print("Scores data saved to scores_data.json")
#        else:
#            print("No scores data fetched or empty response.")
#    
#    except Exception as e:
#        print(f"An error occurred while fetching scores data: {e}")
#
#if __name__ == "__main__":
#    fetch_and_save_json()


def load_json_data(file_path):
    """Load JSON data from a file and print its structure."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Print the top-level structure of the JSON
    print(json.dumps(data, indent=4))  # Pretty-print the JSON to understand its structure
    
    return data

# Load the teams data and print its structure
teams_file_path = 'teams_data.json'
teams_data = load_json_data(teams_file_path)

# Load the scores data and print its structure
scores_file_path = 'scores_data.json'
scores_data = load_json_data(scores_file_path)
