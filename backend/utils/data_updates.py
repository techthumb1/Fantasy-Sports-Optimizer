def update_nfl_data():
    # Fetch and process NFL data
    teams_data = fetch_live_nfl_teams_data()
    scores_data = fetch_live_nfl_scores_data()
    # Process and store data

    print("NFL data updated successfully.")

def update_nba_data():
    # Fetch and process NBA data
    teams_data = fetch_live_nba_teams_data()
    scores_data = fetch_live_nba_scores_data()
    # Process and store data

    print("NBA data updated successfully.")

def update_college_data():
    # Fetch and process college data
    teams_data = fetch_live_college_teams_data()
    scores_data = fetch_live_college_scores_data()
    # Process and store data

    print("College data updated successfully.")
