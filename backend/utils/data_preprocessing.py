# backend/utils/data_processing.py

def process_teams_data(teams_data):
    # Generic processing of teams data
    teams_list = []
    for team_info in teams_data.get('sports', [])[0].get('leagues', [])[0].get('teams', []):
        team = team_info.get('team', {})
        teams_list.append({
            'id': team.get('id'),
            'abbreviation': team.get('abbreviation'),
            'displayName': team.get('displayName'),
            # Add other relevant fields
        })
    return teams_list

def process_scores_data(scores_data):
    # Generic processing of scores data
    scores_list = []
    for event in scores_data.get('events', []):
        competition = event.get('competitions', [])[0]
        for competitor in competition.get('competitors', []):
            scores_list.append({
                'game_id': event.get('id'),
                'date': event.get('date'),
                'team_id': competitor['team']['id'],
                'homeAway': competitor.get('homeAway'),
                'score': competitor.get('score'),
                # Add more fields as needed
            })
    return scores_list

def merge_data(teams_list, scores_list):
    # Merge teams and scores data based on team_id
    valid_team_ids = {team['id'] for team in teams_list}
    merged_data = [
        {**score, **team}
        for score in scores_list if score['team_id'] in valid_team_ids
        for team in [next((team for team in teams_list if team['id'] == score['team_id']), None)] if team
    ]
    if not merged_data:
        raise ValueError("No valid data to process after merging.")
    return merged_data

def process_cfb_teams_data(teams_data):
    # Process college football teams data
    teams_list = []
    for team_info in teams_data.get('sports', [])[0].get('leagues', [])[0].get('teams', []):
        team = team_info.get('team', {})
        teams_list.append({
            'id': team.get('id'),
            'abbreviation': team.get('abbreviation'),
            'displayName': team.get('displayName'),
            # Add other relevant fields
        })
    return teams_list

def process_cfb_scores_data(scores_data):
    # Process college football scores data
    scores_list = []
    for event in scores_data.get('events', []):
        competition = event.get('competitions', [])[0]
        for competitor in competition.get('competitors', []):
            scores_list.append({
                'game_id': event.get('id'),
                'date': event.get('date'),
                'team_id': competitor['team']['id'],
                'homeAway': competitor.get('homeAway'),
                'score': competitor.get('score'),
                # Add other relevant fields
            })
    return scores_list

def process_nba_teams_data(teams_data):
    # Process NBA teams data
    teams_list = []
    for team_info in teams_data.get('sports', [])[0].get('leagues', [])[0].get('teams', []):
        team = team_info.get('team', {})
        teams_list.append({
            'id': team.get('id'),
            'abbreviation': team.get('abbreviation'),
            'displayName': team.get('displayName'),
            # Add other relevant fields
        })
    return teams_list

def process_nba_scores_data(scores_data):
    # Process NBA scores data
    scores_list = []
    for event in scores_data.get('events', []):
        competition = event.get('competitions', [])[0]
        for competitor in competition.get('competitors', []):
            scores_list.append({
                'game_id': event.get('id'),
                'date': event.get('date'),
                'team_id': competitor['team']['id'],
                'homeAway': competitor.get('homeAway'),
                'score': competitor.get('score'),
                # Add other relevant fields
            })
    return scores_list

def process_nfl_teams_data(teams_data):
    # Process NFL teams data
    teams_list = []
    for team_info in teams_data.get('sports', [])[0].get('leagues', [])[0].get('teams', []):
        team = team_info.get('team', {})
        teams_list.append({
            'id': team.get('id'),
            'abbreviation': team.get('abbreviation'),
            'displayName': team.get('displayName'),
            # Add other relevant fields
        })
    return teams_list

def process_nfl_scores_data(scores_data):
    # Process NFL scores data
    scores_list = []
    for event in scores_data.get('events', []):
        competition = event.get('competitions', [])[0]
        for competitor in competition.get('competitors', []):
            scores_list.append({
                'game_id': event.get('id'),
                'date': event.get('date'),
                'team_id': competitor['team']['id'],
                'homeAway': competitor.get('homeAway'),
                'score': competitor.get('score'),
                # Add other relevant fields
            })
    return scores_list
