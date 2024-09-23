# backend/utils/data_processing.py

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
