import logging

logger = logging.getLogger(__name__)

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

def process_teams_data(teams_data):
    teams_list = []
    for team in teams_data:
        teams_list.append({
            'TeamID': team.get('TeamID'),
            'Name': team.get('Name'),
            # Map other fields as needed
        })
    return teams_list

def process_schedules(schedules_data):
    games_list = []
    for game in schedules_data:
        games_list.append({
            'GameID': game.get('GameID'),
            'Season': game.get('Season'),
            'Week': game.get('Week'),
            'HomeTeam': game.get('HomeTeam'),
            'AwayTeam': game.get('AwayTeam'),
            'Status': game.get('Status'),
            # Map other fields as needed
        })
    return games_list

def process_box_scores(box_scores_data):
    player_stats_list = []
    for game in box_scores_data:
        game_id = game.get('Game', {}).get('GameID')
        for player_stat in game.get('PlayerGames', []):
            player_stats_list.append({
                'PlayerID': player_stat.get('PlayerID'),
                'GameID': game_id,
                'TeamID': player_stat.get('TeamID'),
                'FantasyPoints': player_stat.get('FantasyPoints'),
                'Touchdowns': player_stat.get('Touchdowns'),
                'Yards': player_stat.get('Yards'),
                'Receptions': player_stat.get('Receptions'),
                'Fumbles': player_stat.get('FumblesLost'),
                'Interceptions': player_stat.get('Interceptions'),
                'FieldGoals': player_stat.get('FieldGoalsMade'),
                # Map other fields as needed
            })
    return player_stats_list

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

def process_player_game_stats(stats_data):
    stats_list = []
    for player_stat in stats_data:
        stats_list.append({
            'PlayerID': player_stat.get('PlayerID'),
            'GameID': player_stat.get('GameID'),
            'TeamID': player_stat.get('TeamID'),
            'Season': player_stat.get('Season'),
            'Week': player_stat.get('Week'),
            'RushingAttempts': player_stat.get('RushingAttempts'),
            'RushingYards': player_stat.get('RushingYards'),
            'RushingTouchdowns': player_stat.get('RushingTouchdowns'),
            'ReceivingTargets': player_stat.get('ReceivingTargets'),
            'Receptions': player_stat.get('Receptions'),
            'ReceivingYards': player_stat.get('ReceivingYards'),
            'ReceivingTouchdowns': player_stat.get('ReceivingTouchdowns'),
            'PassingAttempts': player_stat.get('PassingAttempts'),
            'PassingCompletions': player_stat.get('PassingCompletions'),
            'PassingInterceptions': player_stat.get('PassingInterceptions'),
            'PassingYards': player_stat.get('PassingYards'),
            'PassingTouchdowns': player_stat.get('PassingTouchdowns'),
            'FieldGoalsAttempted': player_stat.get('FieldGoalsAttempted'),
            'FieldGoalsMade': player_stat.get('FieldGoalsMade'),
            'FieldGoalsMissed': player_stat.get('FieldGoalsMissed'),
            'ExtraPointAttempts': player_stat.get('ExtraPointAttempts'),
            'ExtraPointsMade': player_stat.get('ExtraPointsMade'),
            'ExtraPointsMissed': player_stat.get('ExtraPointsMissed'),
            'PassingSacks': player_stat.get('PassingSacks'),
            'LongestRush': player_stat.get('LongestRush'),
            'LongestReception': player_stat.get('LongestReception'),
            'LongestPass': player_stat.get('LongestPass'),
            'FantasyPointsHalfPPR': player_stat.get('FantasyPointsHalfPointPPR'),
        })
    return stats_list

