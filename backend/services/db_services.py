import pandas as pd
from backend.models.database_models import Team, Game, PlayerStat
from backend.utils.data_preprocessing import process_box_scores
from backend.utils.data_fetching import fetch_live_box_scores_delta
from backend import app
from backend.models.database import db
import logging

logger = logging.getLogger(__name__)

def update_teams_data(teams_list):
    with app.app_context():
        for team_info in teams_list:
            team = Team.query.filter_by(TeamID=team_info['TeamID']).first()
            if team:
                # Update existing team
                team.Name = team_info['Name']
                # Update other fields as needed
            else:
                # Add new team
                team = Team(**team_info)
                db.session.add(team)
        db.session.commit()
    logger.info("Teams data updated successfully.")

def update_schedules(games_list):
    with app.app_context():
        for game_info in games_list:
            game = Game.query.filter_by(GameID=game_info['GameID']).first()
            if game:
                # Update existing game
                game.Status = game_info['Status']
                # Update other fields as needed
            else:
                # Add new game
                game = Game(**game_info)
                db.session.add(game)
        db.session.commit()
    logger.info("Schedules updated successfully.")

def update_box_scores(player_stats_list):
    with app.app_context():
        for stat_info in player_stats_list:
            player_stat = PlayerStat.query.filter_by(
                PlayerID=stat_info['PlayerID'],
                GameID=stat_info['GameID']
            ).first()
            if player_stat:
                # Update existing player stat
                player_stat.FantasyPoints = stat_info['FantasyPoints']
                # Update other fields as needed
            else:
                # Add new player stat
                player_stat = PlayerStat(**stat_info)
                db.session.add(player_stat)
        db.session.commit()
    logger.info("Box scores updated successfully.")

def update_live_box_scores():
    live_data = fetch_live_box_scores_delta(minutes=1)
    player_stats_list = process_box_scores(live_data)
    with app.app_context():
        for stat_info in player_stats_list:
            player_stat = PlayerStat.query.filter_by(
                PlayerID=stat_info['PlayerID'],
                GameID=stat_info['GameID']
            ).first()
            if player_stat:
                # Update existing player stat
                player_stat.FantasyPoints = stat_info['FantasyPoints']
                # Update other fields as needed
            else:
                # Add new player stat
                player_stat = PlayerStat(**stat_info)
                db.session.add(player_stat)
        db.session.commit()
    logger.info("Live box scores updated successfully.")

def load_data_from_database():
    # Query the database to load your data
    query = "SELECT * FROM player_stats"  # Replace with your actual table and query
    df = pd.read_sql(query, db.engine)  # Load data into a pandas DataFrame
    return df
