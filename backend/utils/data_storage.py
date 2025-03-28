# backend/utils/data_storage.py

import logging
import pandas as pd
from backend import app
from backend.models.database import db
from backend.models.database_models import PlayerStat, Player, Team, Game


logger = logging.getLogger(__name__)

def get_player_data():
    return Player.query.all()

def store_player_game_stats(stats_list):
    with app.app_context():
        for stat_info in stats_list:
            player_stat = PlayerStat.query.filter_by(
                PlayerID=stat_info['PlayerID'],
                GameID=stat_info['GameID']
            ).first()
            if player_stat:
                # Update existing record
                for key, value in stat_info.items():
                    setattr(player_stat, key, value)
            else:
                # Create new record
                player_stat = PlayerStat(**stat_info)
                db.session.add(player_stat)
        db.session.commit()
        logger.info("Player game stats stored successfully.")


def load_data_from_database():
    with app.app_context():
        # Query data from PlayerStat, Player, Team, and Game models
        player_stats = db.session.query(
            PlayerStat,
            Player,
            Team,
            Game
        ).join(
            Player, PlayerStat.PlayerID == Player.PlayerID
        ).join(
            Team, PlayerStat.TeamID == Team.TeamID
        ).join(
            Game, PlayerStat.GameID == Game.GameID
        ).all()

        # Construct a DataFrame
        data = []
        for stat, player, team, game in player_stats:
            record = stat.to_dict()
            record.update({
                'team_id': team.TeamID,
                'abbreviation': team.Abbreviation,
                'location': team.City,
                'homeAway': 'Home' if game.HomeTeamID == team.TeamID else 'Away',
                'game_id': game.GameID,
                'score': game.HomeScore if game.HomeTeamID == team.TeamID else game.AwayScore,
                'date': game.Date
            })
            data.append(record)

        df = pd.DataFrame(data)
        return df
