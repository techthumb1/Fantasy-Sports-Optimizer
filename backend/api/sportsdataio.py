import logging
import time

from backend.utils.data_fetching import (
    fetch_player_game_stats,
    fetch_betting_player_props,
    fetch_teams,
    fetch_nfl_schedules,
    fetch_nfl_box_scores,
    fetch_live_box_scores_delta
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("sportsdataio_api.log"),
        logging.StreamHandler()
    ]
)


class SportsDataIOAPI:
    def __init__(self):
        logger.info("SportsDataIO API initialized")

    def get_player_game_stats(self, season, week):
        """Fetch player game stats for a specific NFL season and week."""
        if not isinstance(season, int) or not isinstance(week, int):
            logger.error("Invalid season or week type. Both must be integers.")
            raise ValueError("Season and week must be integers.")
        try:
            stats = fetch_player_game_stats(season, week)
            logger.info(f"Fetched player game stats for season {season}, week {week}.")
            return stats
        except Exception as e:
            logger.error(f"Error fetching player game stats: {e}", exc_info=True)
            raise


    def get_betting_player_props(self, game_id):
        """Fetch betting player props for a specific game."""
        try:
            props = fetch_betting_player_props(game_id)
            logger.info(f"Fetched betting player props for game {game_id}.")
            return props
        except Exception as e:
            logger.error(f"Error fetching betting player props: {e}")
            raise

    def get_teams(self):
        """Fetch NFL teams."""
        try:
            teams = fetch_teams()
            logger.info("Fetched NFL teams.")
            return teams
        except Exception as e:
            logger.error(f"Error fetching NFL teams: {e}")
            raise

    def get_nfl_schedules(self, season):
        """Fetch NFL schedules for a specific season."""
        try:
            schedules = fetch_nfl_schedules(season)
            logger.info(f"Fetched NFL schedules for season {season}.")
            return schedules
        except Exception as e:
            logger.error(f"Error fetching NFL schedules: {e}")
            raise


    def get_nfl_box_scores(self, season, week, retries=3):
        """Fetch NFL box scores for a specific season and week with retry."""
        for attempt in range(retries):
            try:
                box_scores = fetch_nfl_box_scores(season, week)
                logger.info(f"Fetched NFL box scores for season {season}, week {week}.")
                return box_scores
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Error fetching NFL box scores: {e}", exc_info=True)
                    raise


    def get_live_box_scores_delta(self, minutes):
        """Fetch live box scores deltas for the past 'minutes'."""
        try:
            live_box_scores = fetch_live_box_scores_delta(minutes)
            logger.info(f"Fetched live box scores delta for the past {minutes} minutes.")
            return live_box_scores
        except Exception as e:
            logger.error(f"Error fetching live box scores delta: {e}")
            raise

# Add any additional API calls or logic if needed
