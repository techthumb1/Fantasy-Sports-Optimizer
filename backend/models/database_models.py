from sqlalchemy import Column, Integer, String, ForeignKey
from backend.models.database import db

class Team(db.Model):
    __tablename__ = 'teams'
    TeamID = db.Column(db.Integer, primary_key=True)
    Name = db.Column(db.String(100))
    # Add other team fields as needed

class Game(db.Model):
    __tablename__ = 'games'
    GameID = db.Column(db.Integer, primary_key=True)
    Season = db.Column(db.Integer)
    Week = db.Column(db.Integer)
    HomeTeam = db.Column(db.String(10))
    AwayTeam = db.Column(db.String(10))
    Status = db.Column(db.String(50))
    # Add other game fields as needed

class PlayerStat(db.Model):
    __tablename__ = 'player_stats'
    PlayerStatID = db.Column(db.Integer, primary_key=True)
    PlayerID = db.Column(db.Integer)
    GameID = db.Column(db.Integer, db.ForeignKey('games.GameID'))
    TeamID = db.Column(db.Integer, db.ForeignKey('teams.TeamID'))
    FantasyPoints = db.Column(db.Float)
    Touchdowns = db.Column(db.Integer)
    Yards = db.Column(db.Integer)
    Receptions = db.Column(db.Integer)
    Fumbles = db.Column(db.Integer)
    Interceptions = db.Column(db.Integer)
    FieldGoals = db.Column(db.Integer)
    RushingAttempts = db.Column(db.Integer)
    RushingYards = db.Column(db.Integer)
    RushingTouchdowns = db.Column(db.Integer)
    ReceivingTargets = db.Column(db.Integer)
    Receptions = db.Column(db.Integer)
    ReceivingYards = db.Column(db.Integer)
    ReceivingTouchdowns = db.Column(db.Integer)
    PassingAttempts = db.Column(db.Integer)
    PassingCompletions = db.Column(db.Integer)
    PassingInterceptions = db.Column(db.Integer)
    PassingYards = db.Column(db.Integer)
    PassingTouchdowns = db.Column(db.Integer)
    FieldGoalsAttempted = db.Column(db.Integer)
    FieldGoalsMade = db.Column(db.Integer)
    FieldGoalsMissed = db.Column(db.Integer)
    ExtraPointAttempts = db.Column(db.Integer)
    ExtraPointsMade = db.Column(db.Integer)
    ExtraPointsMissed = db.Column(db.Integer)
    PassingSacks = db.Column(db.Integer)
    LongestRush = db.Column(db.Integer)
    LongestReception = db.Column(db.Integer)
    LongestPass = db.Column(db.Integer)
    FantasyPointsHalfPPR = db.Column(db.Float)

PlayerStat.Game = db.relationship('Game', backref='player_stats')
PlayerStat.Team = db.relationship('Team', backref='player_stats')

# Define Player model
class Player(db.Model):
    __tablename__ = 'players'
    player_id = db.Column(db.String, primary_key=True)  # Change to match your PlayerModel
    first_name = db.Column(db.String(100))
    last_name = db.Column(db.String(100))
    position = db.Column(db.String(50))
    team_abbr = db.Column(db.String(10))

    def to_dict(self):
        return {
            'player_id': self.player_id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'position': self.position,
            'team_abbr': self.team_abbr,
        }

class GameStats(db.Model):
    __tablename__ = 'game_stats'
    id = db.Column(db.Integer, primary_key=True)
    player_id = db.Column(db.String, db.ForeignKey('players.player_id'), nullable=False)  # ForeignKey references players.player_id
    game_id = db.Column(db.String, nullable=False)
    stats = db.Column(db.JSON)

    # Define relationship with Player
    player = db.relationship('Player', backref='game_stats')

    def to_dict(self):
        return {
            'id': self.id,
            'player_id': self.player_id,
            'game_id': self.game_id,
            'stats': self.stats
        }

class BettingProp(db.Model):
    __tablename__ = 'betting_props'
    id = db.Column(db.Integer, primary_key=True)
    GameID = db.Column(db.Integer)
    #PlayerID = Column(Integer, ForeignKey('players.PlayerID'), nullable=False)
    BetType = db.Column(db.String(100))
    MarketType = db.Column(db.String(50))
    Sportsbook = db.Column(db.String(50))
    Odds = db.Column(db.Float)
    WinProbability = db.Column(db.Float)
    ExpectedValue = db.Column(db.Float)
    Timestamp = db.Column(db.DateTime)

# Define BettingProps model
class BettingProps(db.Model):
    __tablename__ = 'betting_props'
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    player_id = db.Column(db.String, db.ForeignKey('players.player_id'), nullable=False)  # Corrected ForeignKey to match 'player_id'
    prop_type = db.Column(db.String(100), nullable=False)
    value = db.Column(db.Float, nullable=False)

    # Define relationship between BettingProps and Player
    player = db.relationship('Player', backref='betting_props')

    def to_dict(self):
        return {
            'id': self.id,
            'player_id': self.player_id,
            'prop_type': self.prop_type,
            'value': self.value
        }
