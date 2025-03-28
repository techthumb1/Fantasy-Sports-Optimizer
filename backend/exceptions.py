class SleeperAPIError(Exception):
    """Custom exception for Sleeper API related errors."""
    def __init__(self, message):
        super().__init__(message)
