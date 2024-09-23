import pandas as pd
from backend.utils.preprocessing import preprocess_data
from backend.models.train_models import load_models

class PredictionService:
    xgb_model = load_models()

    @staticmethod
    def make_prediction(data):
        df = pd.DataFrame([data])
        df, _, _ = preprocess_data(df)
        prediction = PredictionService.xgb_model.predict(df)
        targets = ['fantasy_points', 'touchdowns', 'yards', 'receptions', 'fumbles', 'interceptions', 'field_goal']
        return dict(zip(targets, prediction[0]))
