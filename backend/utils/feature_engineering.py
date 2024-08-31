import pandas as pd
import numpy as np

def calculate_exponential_moving_average(data, span):
    """
    Calculate the Exponential Moving Average (EMA) for a given span.
    """
    return data.ewm(span=span, adjust=False).mean()

def calculate_rolling_statistics(data, window):
    """
    Calculate rolling statistics such as mean, standard deviation, and sum.
    """
    return {
        'rolling_mean': data.rolling(window=window).mean(),
        'rolling_std': data.rolling(window=window).std(),
        'rolling_sum': data.rolling(window=window).sum()
    }

def generate_lag_features(data, lags):
    """
    Generate lag features for time series data.
    """
    return {f'lag_{lag}': data.shift(lag) for lag in lags}

def apply_feature_engineering(data, feature_type='player'):
    if feature_type == 'player':
        data['ema_performance'] = calculate_exponential_moving_average(data['points'], span=10)
        rolling_stats = calculate_rolling_statistics(data['points'], window=5)
        for key, value in rolling_stats.items():
            data[key] = value

        lag_features = generate_lag_features(data['points'], lags=[1, 2, 3])
        for key, value in lag_features.items():
            data[key] = value

    elif feature_type == 'team':
        data['team_performance_trend'] = calculate_exponential_moving_average(data['team_points'], span=10)

    return data
