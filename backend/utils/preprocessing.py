import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def handle_missing_values(df):
    """
    Handles missing values in the DataFrame by filling or dropping them.
    """
    # Fill missing numerical values with the median
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        df[column].fillna(df[column].median(), inplace=True)
    
    # Fill missing categorical values with the mode
    for column in df.select_dtypes(include=['object']).columns:
        df[column].fillna(df[column].mode()[0], inplace=True)
    
    return df

def encode_categorical_variables(df):
    """
    Encodes categorical variables using Label Encoding or One-Hot Encoding.
    """
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    return df, label_encoders

def normalize_data(df):
    """
    Normalizes the data using StandardScaler.
    """
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df, scaler

def preprocess_data(df):
    """
    Preprocess the DataFrame by encoding categorical variables and scaling numerical features.
    The function also handles multi-target columns for regression.
    """
    # Assume targets are not to be encoded or scaled
    targets = ['fantasy_points', 'touchdowns', 'yards', 'receptions', 'fumbles', 'interceptions', 'field_goal']
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Identify categorical and numerical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # Remove targets from numerical columns if they are included
    numerical_columns = [col for col in numerical_columns if col not in targets]
    
    # Encode categorical columns
    df, label_encoders = encode_categorical_variables(df)
    
    # Normalize numerical columns
    df[numerical_columns], scaler = normalize_data(df[numerical_columns])
    
    return df, label_encoders, scaler
