import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, LabelBinarizer

def preprocess_data(df):
    """Preprocess the DataFrame for machine learning models."""
    # Handle missing values
    df.fillna(0, inplace=True)

    # Initialize label encoders, scaler, and label binarizers
    label_encoders = {}
    scaler = StandardScaler()
    binarizers = {}

    # Encode categorical features
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Scale numerical features
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Apply LabelBinarizer for target variables if necessary
    target_columns = [col for col in df.columns if 'target' in col]  # Adjust based on actual target column names
    for col in target_columns:
        lb = LabelBinarizer()
        df[col] = lb.fit_transform(df[col])
        binarizers[col] = lb

    return df, label_encoders, scaler, binarizers

def integrate_data(sleeper_data, espn_data):
    """Integrate Sleeper and ESPN data into a single DataFrame and preprocess it."""
    # Convert the data to DataFrames if not already
    sleeper_df = pd.DataFrame(sleeper_data)
    espn_df = pd.DataFrame(espn_data)
    
    # Ensure the merge key is consistent (e.g., player_id)
    if 'player_id' not in sleeper_df.columns or 'player_id' not in espn_df.columns:
        raise ValueError("Both datasets must contain the 'player_id' column for merging.")

    # Merge or join the data on the 'player_id' key
    integrated_df = pd.merge(sleeper_df, espn_df, on='player_id', how='inner')
    
    # Perform preprocessing on the integrated DataFrame
    integrated_df, label_encoders, scaler, binarizers = preprocess_data(integrated_df)
    
    return integrated_df, label_encoders, scaler, binarizers
