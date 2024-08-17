import pandas as pd

def clean_data(data):
    # Example cleaning steps
    df = pd.DataFrame(data)
    df.fillna(0, inplace=True)  # Replace NaNs with 0
    df.drop_duplicates(inplace=True)  # Remove duplicates
    return df

def normalize_data(df):
    # Example normalization
    df = (df - df.mean()) / df.std()
    return df

# Example usage:
# cleaned_data = clean_data(api_response_data)
# normalized_data = normalize_data(cleaned_data)
