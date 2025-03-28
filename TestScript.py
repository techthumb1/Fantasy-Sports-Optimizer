from backend.models.train_models import load_data_from_database

# Test script
if __name__ == '__main__':
    df = load_data_from_database()
    print(df.head())
