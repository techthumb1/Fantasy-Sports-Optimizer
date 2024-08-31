import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

def train_model(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Select the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    
    # Save the trained model
    os.makedirs('backend/models', exist_ok=True)
    joblib.dump(model, 'backend/models/sports_optimizer_model.pkl')
    print("Random Forest model saved as 'backend/models/sports_optimizer_model.pkl'")
    
    return model, mse
