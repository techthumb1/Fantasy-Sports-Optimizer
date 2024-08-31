import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.multioutput import MultiOutputRegressor
import numpy as np

def xgboost_model(X, y, problem_type="regression", save_model=True, model_dir='backend/models'):
    # Ensure directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Define the model
    if problem_type == "classification":
        model = xgb.XGBClassifier(objective="binary:logistic", n_estimators=100, learning_rate=0.001)
    elif problem_type == "regression":
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.001)
    else:
        raise ValueError("Invalid problem type. Choose either 'classification' or 'regression'.")

    # Handle multi-output regression or classification
    if y.ndim > 1 and y.shape[1] > 1:  # Multi-output scenario
        model = MultiOutputRegressor(model)
    
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model based on problem type
    if problem_type == "classification":
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.4f}')
    elif problem_type == "regression":
        if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
            mse = np.mean([mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(y_pred.shape[1])])
        else:
            mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error: {mse:.4f}')
    
    # Save the trained models
    if save_model:
        if isinstance(model, MultiOutputRegressor):
            for i, estimator in enumerate(model.estimators_):
                model_filename = os.path.join(model_dir, f"xgb_model_target_{i}.json")
                estimator.save_model(model_filename)
                print(f"Model for target {i} saved as {model_filename}")
        else:
            model_filename = os.path.join(model_dir, "xgb_model.json")
            model.save_model(model_filename)
            print(f"Model saved as {model_filename}")
    
    return model
