import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import logging


from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier 


# Set up logging if not already configured
logger = logging.getLogger(__name__)

def extract_features_and_targets(df):
    # Drop rows with any missing values
    df.dropna(inplace=True)

    # Define feature and target columns
    feature_columns = [
        'RushingAttempts', 'RushingYards', 'RushingTouchdowns',
        'ReceivingTargets', 'Receptions', 'ReceivingYards', 'ReceivingTouchdowns',
        'PassingAttempts', 'PassingCompletions', 'PassingInterceptions',
        'PassingYards', 'PassingTouchdowns',
        'FieldGoalsAttempted', 'FieldGoalsMade', 'FieldGoalsMissed',
        'ExtraPointAttempts', 'ExtraPointsMade', 'ExtraPointsMissed',
        'PassingSacks', 'LongestRush', 'LongestReception', 'LongestPass',
        'team_id', 'abbreviation', 'location', 'homeAway', 'game_id',
        'score', 'date'  # Include 'date' for processing
    ]
    target_columns = ['FantasyPointsHalfPPR']

    # Ensure that all feature columns are present
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise KeyError(f"Missing feature columns: {missing_features}")

    # Ensure that at least one target column is present
    available_target_columns = [col for col in target_columns if col in df.columns]
    missing_targets = [col for col in target_columns if col not in df.columns]
    if missing_targets:
        logger.warning(f"Missing target columns: {missing_targets}")
        if not available_target_columns:
            raise KeyError("No target columns are available in the data.")

    # Extract features and targets
    X = df[feature_columns].copy()
    y = df[available_target_columns].copy()

    # Process 'date' column into datetime and extract components
    X['date'] = pd.to_datetime(X['date'])
    X['year'] = X['date'].dt.year
    X['month'] = X['date'].dt.month
    X['day'] = X['date'].dt.day
    X['day_of_week'] = X['date'].dt.dayofweek
    X.drop('date', axis=1, inplace=True)

    # Define categorical and numerical features
    categorical_features = ['team_id', 'abbreviation', 'location', 'homeAway', 'game_id']
    numerical_features = [
        'RushingAttempts', 'RushingYards', 'RushingTouchdowns',
        'ReceivingTargets', 'Receptions', 'ReceivingYards', 'ReceivingTouchdowns',
        'PassingAttempts', 'PassingCompletions', 'PassingInterceptions',
        'PassingYards', 'PassingTouchdowns',
        'FieldGoalsAttempted', 'FieldGoalsMade', 'FieldGoalsMissed',
        'ExtraPointAttempts', 'ExtraPointsMade', 'ExtraPointsMissed',
        'PassingSacks', 'LongestRush', 'LongestReception', 'LongestPass',
        'score', 'year', 'month', 'day', 'day_of_week'
    ]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Fit and transform the features
    X_processed = pipeline.fit_transform(X)
    y_processed = y.values.ravel()  # Flatten y to a 1D array if necessary

    return X_processed, y_processed



# Define file paths
XGB_MODEL_PATH = 'backend/models/xgb_model_{}.joblib'
TRANSFER_MODEL_PATH = 'backend/models/transfer_model.h5'

# Train the XGBoost model
def train_xgboost_model(X, y):
    """Train and return a MultiOutput XGBoost model for multi-output classification."""
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the XGBoost model with MultiOutputClassifier
    xgb_base_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    multi_output_model = MultiOutputClassifier(xgb_base_model, n_jobs=-1)
    multi_output_model.fit(X_train, y_train)

    # Predict
    y_pred = multi_output_model.predict(X_test)

    # Evaluate: Accuracy can be calculated for each output separately
    for i in range(y.shape[1]):
        accuracy = accuracy_score(y_test[:, i], y_pred[:, i])
        print(f"Output {i} Accuracy: {accuracy:.2f}")

    return multi_output_model

# Train a dense neural network model for tabular data
def train_dense_nn_model(X, y):
    """Train and return a dense neural network model for multi-output classification."""
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define a simple feedforward neural network
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(y_train.shape[1], activation='softmax'))  # Adjust output layer for multi-output classification

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    transfer_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Dense Neural Network Model Accuracy: {transfer_accuracy[1]:.2f}")

    return model

# Save and load models
def save_models(xgb_model, nn_model):
    """Save the XGBoost and Dense Neural Network models."""
    # Save each individual XGBoost model
    for idx, estimator in enumerate(xgb_model.estimators_):
        joblib.dump(estimator, XGB_MODEL_PATH.format(idx))
        print(f"XGBoost model for target {idx} saved as {XGB_MODEL_PATH.format(idx)}")
    
    # Save the dense neural network model
    nn_model.save(TRANSFER_MODEL_PATH)
    print(f"Dense Neural Network model saved as {TRANSFER_MODEL_PATH}")

def load_models(num_targets):
    """Load the XGBoost and Dense Neural Network models."""
    # Load XGBoost models
    estimators = []
    for idx in range(num_targets):
        estimator = joblib.load(XGB_MODEL_PATH.format(idx))
        estimators.append(estimator)
    
    xgb_model = MultiOutputClassifier(XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3))
    xgb_model.estimators_ = estimators
    
    # Load dense neural network model
    nn_model = tf.keras.models.load_model(TRANSFER_MODEL_PATH)
    
    return xgb_model, nn_model

# Train the ensemble model
def train_ensemble_model(X, y, xgb_model):
    """Train and return an ensemble model combining XGBoost predictions."""
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get predictions from XGBoost model
    xgb_proba = np.array(xgb_model.predict_proba(X_test))  # This will be a list of arrays for each output

    # Reshape xgb_proba to be consistent with y_test
    if len(xgb_proba.shape) == 3:
        xgb_proba = np.mean(xgb_proba, axis=2)  # Average across the classes if multilabel

    # Get final predictions
    if len(xgb_proba.shape) == 2:  # Multilabel indicator
        ensemble_preds = (xgb_proba > 0.5).astype(int)
    else:
        ensemble_preds = np.argmax(xgb_proba, axis=-1)

    # Ensure y_test is in the correct shape
    if len(y_test.shape) == 1:
        y_test = np.expand_dims(y_test, axis=-1)

    # Evaluate ensemble predictions
    ensemble_accuracy = accuracy_score(y_test, ensemble_preds)
    print(f"Ensemble Model Accuracy: {ensemble_accuracy:.2f}")

    return ensemble_accuracy



def retrain_models():
    df = load_data_from_database()
    X, y = extract_features_and_targets(df)
    xgb_model = train_xgboost_model(X, y)
    nn_model = train_dense_nn_model(X, y)
    save_models(xgb_model, nn_model)
    logger.info("Models retrained with updated parameters.")


# Main execution to train models
if __name__ == "__main__":
    X, y = preprocess_data() 

    # Train individual models
    xgb_model = train_xgboost_model(X, y)
    nn_model = train_dense_nn_model(X, y)

    # Train ensemble model
    ensemble_accuracy = train_ensemble_model(X, y, xgb_model, nn_model)
    print(f"Final Ensemble Model Accuracy: {ensemble_accuracy:.2f}")
