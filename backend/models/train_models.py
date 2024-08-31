import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from utils.preprocessing import preprocess_data

def train_xgboost_model(X, y):
    """Train an XGBoost model and return the trained model and accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost Model
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    xgb_model.fit(X_train, y_train)

    # Generate predictions and calculate accuracy
    xgb_preds = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_preds)
    print(f"XGBoost Model Accuracy: {xgb_accuracy:.2f}")

    return xgb_model, xgb_accuracy

def train_transfer_model(X, y):
    """Train a transfer learning model using ResNet50 and return the trained model and accuracy."""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top of the base model
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(len(np.unique(y)), activation='softmax')(x)

    # Define the model
    transfer_model = Model(inputs=base_model.input, outputs=predictions)
    transfer_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Assuming X contains preprocessed images
    X_train_images, X_test_images, y_train_images, y_test_images = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    transfer_model.fit(X_train_images, y_train_images, epochs=10, batch_size=32, validation_data=(X_test_images, y_test_images))

    # Evaluate the model
    transfer_accuracy = transfer_model.evaluate(X_test_images, y_test_images, verbose=0)
    print(f"Transfer Learning Model Accuracy: {transfer_accuracy[1]:.2f}")

    return transfer_model, transfer_accuracy

def train_ensemble_model(X, y):
    """Train an ensemble model using XGBoost and transfer learning, and return ensemble accuracy."""
    # Train XGBoost and transfer learning models
    xgb_model, xgb_accuracy = train_xgboost_model(X, y)
    transfer_model, transfer_accuracy = train_transfer_model(X, y)

    # Generate predictions
    X_train_images, X_test_images, y_train_images, y_test_images = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb_proba = xgb_model.predict_proba(X_test_images)
    transfer_proba = transfer_model.predict(X_test_images)

    # Average the probabilities
    ensemble_proba = (xgb_proba + transfer_proba) / 2
    ensemble_preds = np.argmax(ensemble_proba, axis=1)

    ensemble_accuracy = accuracy_score(y_test_images, ensemble_preds)
    print(f"Ensemble Model Accuracy: {ensemble_accuracy:.2f}")

    # Save the models
    transfer_model.save('backend/models/transfer_model.h5')
    xgb_model.save_model('backend/models/xgb_model.json')

    return ensemble_accuracy

# If you want to train all models when running the script directly
if __name__ == "__main__":
    X, y = preprocess_data()  # Ensure preprocess_data is correctly implemented

    # Train ensemble model
    ensemble_accuracy = train_ensemble_model(X, y)
    print(f"Final Ensemble Model Accuracy: {ensemble_accuracy:.2f}")
