import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

def train_xgboost_model(X, y, problem_type="classification"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if problem_type == "classification":
        model = xgb.XGBClassifier(objective="binary:logistic", n_estimators=100, learning_rate=0.1)
    else:
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if problem_type == "classification":
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.4f}')
    else:
        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error: {mse:.4f}')

    return model
