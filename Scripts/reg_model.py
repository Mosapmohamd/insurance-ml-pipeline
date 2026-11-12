#!/usr/bin/env python
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint, uniform
import os

def load_processed_data():
    """
    Load the processed data for modeling
    """
    X_train = pd.read_csv("processed_X_train.csv")
    X_test = pd.read_csv("processed_X_test.csv")
    y_train = pd.read_csv("processed_y_train.csv").squeeze()
    y_test = pd.read_csv("processed_y_test.csv").squeeze()
    
    print("Loaded processed data:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def define_regression_models():
    """
    Define regression models and their parameter distributions for tuning
    """
    # Define the pipeline
    pipeline = Pipeline([('reg', RandomForestRegressor(random_state=42))])

    # Define parameter distributions for different models
    parameters = [
        {
            'reg': [RandomForestRegressor(random_state=42)],
            'reg__n_estimators': randint(10, 500),
            'reg__max_depth': [5, 10, 15, 20, 25, 30, None],
            'reg__max_features': ['sqrt', 'log2', None],
            'reg__min_samples_split': randint(2, 20),
            'reg__min_samples_leaf': randint(1, 10),
            'reg__bootstrap': [True, False]
        },
        {
            'reg': [Ridge(random_state=42)],
            'reg__alpha': uniform(0.001, 10.0),
            'reg__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            'reg__fit_intercept': [True, False],
        },
        {
            'reg': [Lasso(random_state=42)],
            'reg__alpha': uniform(0.001, 10.0),
            'reg__fit_intercept': [True, False],
            'reg__selection': ['cyclic', 'random']
        },
        {
            'reg': [ElasticNet(random_state=42)],
            'reg__alpha': uniform(0.001, 10.0),
            'reg__l1_ratio': uniform(0.1, 1.0),
            'reg__fit_intercept': [True, False],
            'reg__selection': ['cyclic', 'random']
        },
        {
            'reg': [GradientBoostingRegressor(random_state=42)],
            'reg__learning_rate': uniform(0.001, 0.5),
            'reg__n_estimators': randint(50, 500),
            'reg__subsample': uniform(0.5, 0.9),  
            'reg__max_depth': randint(3, 15),
            'reg__min_samples_split': randint(2, 20),
            'reg__min_samples_leaf': randint(1, 10),
            'reg__max_features': ['sqrt', 'log2', None],
            'reg__loss': ['squared_error', 'absolute_error', 'huber', 'quantile']
        },
        {
            'reg': [DecisionTreeRegressor(random_state=42)],
            'reg__max_depth': [5, 10, 15, 20, 25, 30, None],
            'reg__min_samples_split': randint(2, 20),
            'reg__min_samples_leaf': randint(1, 10),
            'reg__max_features': ['sqrt', 'log2', None],
            'reg__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
        },
        {
            'reg': [XGBRegressor(random_state=42, objective='reg:squarederror')],
            'reg__n_estimators': randint(50, 500),
            'reg__max_depth': randint(3, 40),
            'reg__learning_rate': uniform(0.001, 0.5),
            'reg__subsample': uniform(0.5, 1.0),  
            'reg__colsample_bylevel': uniform(0.3, 0.7),
            'reg__gamma': uniform(0, 5),
            'reg__min_child_weight': randint(1, 10),
            'reg__reg_alpha': uniform(0, 10),
            'reg__reg_lambda': uniform(0, 10),
            'reg__colsample_bytree': uniform(0.3, 1.0),
        },
        {
            'reg': [LGBMRegressor(random_state=42)],
            'reg__n_estimators': randint(50, 500),
            'reg__learning_rate': uniform(0.001, 0.5),
            'reg__colsample_bytree': uniform(0.3, 1.0),
            'reg__min_child_samples': randint(1, 50),
            'reg__reg_alpha': uniform(0, 10),
            'reg__reg_lambda': uniform(0, 10),
            'reg__num_leaves': randint(20, 100),
            'reg__boosting_type': ['gbdt', 'dart'],
            'reg__bagging_fraction': uniform(0.5, 1.0),
        },
        {
            'reg': [AdaBoostRegressor(random_state=42)],
            'reg__n_estimators': randint(50, 500),
            'reg__learning_rate': uniform(0.001, 1.0),
            'reg__loss': ['linear', 'square', 'exponential']
        }
    ]
    
    return pipeline, parameters

def train_regression_model(X_train, y_train, pipeline, parameters, n_iter=100):
    """
    Train and tune regression models using RandomizedSearchCV
    """
    print("Starting regression model training with randomized search...")
    
    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        pipeline, 
        parameters, 
        cv=5, 
        n_iter=n_iter, 
        n_jobs=-1, 
        verbose=1, 
        random_state=42,
        scoring='neg_mean_squared_error'
    )
    
    # Run the search
    random_search.fit(X_train, y_train)
    
    # Best model & parameters
    print("\n===== BEST REGRESSION MODEL =====")
    print("Best Model:", random_search.best_estimator_)
    print("Best Parameters:", random_search.best_params_)
    print("Best CV Score:", -random_search.best_score_)  # Convert back from negative MSE
    
    return random_search

def evaluate_regression_model(model, X_test, y_test):
    """
    Evaluate the best regression model on the test set
    """
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print("\n===== MODEL EVALUATION =====")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    
    # Since we're working with log-transformed target, convert back to original scale
    print("\nNote: These metrics are on log-transformed charges. For interpretation,")
    print("you may want to transform predictions back to original scale using np.exp().")
    
    return {
        'r2': r2,
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }

def save_model(model, filename='best_regression_model.pkl'):
    """
    Save the trained model to a file
    """
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model, os.path.join('models',filename))
    print(f"Model saved to {filename}")

def main():
    # Load processed data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Define models and parameters
    pipeline, parameters = define_regression_models()
    
    # Train model
    best_model = train_regression_model(X_train, y_train, pipeline, parameters, n_iter=100)
    
    # Evaluate model
    metrics = evaluate_regression_model(best_model, X_test, y_test)
    
    # Save model
    save_model(best_model.best_estimator_, 'best_regression_model.pkl')
    
    print("\nRegression model training and evaluation completed.")

if __name__ == "__main__":
    main()