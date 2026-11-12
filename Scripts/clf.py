#!/usr/bin/env python
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from scipy.stats import randint, uniform
import os 

def load_processed_data():
    """
    Load the processed data for classification modeling
    """
    X_train = pd.read_csv("processed_X_train.csv")
    X_test = pd.read_csv("processed_X_test.csv")
    y_train = pd.read_csv("processed_y_class_train.csv").squeeze()
    y_test = pd.read_csv("processed_y_class_test.csv").squeeze()
    
    print("Loaded processed data for classification:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Class distribution in training: {dict(y_train.value_counts())}")
    
    return X_train, X_test, y_train, y_test

def define_classification_models():
    """
    Define classification models and their parameter distributions for tuning
    """
    # Define the pipeline
    pipeline = Pipeline([('clf', RandomForestClassifier(random_state=42))])

    # Define parameter distributions for different models
    parameters = [
        {
            'clf': [RandomForestClassifier(random_state=42)],
            'clf__n_estimators': randint(10, 500),
            'clf__max_depth': [5, 10, 15, 20, 25, 30, None],
            'clf__max_features': ['sqrt', 'log2', None],
            'clf__min_samples_split': randint(2, 20),
            'clf__min_samples_leaf': randint(1, 10),
            'clf__bootstrap': [True, False]
        },
        {
            'clf': [LogisticRegression(random_state=42, max_iter=500)],
            'clf__C': uniform(0.001, 10.0),
            'clf__solver': ['liblinear', 'lbfgs', 'saga'],
            'clf__fit_intercept': [True, False]
        },
        {
            'clf': [GradientBoostingClassifier(random_state=42)],
            'clf__learning_rate': uniform(0.001, 0.5),
            'clf__n_estimators': randint(50, 500),
            'clf__subsample': uniform(0.1, 0.9),
            'clf__max_depth': randint(3, 15),
            'clf__min_samples_split': randint(2, 20),
            'clf__min_samples_leaf': randint(1, 10),
            'clf__max_features': ['sqrt', 'log2', None]
        },
        {
            'clf': [DecisionTreeClassifier(random_state=42)],
            'clf__max_depth': [5, 10, 15, 20, 25, 30, None],
            'clf__min_samples_split': randint(2, 20),
            'clf__min_samples_leaf': randint(1, 10),
            'clf__max_features': ['sqrt', 'log2', None],
            'clf__criterion': ['gini', 'entropy', 'log_loss']
        },
        {
            'clf': [XGBClassifier(random_state=42, eval_metric='logloss')],
            'clf__n_estimators': randint(50, 500),
            'clf__max_depth': randint(3, 40),
            'clf__learning_rate': uniform(0.001, 0.5),
            'clf__subsample': uniform(0.1, 0.9),
            'clf__colsample_bytree': uniform(0.3, 0.9),
            'clf__colsample_bylevel': uniform(0.3, 0.7),
            'clf__gamma': uniform(0, 5),
            'clf__min_child_weight': randint(1, 10),
            'clf__reg_alpha': uniform(0, 10),
            'clf__reg_lambda': uniform(0, 10)
        },
        {
            'clf': [LGBMClassifier(random_state=42)],
            'clf__n_estimators': randint(50, 500),
            'clf__learning_rate': uniform(0.001, 0.5),
            'clf__subsample': uniform(0.1, 0.9),
            'clf__colsample_bytree': uniform(0.3, 0.9),
            'clf__min_child_samples': randint(1, 50),
            'clf__reg_alpha': uniform(0, 10),
            'clf__reg_lambda': uniform(0, 10),
            'clf__num_leaves': randint(20, 150),
            'clf__boosting_type': ['gbdt', 'dart', 'goss'],
            'clf__bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0]
        },
        {
            'clf': [AdaBoostClassifier(random_state=42)],
            'clf__n_estimators': randint(50, 500),
            'clf__learning_rate': uniform(0.001, 1.0)
        }
    ]
    
    return pipeline, parameters

def train_classification_model(X_train, y_train, pipeline, parameters, n_iter=100):
    """
    Train and tune classification models using RandomizedSearchCV
    """
    print("Starting classification model training with randomized search...")
    
    random_search = RandomizedSearchCV(
        pipeline, 
        parameters,
        n_iter=n_iter,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    # Fit the model
    random_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    print(f"Best model: {best_model}")
    print(f"Best parameters: {best_params}")
    
    return best_model, random_search

def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluate the classification model on the test set
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # Print results
    print("\nClassification Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred
    }

def save_classification_model(model, filename='classification_model.pkl'):
    """
    Save the trained classification model
    """
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model, os.path.join('models',filename))
    print(f"Model saved to models/{filename}")

def main():
    """
    Main function to execute the classification modeling pipeline
    """
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Define models
    pipeline, parameters = define_classification_models()
    
    # Train and tune the model
    best_model, random_search = train_classification_model(X_train, y_train, pipeline, parameters)
    
    # Evaluate the model
    evaluation_results = evaluate_classification_model(best_model, X_test, y_test)
    
    # Save the model
    save_classification_model(best_model)
    
    return best_model, evaluation_results, random_search

if __name__ == "__main__":
    best_model, evaluation_results, random_search = main()
    
    # Extract best model details for further analysis
    best_model_name = type(best_model.named_steps['clf']).__name__
    print(f"\nBest model type: {best_model_name}")
    
    # If desired, save more detailed results
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df.to_csv('classification_model_search_results.csv', index=False)
    
    print("Classification modeling pipeline completed successfully!")