#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
import argparse

def load_preprocessed_data(filepath="preprocessed_insurance.csv"):
    """
    Load preprocessed insurance dataset
    """
    df = pd.read_csv(filepath)
    print("Preprocessed data loaded successfully with shape:", df.shape)
    return df

def encode_categorical_variables(df):
    """
    Convert categorical variables to numerical
    """
    # Convert sex to numerical (1 for male, 0 for female)
    df['sex'] = df['sex'].apply(lambda x: 1 if x == "male" else 0)
    
    # Convert smoker to numerical (1 for yes, 0 for no)
    df['smoker'] = df['smoker'].apply(lambda x: 1 if x == "yes" else 0)
    
    return df

def create_interaction_features(df):
    """
    Create interaction features to capture relationships
    """
    # Create age and BMI interaction
    df["age_bmi_interaction"] = df["age"] * df["bmi"]
    
    # Create smoker risk feature
    df["smoker_risk"] = df["smoker"] * df["bmi"]
    
    return df

def check_multicollinearity(df, features):
    """
    Check for multicollinearity among features using VIF
    """
    features_df = df[features]
    vif = pd.DataFrame()
    vif["Feature"] = features_df.columns
    vif["VIF Score"] = [variance_inflation_factor(features_df.values, i) for i in range(features_df.shape[1])]
    
    print("\n===== MULTICOLLINEARITY CHECK =====")
    print(vif)
    
    return vif

def remove_collinear_features(df):
    """
    Remove features with high multicollinearity
    """
    # Check multicollinearity first
    vif = check_multicollinearity(df, ["age", "bmi", "age_bmi_interaction"])
    
    # Based on VIF scores, drop the original features to prevent multicollinearity
    df.drop(['age', 'bmi'], axis=1, inplace=True)
    
    return df

def transform_target_variable(df):
    """
    Apply log transformation to skewed target variable
    """
    df['charges'] = np.log(df['charges'])
    print("Applied log transformation to charges")
    return df

def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    """
    from sklearn.model_selection import train_test_split
    
    X = df.drop(['charges'], axis=1)
    y = df['charges']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def scale_numerical_features(X_train, X_test):
    """
    Scale numerical features using StandardScaler
    """
    sc = StandardScaler()
    columns_to_scale = ['age_bmi_interaction', 'smoker_risk']
    
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    
    X_train_copy[columns_to_scale] = sc.fit_transform(X_train_copy[columns_to_scale].values)
    X_test_copy[columns_to_scale] = sc.transform(X_test_copy[columns_to_scale].values)
    
    return X_train_copy, X_test_copy, sc

def encode_categorical_features(X_train, X_test):
    """
    One-hot encode categorical features
    """
    # Identify categorical columns
    categorical_cols = X_train.select_dtypes(include='object').columns
    
    if len(categorical_cols) == 0:
        print("No categorical columns to encode")
        return X_train, X_test
    
    # Initialize one-hot encoder
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    
    # Apply one-hot encoding
    X_train_encoded = ohe.fit_transform(X_train[categorical_cols].values)
    X_test_encoded = ohe.transform(X_test[categorical_cols].values)
    
    # Convert to DataFrame with proper column names
    feature_names = ohe.get_feature_names_out(categorical_cols)
    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=feature_names, index=X_train.index)
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=feature_names, index=X_test.index)
    
    # Drop original categorical columns and join encoded columns
    X_train_result = X_train.drop(categorical_cols, axis=1).join(X_train_encoded_df)
    X_test_result = X_test.drop(categorical_cols, axis=1).join(X_test_encoded_df)
    
    return X_train_result, X_test_result, ohe

def prepare_classification_target(y_train, y_test):
    """
    Prepare target variable for classification by binarizing based on median
    """
    threshold = y_train.median()
    print(f"Classification threshold (median of log charges): {threshold}")
    
    y_class_train = (y_train > threshold).astype(int)
    y_class_test = (y_test > threshold).astype(int)
    
    print(f"Class distribution in training set: {y_class_train.value_counts().to_dict()}")
    print(f"Class distribution in test set: {y_class_test.value_counts().to_dict()}")
    
    return y_class_train, y_class_test, threshold

def main():

    parser = argparse.ArgumentParser(description="Load and process a dataset.")

    parser.add_argument("-f",'--file_path', type=str, help="Path to the CSV file")

    args = parser.parse_args()  

    if args.file_path :

        df = load_preprocessed_data(args.file_path) 

    else :

        df = load_preprocessed_data()

    
    # Encode categorical variables (sex, smoker)
    df = encode_categorical_variables(df)
    
    # Create interaction features
    df = create_interaction_features(df)
    
    # Check and remove collinear features
    df = remove_collinear_features(df)
    
    # Transform target variable
    df = transform_target_variable(df)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Scale numerical features
    X_train_scaled, X_test_scaled, scaler = scale_numerical_features(X_train, X_test)
    
    # Encode remaining categorical features
    X_train_processed, X_test_processed, encoder = encode_categorical_features(X_train_scaled, X_test_scaled)
    
    # Prepare classification target
    y_class_train, y_class_test, threshold = prepare_classification_target(y_train, y_test)
    
    # Save processed data for modeling
    X_train_processed.to_csv("processed_X_train.csv", index=False)
    X_test_processed.to_csv("processed_X_test.csv", index=False)
    y_train.to_csv("processed_y_train.csv", index=False)
    y_test.to_csv("processed_y_test.csv", index=False)
    y_class_train.to_csv("processed_y_class_train.csv", index=False)
    y_class_test.to_csv("processed_y_class_test.csv", index=False)
    
    print("Feature engineering completed. Processed data saved for modeling.")

if __name__ == "__main__":
    main()