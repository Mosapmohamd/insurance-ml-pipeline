#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def load_data(filepath):
    """
    Load insurance dataset from CSV file
    """
    df = pd.read_csv(filepath)
    print("Data loaded successfully with shape:", df.shape)
    return df

def check_data(df):
    """
    Perform initial data checks
    """
    print("\n===== DATA INFO =====")
    df.info()
    
    print("\n===== DATA SUMMARY =====")
    print(df.describe())
    
    print("\n===== MISSING VALUES =====")
    print(df.isna().sum())

def remove_outliers(df, col_name):
    """
    Remove outliers using IQR method
    """
    print(f"Data shape before outliers removal: {df.shape}")
    
    # Calculate the upper and lower limits
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    
    # Create arrays of Boolean values indicating the outlier rows
    upper_array = np.where(df[col_name] >= upper)[0]
    lower_array = np.where(df[col_name] <= lower)[0]
    
    # Removing the outliers
    df.drop(index=upper_array, inplace=True)
    df.drop(index=lower_array, inplace=True)
    print(f"Data shape after outliers removal: {df.shape}")
    return df

def categorize_bmi(bmi):
    """
    Categorize BMI values into standard categories
    """
    if bmi < 16:
        return "Severely Underweight"
    elif 16 <= bmi <= 18.4:
        return "Underweight"
    elif 18.5 <= bmi <= 24.9:
        return "Normal Weight"
    elif 25 <= bmi <= 29.9:
        return "Overweight"
    elif 30 <= bmi <= 34.9:
        return "Obesity (Class 1)"
    elif 35 <= bmi <= 39.9:
        return "Obesity (Class 2)"
    else:
        return "Morbid Obesity (Class 3)"

def categorize_age(age):
    """
    Categorize age into groups
    """
    if 18 <= age <= 24:
        return "Young Adult"
    elif 25 <= age <= 44:
        return "Adult"
    else:
        return "Elder"

def preprocess_data(df):
    """
    Preprocess the insurance dataset
    """
    # Remove outliers from BMI
    df = remove_outliers(df, 'bmi')
    
    # Create BMI categories
    df["bmi_category"] = df["bmi"].apply(categorize_bmi)
    
    # Create age categories
    df["age_category"] = df["age"].apply(categorize_age)
    
    return df

def main():

    parser = argparse.ArgumentParser(description="Load and process a dataset.")
    parser.add_argument("-f",'--file_path', type=str, help="Path to the CSV file")

    args = parser.parse_args()  

    # Load and process data
    df = load_data(args.file_path)
    
    # Check data quality
    check_data(df)
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Save preprocessed data
    df.to_csv("preprocessed_insurance.csv", index=False)
    print("Preprocessed data saved to 'preprocessed_insurance.csv'")

if __name__ == "__main__":
    main()