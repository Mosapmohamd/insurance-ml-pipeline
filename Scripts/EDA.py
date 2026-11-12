#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import argparse 
import os

def load_preprocessed_data(filepath="preprocessed_insurance.csv"):
    """
    Load preprocessed insurance dataset
    """
    df = pd.read_csv(filepath)
    print("Preprocessed data loaded successfully with shape:", df.shape)
    return df

def visualize_target_distribution(df):
    """
    Visualize the distribution of the target variable (charges)
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='charges', kde=True)
    plt.title("Distribution of Insurance Charges")
    plt.xlabel("Charges ($)")
    plt.savefig("plots/charges_distribution.png")
    plt.close()
    
    print("Note: Target variable is skewed and may require log transformation")

def visualize_age_distribution(df):
    """
    Visualize the distribution of age
    """
    plt.figure(figsize=(10, 6))
    plt.title("Age Distribution")
    sns.histplot(data=df, x="age", bins=23)
    plt.savefig("plots/age_distribution.png")
    plt.close()

def visualize_gender_distribution(df):
    """
    Visualize the distribution of gender
    """
    plt.figure(figsize=(8, 8))
    counts = df['sex'].value_counts()
    plt.title("Gender Distribution")
    plt.pie(x=counts.values, labels=counts.index, autopct="%1.2f%%", 
            shadow=True, normalize=True)
    plt.savefig("plots/gender_distribution.png")
    plt.close()

def visualize_bmi_categories(df):
    """
    Visualize the distribution of BMI categories
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x="bmi_category")
    plt.title("BMI Category Distribution")
    plt.xticks(rotation=45)
    plt.savefig("plots/bmi_categories.png")
    plt.close()
    
    # Alternative pie chart visualization
    plt.figure(figsize=(10, 10))
    count = df["bmi_category"].value_counts()
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0','#ffb3e6','#c4e17f']
    plt.pie(count.values, labels=count.index, autopct='%1.1f%%', 
            startangle=140, colors=colors, wedgeprops={'edgecolor': 'black'})
    plt.title("BMI Category Distribution", fontsize=14, fontweight='bold')
    plt.savefig("plots/bmi_categories_pie.png")
    plt.close()

def visualize_children_distribution(df):
    """
    Visualize the distribution of children
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df.children)
    plt.title("Number of Children Distribution")
    plt.savefig("plots/children_distribution.png")
    plt.close()

def visualize_smoker_distribution(df):
    """
    Visualize the distribution of smokers
    """
    plt.figure(figsize=(8, 8))
    count = df.smoker.value_counts()
    plt.pie(x=count.values, labels=count.index, colors=["#9b111e", "#008000"],
            autopct="%1.1f%%", shadow=True)
    plt.title("Smoker vs Non-Smoker Distribution")
    plt.savefig("plots/smoker_distribution.png")
    plt.close()

def analyze_gender_effect(df):
    """
    Analyze how gender affects charges
    """
    plt.figure(figsize=(8, 6))
    res = df.groupby(["sex"])['charges'].mean()
    sns.barplot(x=res.index, y=res.values)
    plt.title("Average Charges by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Average Charges ($)")
    plt.savefig("plots/gender_vs_charges.png")
    plt.close()

def analyze_bmi_effect(df):
    """
    Analyze how BMI affects charges
    """
    plt.figure(figsize=(12, 6))
    res = df.groupby(['bmi_category'])['charges'].mean()
    plt.title("Effect of BMI Category on Charges")
    sns.barplot(x=res.index, y=res.values)
    plt.xticks(rotation=20, ha='right')
    plt.xlabel("BMI Category")
    plt.ylabel("Average Charges ($)")
    plt.savefig("plots/bmi_vs_charges.png")
    plt.close()

def analyze_children_effect(df):
    """
    Analyze how number of children affects charges
    """
    plt.figure(figsize=(10, 6))
    res = df.groupby(['children'])['charges'].mean()
    plt.title("Effect of Number of Children on Charges")
    sns.barplot(x=res.index, y=res.values)
    plt.xlabel("Number of Children")
    plt.ylabel("Average Charges ($)")
    plt.savefig("plots/children_vs_charges.png")
    plt.close()

def analyze_region_effect(df):
    """
    Analyze how region affects charges
    """
    plt.figure(figsize=(10, 6))
    res = df.groupby(['region'])['charges'].mean()
    plt.title("Effect of Region on Charges")
    sns.barplot(x=res.index, y=res.values)
    plt.xlabel("Region")
    plt.ylabel("Average Charges ($)")
    plt.savefig("plots/region_vs_charges.png")
    plt.close()

def analyze_age_effect(df):
    """
    Analyze how age category affects charges
    """
    plt.figure(figsize=(10, 6))
    res = df.groupby("age_category")["charges"].agg(["mean", "count"]).reset_index()
    ax = sns.barplot(x="age_category", y="mean", data=res, palette="Blues")
    
    for i, row in res.iterrows():
        ax.text(i, row["mean"] + 200, f'n={row["count"]}', ha='center', fontsize=12, fontweight='bold')
    
    plt.xlabel("Age Category", fontsize=14)
    plt.ylabel("Average Charges ($)", fontsize=14)
    plt.title("Average Charges per Age Category (with Sample Size)", fontsize=16)
    plt.xticks(rotation=20)
    plt.savefig("plots/age_vs_charges.png")
    plt.close()

def analyze_correlations(df):
    """
    Analyze correlations between numeric features
    """
    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig("plots/correlation_heatmap.png")
    plt.close()
    
    print("Note: Correlation analysis shows no high correlations with target variable")

def main():    
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    parser = argparse.ArgumentParser(description="Load and process a dataset.")
    parser.add_argument("-f",'--file_path', type=str, help="Path to the CSV file")

    args = parser.parse_args()  

    if args.file_path :
        df = load_preprocessed_data(args.file_path) 
    else :
        df = load_preprocessed_data()
    
    # Visualize data distributions
    visualize_target_distribution(df)
    visualize_age_distribution(df)
    visualize_gender_distribution(df)
    visualize_bmi_categories(df)
    visualize_children_distribution(df)
    visualize_smoker_distribution(df)
    
    # Analyze effects on charges
    analyze_gender_effect(df)
    analyze_bmi_effect(df)
    analyze_children_effect(df)
    analyze_region_effect(df)
    analyze_age_effect(df)
    
    # Analyze correlations
    analyze_correlations(df)
    
    print("EDA completed successfully. Plots saved to 'plots' directory.")

if __name__ == "__main__":
    main()