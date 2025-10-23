import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load and return the Titanic dataset"""
    df = pd.read_csv("titanic.csv")
    print(f"Dataset loaded with shape: {df.shape}")
    return df

def explore_data(df):
    """Perform basic exploratory data analysis"""
    print("\nData Info:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nMissing Values:\n", df.isnull().sum())

def plot_survival_stats(df):
    """Plot survival related visualizations"""
    # Overall survival
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x='Survived')
    plt.title('Survival Distribution')
    plt.show()
    
    # Survival by gender
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x='Sex', hue='Survived')
    plt.title('Survival by Gender')
    plt.show()

def plot_demographics(df):
    """Plot demographic visualizations"""
    # Age distribution
    plt.figure(figsize=(10,6))
    sns.histplot(df['Age'].dropna(), bins=30, kde=True)
    plt.title('Age Distribution')
    plt.show()
    
    # Class distribution
    plt.figure(figsize=(8,5))
    sns.countplot(data=df, x='Pclass')
    plt.title('Passenger Class Distribution')
    plt.show()

def correlation_analysis(df):
    """Generate and plot correlation heatmap"""
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

def detect_outliers(df):
    """Plot boxplots for numerical variables"""
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    sns.boxplot(x=df['Age'])
    plt.title('Age Distribution (Boxplot)')
    
    plt.subplot(1,2,2)
    sns.boxplot(x=df['Fare'])
    plt.title('Fare Distribution (Boxplot)')
    plt.show()

def main():
    """Main function to run the EDA"""
    # Load data
    df = load_data()
    
    # Run analyses
    explore_data(df)
    plot_survival_stats(df)
    plot_demographics(df)
    correlation_analysis(df)
    detect_outliers(df)
    
    print("\nEDA Completed Successfully!")

if __name__ == "__main__":
    main()