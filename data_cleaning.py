import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Load the credit loan dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    # Load the dataset
    cr_loan = pd.read_csv(file_path)
    print(f"Dataset loaded with {cr_loan.shape[0]} rows and {cr_loan.shape[1]} columns")
    return cr_loan

def check_missing_values(data):
    """
    Check and display columns with missing values.
    
    Args:
        data (pandas.DataFrame): The dataset to check
        
    Returns:
        list: Columns with missing values
    """
    # Get columns with missing values
    cols_with_nulls = data.columns[data.isnull().any()]
    print("Columns with missing values:")
    print(cols_with_nulls)
    
    # For each column with missing values, print count
    for col in cols_with_nulls:
        print(f"{col}: {data[col].isnull().sum()} missing values")
    
    return cols_with_nulls

def handle_missing_employment_length(data):
    """
    Impute missing values in employment length with median.
    
    Args:
        data (pandas.DataFrame): Dataset with missing employment length
        
    Returns:
        pandas.DataFrame: Dataset with imputed employment length
    """
    # Check examples of rows with missing employment length
    print("Examples of rows with missing employment length:")
    print(data[data['person_emp_length'].isnull()].head())
    
    # Impute missing values with median
    median_emp_length = data['person_emp_length'].median()
    data['person_emp_length'].fillna(median_emp_length, inplace=True)
    print(f"Filled missing employment length values with median: {median_emp_length}")
    
    return data

def visualize_employment_length(data):
    """
    Create a histogram of employment length distribution.
    
    Args:
        data (pandas.DataFrame): Dataset with employment length
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data['person_emp_length'], bins='auto', color='blue')
    plt.xlabel("Person Employment Length (Years)")
    plt.ylabel("Count")
    plt.title("Distribution of Employment Length")
    plt.grid(alpha=0.3)
    plt.savefig("employment_length_distribution.png")
    plt.show()

def handle_missing_interest_rate(data):
    """
    Remove rows with missing interest rate values.
    
    Args:
        data (pandas.DataFrame): Dataset with missing interest rates
        
    Returns:
        pandas.DataFrame: Clean dataset without missing interest rates
    """
    # Count missing interest rate values
    missing_count = data['loan_int_rate'].isnull().sum()
    print(f"Number of rows with missing interest rate: {missing_count}")
    
    # Get indices of rows with missing interest rates
    indices_to_drop = data[data['loan_int_rate'].isnull()].index
    
    # Create clean dataset by dropping those rows
    data_clean = data.drop(indices_to_drop)
    print(f"Removed {len(indices_to_drop)} rows with missing interest rates")
    print(f"Clean dataset has {data_clean.shape[0]} rows")
    
    return data_clean

def one_hot_encode_categorical(data):
    """
    Apply one-hot encoding to categorical variables.
    
    Args:
        data (pandas.DataFrame): Dataset with categorical variables
        
    Returns:
        pandas.DataFrame: Dataset with one-hot encoded variables
    """
    # Separate numeric and categorical columns
    numeric_data = data.select_dtypes(exclude=['object'])
    categorical_data = data.select_dtypes(include=['object'])
    
    print(f"Categorical columns to encode: {categorical_data.columns.tolist()}")
    
    # Apply one-hot encoding
    categorical_encoded = pd.get_dummies(categorical_data)
    
    # Combine numeric and encoded categorical data
    processed_data = pd.concat([numeric_data, categorical_encoded], axis=1)
    
    print(f"Dataset after one-hot encoding has {processed_data.shape[1]} columns")
    return processed_data

def main():
    # Load data
    cr_loan = load_data("credit_loan_data.csv")
    
    # Check for missing values
    check_missing_values(cr_loan)
    
    # Handle missing employment length
    cr_loan = handle_missing_employment_length(cr_loan)
    
    # Visualize employment length distribution
    visualize_employment_length(cr_loan)
    
    # Handle missing interest rates
    cr_loan_clean = handle_missing_interest_rate(cr_loan)
    
    # One-hot encode categorical variables
    cr_loan_prep = one_hot_encode_categorical(cr_loan_clean)
    
    # Save the cleaned data
    cr_loan_prep.to_csv("processed_loan_data.csv", index=False)
    print("Processed data saved to 'processed_loan_data.csv'")

if __name__ == "__main__":
    main()
