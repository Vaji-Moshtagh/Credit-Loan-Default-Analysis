import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def load_processed_data(file_path):
    """
    Load the preprocessed credit loan dataset.
    
    Args:
        file_path (str): Path to the preprocessed CSV file
        
    Returns:
        pandas.DataFrame: Processed dataset
    """
    data = pd.read_csv(file_path)
    print(f"Loaded processed data with {data.shape[0]} rows and {data.shape[1]} columns")
    return data

def build_single_feature_model(data, feature, target='loan_status'):
    """
    Build a logistic regression model with a single feature.
    
    Args:
        data (pandas.DataFrame): Processed dataset
        feature (str): Feature to use in the model
        target (str): Target variable
        
    Returns:
        sklearn.linear_model.LogisticRegression: Fitted model
    """
    # Prepare data
    X = data[[feature]]
    y = data[[target]]
    
    # Create and fit model
    model = LogisticRegression(solver='lbfgs')
    model.fit(X, np.ravel(y))
    
    # Display model info
    print(f"\nSingle feature model using {feature}:")
    print(f"Intercept: {model.intercept_[0]:.6f}")
    print(f"Coefficient: {model.coef_[0][0]:.6f}")
    
    return model

def build_multi_feature_model(data, features, target='loan_status'):
    """
    Build a logistic regression model with multiple features.
    
    Args:
        data (pandas.DataFrame): Processed dataset
        features (list): Features to use in the model
        target (str): Target variable
        
    Returns:
        sklearn.linear_model.LogisticRegression: Fitted model
    """
    # Prepare data
    X = data[features]
    y = data[[target]]
    
    # Create and fit model
    model = LogisticRegression(solver='lbfgs')
    model.fit(X, np.ravel(y))
    
    # Display model info
    print(f"\nMulti-feature model using {features}:")
    print(f"Intercept: {model.intercept_[0]:.6f}")
    print("Coefficients:")
    for feature, coef in zip(features, model.coef_[0]):
        print(f"  {feature}: {coef:.6f}")
    
    return model

def train_test_model(data, features, target='loan_status', test_size=0.4, random_state=123):
    """
    Train a model with train-test split for evaluation.
    
    Args:
        data (pandas.DataFrame): Processed dataset
        features (list): Features to use in the model
        target (str): Target variable
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (model, X_train, X_test, y_train, y_test)
    """
    # Prepare data
    X = data[features]
    y = data[[target]]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\nSplit data into {X_train.shape[0]} training and {X_test.shape[0]} testing samples")
    
    # Train model
    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, np.ravel(y_train))
    
    # Display model info
    print(f"Model trained with features: {features}")
    
    return model, X_train, X_test, y_train, y_test

def evaluate_predictions(model, X_test, y_test):
    """
    Evaluate model predictions and display results.
    
    Args:
        model (sklearn.linear_model.LogisticRegression): Trained model
        X_test (pandas.DataFrame): Test features
        y_test (pandas.DataFrame): Test target values
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Create comparison dataframe for first 5 samples
    results_df = pd.DataFrame({
        'Actual': np.ravel(y_test.head()),
        'Predicted Probability': y_pred_prob[:5]
    })
    
    print("\nSample predictions:")
    print(results_df)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def main():
    # Load processed data
    data = load_processed_data("processed_loan_data.csv")
    
    # Build single-feature model
    model_single = build_single_feature_model(data, 'loan_int_rate')
    
    # Build two-feature model
    model_two = build_multi_feature_model(data, ['loan_int_rate', 'person_emp_length'])
    
    # Build three-feature model
    model_three = build_multi_feature_model(data, 
                                           ['loan_int_rate', 'person_emp_length', 'person_income'])
    
    # Train-test model with evaluation
    features = ['loan_int_rate', 'person_emp_length', 'person_income']
    model, X_train, X_test, y_train, y_test = train_test_model(data, features)
    
    # Evaluate predictions
    evaluate_predictions(model, X_test, y_test)
    
    # Compare different feature sets
    features1 = ['person_income', 'person_emp_length', 'loan_amnt']
    features2 = ['person_income', 'loan_percent_income', 'cb_person_cred_hist_length']
    
    print("\nComparing different feature sets:")
    model1, _, _, _, _ = train_test_model(data, features1)
    model2, _, _, _, _ = train_test_model(data, features2)

if __name__ == "__main__":
    main()
