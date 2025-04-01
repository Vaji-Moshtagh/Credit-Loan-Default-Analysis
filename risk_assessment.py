# risk_assessment.py
# This module calculates risk metrics based on model predictions



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def calculate_expected_loss(probabilities, loan_amounts, lgd=0.2):
    """
    Calculate expected loss for a portfolio of loans
    
    Parameters:
    -----------
    probabilities : array-like
        Probability of default for each loan
    loan_amounts : array-like
        Principal amount of each loan
    lgd : float, default=0.2
        Loss Given Default - the percentage of the loan amount lost when a default occurs
        
    Returns:
    --------
    expected_loss : float
        Total expected loss amount
    expected_loss_percentage : float
        Expected loss as a percentage of total portfolio
    loan_risk_df : pandas DataFrame
        DataFrame with risk calculations for each loan
    """
    # Input validation
    if len(probabilities) != len(loan_amounts):
        raise ValueError("Probabilities and loan amounts must have the same length")
    
    # Create a DataFrame to hold our calculations
    loan_risk_df = pd.DataFrame({
        'probability_of_default': probabilities,
        'loan_amount': loan_amounts
    })
    
    # Calculate expected loss for each loan
    loan_risk_df['expected_loss'] = loan_risk_df['probability_of_default'] * loan_risk_df['loan_amount'] * lgd
    
    # Calculate risk metrics
    total_exposure = loan_risk_df['loan_amount'].sum()
    total_expected_loss = loan_risk_df['expected_loss'].sum()
    expected_loss_percentage = (total_expected_loss / total_exposure) * 100
    
    # Add risk categories
    loan_risk_df['risk_category'] = pd.cut(
        loan_risk_df['probability_of_default'], 
        bins=[0, 0.05, 0.15, 0.3, 1.0], 
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    return total_expected_loss, expected_loss_percentage, loan_risk_df

def plot_risk_distribution(loan_risk_df):
    """
    Create visualizations of the loan risk distribution
    
    Parameters:
    -----------
    loan_risk_df : pandas DataFrame
        DataFrame with risk calculations from calculate_expected_loss function
    
    Returns:
    --------
    None (displays plots)
    """
    plt.figure(figsize=(14, 8))
    
    # Plot 1: Distribution of default probabilities
    plt.subplot(2, 2, 1)
    sns.histplot(loan_risk_df['probability_of_default'], bins=20, kde=True)
    plt.title('Distribution of Default Probabilities')
    plt.xlabel('Probability of Default')
    plt.ylabel('Count')
    
    # Plot 2: Risk category breakdown
    plt.subplot(2, 2, 2)
    risk_counts = loan_risk_df['risk_category'].value_counts().sort_index()
    sns.barplot(x=risk_counts.index, y=risk_counts.values)
    plt.title('Loans by Risk Category')
    plt.xlabel('Risk Category')
    plt.ylabel('Number of Loans')
    
    # Plot 3: Expected loss by risk category
    plt.subplot(2, 2, 3)
    risk_losses = loan_risk_df.groupby('risk_category')['expected_loss'].sum()
    sns.barplot(x=risk_losses.index, y=risk_losses.values)
    plt.title('Expected Loss by Risk Category')
    plt.xlabel('Risk Category')
    plt.ylabel('Expected Loss ($)')
    
    # Plot 4: Concentration risk (loan amount vs default probability)
    plt.subplot(2, 2, 4)
    sns.scatterplot(
        x='probability_of_default', 
        y='loan_amount', 
        hue='risk_category',
        data=loan_risk_df
    )
    plt.title('Loan Amount vs Default Probability')
    plt.xlabel('Probability of Default')
    plt.ylabel('Loan Amount ($)')
    
    plt.tight_layout()
    plt.show()

def compare_model_performance(actuals, predictions_model1, predictions_model2, model_names=['Model 1', 'Model 2']):
    """
    Compare ROC curves of two different prediction models
    
    Parameters:
    -----------
    actuals : array-like
        Actual loan default outcomes (1 = default, 0 = non-default)
    predictions_model1 : array-like
        Predicted probabilities from first model
    predictions_model2 : array-like
        Predicted probabilities from second model
    model_names : list, default=['Model 1', 'Model 2']
        Names of the models for the legend
        
    Returns:
    --------
    None (displays plot)
    """
    # Calculate ROC curve for Model 1
    fpr1, tpr1, _ = roc_curve(actuals, predictions_model1)
    roc_auc1 = auc(fpr1, tpr1)
    
    # Calculate ROC curve for Model 2
    fpr2, tpr2, _ = roc_curve(actuals, predictions_model2)
    roc_auc2 = auc(fpr2, tpr2)
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    plt.plot(fpr1, tpr1, lw=2, label=f'{model_names[0]} (AUC = {roc_auc1:.3f})')
    plt.plot(fpr2, tpr2, lw=2, label=f'{model_names[1]} (AUC = {roc_auc2:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.show()
    
    # Calculate and print performance improvement
    performance_diff = abs(roc_auc2 - roc_auc1)
    better_model = model_names[1] if roc_auc2 > roc_auc1 else model_names[0]
    print(f"{better_model} performs better by {performance_diff:.3f} AUC")
    

def calibration_analysis(probabilities, actuals, bins=10):
    """
    Analyze how well calibrated the default probabilities are
    
    Parameters:
    -----------
    probabilities : array-like
        Predicted probabilities of default
    actuals : array-like
        Actual loan outcomes (1 = default, 0 = non-default)
    bins : int, default=10
        Number of bins to use for calibration plot
        
    Returns:
    --------
    None (displays plot)
    """
    # Create a DataFrame for analysis
    calib_df = pd.DataFrame({
        'predicted_prob': probabilities,
        'actual': actuals
    })
    
    # Create bins based on predicted probabilities
    calib_df['prob_bin'] = pd.cut(calib_df['predicted_prob'], bins=bins)
    
    # Calculate actual default rate in each bin
    calibration = calib_df.groupby('prob_bin')['actual'].agg(['mean', 'count']).reset_index()
    calibration.columns = ['prob_bin', 'actual_default_rate', 'count']
    
    # Extract bin midpoints for plotting
    calibration['bin_mid'] = calibration['prob_bin'].apply(lambda x: (x.left + x.right) / 2)
    
    # Plot calibration
    plt.figure(figsize=(12, 8))
    
    # Plot the ideal calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Plot actual calibration
    plt.scatter(calibration['bin_mid'], calibration['actual_default_rate'], 
                s=calibration['count']/sum(calibration['count'])*5000, 
                alpha=0.6, label='Model Calibration')
    
    plt.xlabel('Predicted Default Probability')
    plt.ylabel('Actual Default Rate')
    plt.title('Model Calibration Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # Example usage with dummy data
    print("This module provides risk assessment tools for your loan default models.")
    print("Import the module in your analysis script to use its functions.")
