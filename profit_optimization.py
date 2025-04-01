# profit_optimization.py
# A module for determining optimal loan acceptance thresholds based on profitability analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_expected_profit(df, prob_col='prob_default', amount_col='loan_amnt', 
                             status_col='true_loan_status', interest_rate=0.15, 
                             recovery_rate=0.2, term_years=3):
    """
    Calculate expected profit for each loan based on probability of default
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing loan data
    prob_col : str
        Column name for default probability
    amount_col : str
        Column name for loan amount
    status_col : str
        Column name for actual loan status (1=default, 0=non-default)
    interest_rate : float
        Annual interest rate charged on loans
    recovery_rate : float
        Fraction of loan amount recovered in case of default
    term_years : int
        Loan term in years
        
    Returns:
    --------
    pandas.DataFrame
        Original dataframe with additional columns for expected profit
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Calculate expected profit for each loan
    # For non-defaulting loans: Present value of interest payments
    # For defaulting loans: Recovery amount minus principal
    
    # Simple present value calculation of interest payments
    interest_value = result_df[amount_col] * interest_rate * term_years
    
    # Expected loss calculation
    expected_loss = result_df[prob_col] * result_df[amount_col] * (1 - recovery_rate)
    
    # Expected profit: interest income minus expected loss
    result_df['expected_profit'] = interest_value - expected_loss
    
    return result_df

def find_optimal_threshold(df, threshold_range=None, prob_col='prob_default', 
                          profit_col='expected_profit', steps=20, plot=True):
    """
    Find the optimal threshold for loan acceptance based on maximizing total profit
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing loan data with expected profit column
    threshold_range : tuple
        (min, max) for threshold values to test
    prob_col : str
        Column name for default probability
    profit_col : str
        Column name for expected profit
    steps : int
        Number of threshold values to test
    plot : bool
        Whether to generate visualization of results
        
    Returns:
    --------
    dict
        Results including optimal threshold, acceptance rate, and expected profit
    """
    if threshold_range is None:
        threshold_range = (0.01, 0.99)
        
    # Create array of thresholds to test
    thresholds = np.linspace(threshold_range[0], threshold_range[1], steps)
    
    # Results storage
    results = {
        'threshold': [],
        'acceptance_rate': [],
        'total_profit': [],
        'bad_rate': []
    }
    
    # Test each threshold
    for threshold in thresholds:
        # Determine which loans would be accepted
        accepted = df[df[prob_col] <= threshold]
        
        # Calculate acceptance rate
        acceptance_rate = len(accepted) / len(df)
        
        # Calculate total profit from accepted loans
        total_profit = accepted[profit_col].sum()
        
        # Calculate bad rate (percentage of accepted loans that default)
        if len(accepted) > 0:
            bad_rate = accepted[accepted['true_loan_status'] == 1].shape[0] / len(accepted)
        else:
            bad_rate = 0
            
        # Store results
        results['threshold'].append(threshold)
        results['acceptance_rate'].append(acceptance_rate)
        results['total_profit'].append(total_profit)
        results['bad_rate'].append(bad_rate)
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    # Find optimal threshold
    optimal_idx = results_df['total_profit'].idxmax()
    optimal_results = {
        'optimal_threshold': results_df.loc[optimal_idx, 'threshold'],
        'optimal_acceptance_rate': results_df.loc[optimal_idx, 'acceptance_rate'],
        'optimal_bad_rate': results_df.loc[optimal_idx, 'bad_rate'],
        'max_profit': results_df.loc[optimal_idx, 'total_profit'],
        'results_df': results_df
    }
    
    # Generate visualization if requested
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot profit vs acceptance rate
        ax1.plot(results_df['acceptance_rate'], results_df['total_profit'])
        ax1.set_title('Expected Profit by Acceptance Rate')
        ax1.set_xlabel('Acceptance Rate')
        ax1.set_ylabel('Expected Profit ($)')
        ax1.axvline(x=optimal_results['optimal_acceptance_rate'], color='r', linestyle='--')
        ax1.grid(True)
        
        # Plot bad rate vs acceptance rate
        ax2.plot(results_df['acceptance_rate'], results_df['bad_rate'])
        ax2.set_title('Bad Rate by Acceptance Rate')
        ax2.set_xlabel('Acceptance Rate')
        ax2.set_ylabel('Bad Rate')
        ax2.axvline(x=optimal_results['optimal_acceptance_rate'], color='r', linestyle='--')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    return optimal_results

def visualize_profit_components(df, threshold, prob_col='prob_default', 
                               amount_col='loan_amnt', status_col='true_loan_status'):
    """
    Create visualizations showing the components of profit at a given threshold
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing loan data
    threshold : float
        Probability threshold for loan acceptance
    prob_col : str
        Column name for default probability
    amount_col : str
        Column name for loan amount
    status_col : str
        Column name for actual loan status
        
    Returns:
    --------
    None (displays visualizations)
    """
    # Separate accepted and rejected loans
    accepted = df[df[prob_col] <= threshold]
    rejected = df[df[prob_col] > threshold]
    
    # Create confusion matrix data
    true_positive = accepted[accepted[status_col] == 0].shape[0]  # Good loans correctly accepted
    false_positive = accepted[accepted[status_col] == 1].shape[0]  # Bad loans incorrectly accepted
    false_negative = rejected[rejected[status_col] == 0].shape[0]  # Good loans incorrectly rejected
    true_negative = rejected[rejected[status_col] == 1].shape[0]  # Bad loans correctly rejected
    
    # Calculate key metrics
    acceptance_rate = len(accepted) / len(df)
    bad_rate = false_positive / len(accepted) if len(accepted) > 0 else 0
    
    # Create a confusion matrix visualization
    cm = np.array([[true_positive, false_negative], [false_positive, true_negative]])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Accepted', 'Rejected'],
               yticklabels=['Good Loan', 'Bad Loan'])
    plt.title(f'Loan Decision Outcomes (Threshold = {threshold:.3f})')
    plt.ylabel('Actual Outcome')
    plt.xlabel('Model Decision')
    
    # Add key metrics as text
    plt.figtext(0.5, 0.01, 
                f'Acceptance Rate: {acceptance_rate:.2%} | Bad Rate: {bad_rate:.2%}',
                ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    print("This module provides functions for loan profitability analysis and optimization.")
    print("Import and use these functions in your main analysis script.")
