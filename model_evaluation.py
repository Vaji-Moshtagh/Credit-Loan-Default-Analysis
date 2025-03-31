import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, roc_curve, roc_auc_score
import seaborn as sns

def evaluate_threshold_impact(y_true, probabilities, thresholds=None, avg_loan_amount=15000):
    """
    Evaluate the impact of different probability thresholds on model performance
    and calculate potential financial impact.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    probabilities : array-like
        Predicted probabilities of the positive class
    thresholds : array-like or None
        Threshold values to evaluate, if None uses np.arange(0.1, 0.7, 0.05)
    avg_loan_amount : float
        Average loan amount for financial impact calculation
        
    Returns:
    --------
    DataFrame containing threshold evaluation metrics and a plot
    """
    # Default thresholds if none provided
    if thresholds is None:
        thresholds = np.arange(0.1, 0.7, 0.05)
    
    results = []
    
    for threshold in thresholds:
        # Create binary predictions based on threshold
        y_pred = (probabilities > threshold).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate recall rates
        default_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        non_default_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # Calculate precision
        default_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Calculate F1 score
        f1 = 2 * (default_precision * default_recall) / (default_precision + default_recall) if (default_precision + default_recall) > 0 else 0
        
        # Calculate financial impact (simplified model)
        predicted_defaults = tp + fp
        missed_defaults = fn
        financial_impact = missed_defaults * avg_loan_amount
        
        # Store results
        results.append({
            'threshold': threshold,
            'default_recall': default_recall,
            'non_default_recall': non_default_recall,
            'accuracy': accuracy,
            'default_precision': default_precision,
            'f1_score': f1,
            'predicted_defaults': predicted_defaults,
            'true_defaults': tp + fn,
            'missed_defaults': missed_defaults,
            'financial_impact': financial_impact
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['threshold'], results_df['default_recall'], 'b-', label='Default Recall')
    plt.plot(results_df['threshold'], results_df['non_default_recall'], 'r-', label='Non-default Recall')
    plt.plot(results_df['threshold'], results_df['accuracy'], 'g-', label='Model Accuracy')
    plt.xlabel('Probability Threshold')
    plt.ylabel('Score')
    plt.title('Impact of Threshold on Model Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return results_df

def plot_confusion_matrix(y_true, y_pred, labels=['Non-Default', 'Default']):
    """
    Plot a confusion matrix with percentages and counts.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels
    labels : list
        Class labels for display
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Calculate percentages
    total = cm.sum()
    accuracy = (cm[0, 0] + cm[1, 1]) / total
    
    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"True Negatives: {cm[0, 0]} ({cm[0, 0]/total:.2%} of total)")
    print(f"False Positives: {cm[0, 1]} ({cm[0, 1]/total:.2%} of total)")
    print(f"False Negatives: {cm[1, 0]} ({cm[1, 0]/total:.2%} of total)")
    print(f"True Positives: {cm[1, 1]} ({cm[1, 1]/total:.2%} of total)")

def calculate_business_impact(y_true, y_pred, avg_loan_amount=15000, avg_interest_rate=0.10, default_loss_rate=0.6):
    """
    Calculate the business impact of the model's predictions.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (1 = default, 0 = non-default)
    y_pred : array-like
        Predicted binary labels
    avg_loan_amount : float
        Average loan amount
    avg_interest_rate : float
        Average annual interest rate
    default_loss_rate : float
        Average percentage of loan amount lost in case of default
        
    Returns:
    --------
    Dictionary with various business metrics
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate business metrics
    # True negatives: Loans correctly identified as non-default (revenue from interest)
    revenue_from_good_loans = tn * avg_loan_amount * avg_interest_rate
    
    # False positives: Good loans incorrectly rejected (opportunity cost)
    opportunity_cost = fp * avg_loan_amount * avg_interest_rate
    
    # False negatives: Bad loans incorrectly approved (losses)
    losses_from_missed_defaults = fn * avg_loan_amount * default_loss_rate
    
    # True positives: Bad loans correctly rejected (savings)
    savings_from_caught_defaults = tp * avg_loan_amount * default_loss_rate
    
    # Net impact
    net_impact = revenue_from_good_loans - losses_from_missed_defaults
    
    # Return the results
    return {
        'revenue_from_good_loans': revenue_from_good_loans,
        'opportunity_cost': opportunity_cost,
        'losses_from_missed_defaults': losses_from_missed_defaults,
        'savings_from_caught_defaults': savings_from_caught_defaults,
        'net_impact': net_impact
    }

def find_optimal_threshold(y_true, probabilities, metric='f1', thresholds=None):
    """
    Find the optimal threshold based on a specific metric.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    probabilities : array-like
        Predicted probabilities of the positive class
    metric : str
        Metric to optimize ('f1', 'accuracy', 'balanced', 'business')
    thresholds : array-like or None
        Threshold values to evaluate, if None uses np.arange(0.1, 0.9, 0.01)
        
    Returns:
    --------
    Optimal threshold value and its corresponding metrics
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.01)
    
    best_threshold = None
    best_score = -np.inf
    
    for threshold in thresholds:
        y_pred = (probabilities > threshold).astype(int)
        
        if metric == 'f1':
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            score = f1
        elif metric == 'accuracy':
            score = (y_true == y_pred).mean()
        elif metric == 'balanced':
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            score = (specificity + sensitivity) / 2
        elif metric == 'business':
            # A custom business metric that balances false positives and false negatives
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            # Assuming default costs 5 times more than a lost opportunity
            score = -(fn * 5 + fp)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    # Calculate metrics at the best threshold
    y_pred_best = (probabilities > best_threshold).astype(int)
    
    return {
        'optimal_threshold': best_threshold,
        'best_score': best_score,
        'confusion_matrix': confusion_matrix(y_true, y_pred_best),
        'classification_report': classification_report(y_true, y_pred_best, output_dict=True)
    }

# Example usage
if __name__ == "__main__":
    # This would be called when running this file directly
    print("Credit Risk Model Evaluation Utilities")
    print("Import this module to use these functions in your model evaluation.")
