import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(data, columns=None):
    """
    Create a correlation heatmap for selected features.
    
    Args:
        data (pandas.DataFrame): Dataset
        columns (list): Columns to include in the heatmap
    """
    if columns is None:
        # Use numeric columns only
        numeric_data = data.select_dtypes(include=[np.number])
    else:
        numeric_data = data[columns]
    
    # Calculate correlation matrix
    corr = numeric_data.corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.show()

def analyze_categorical_impact(data, categorical_col, target_col='loan_status'):
    """
    Analyze the impact of a categorical variable on the target.
    
    Args:
        data (pandas.DataFrame): Dataset
        categorical_col (str): Categorical column to analyze
        target_col (str): Target column
    """
    # Group by categorical column and calculate target mean
    impact = data.groupby(categorical_col)[target_col].agg(['mean', 'count'])
    impact.columns = ['Default Rate', 'Count']
    impact = impact.sort_values('Default Rate', ascending=False)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Plot default rate
    ax1 = plt.subplot(1, 2, 1)
    impact['Default Rate'].plot(kind='bar', ax=ax1, color='skyblue')
    plt.title(f'Default Rate by {categorical_col}')
    plt.ylabel('Default Rate')
    plt.xticks(rotation=45)
    
    # Plot count
    ax2 = plt.subplot(1, 2, 2)
    impact['Count'].plot(kind='bar', ax=ax2, color='lightgreen')
    plt.title(f'Count by {categorical_col}')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{categorical_col}_impact.png')
    plt.show()
    
    return impact

def calculate_vif(data, features):
    """
    Calculate Variance Inflation Factor to detect multicollinearity.
    
    Args:
        data (pandas.DataFrame): Dataset
        features (list): Features to check for multicollinearity
        
    Returns:
        pandas.DataFrame: VIF values for each feature
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # Create a dataframe with only the features
    X = data[features]
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    # Sort by VIF value
    vif_data = vif_data.sort_values("VIF", ascending=False)
    
    return vif_data
