# Credit Loan Data Analysis

This repository contains analysis of a credit loan dataset to understand factors affecting loan defaults.

## Description

This project analyzes a credit loan dataset to identify patterns and relationships between various factors such as:
- Loan amounts
- Interest rates
- Income levels
- Age
- Employment length
- Home ownership status
- Loan intent

The analysis helps understand which factors are most strongly associated with loan defaults.

## Files

- `credit_loan_analysis.py`: Main Python script containing all the analysis code
- `credit_loan_data.csv`: Dataset file (you need to provide this)
- Generated visualizations:
  - `loan_amount_histogram.png`: Distribution of loan amounts
  - `income_vs_age_scatter.png`: Scatter plot of income vs age
  - `percent_income_boxplot.png`: Box plot of loan percent income by loan status
  - `age_vs_loan_amount.png`: Scatter plot of age vs loan amount
  - `age_vs_interest_rate.png`: Scatter plot of age vs interest rate colored by loan status

## Key Findings

- Higher loan-to-income percentages are associated with higher default rates
- Borrowers with MORTGAGE status tend to have lower default rates
- Medical loans have the highest number of defaults
- Employment stability (longer employment lengths) appears to correlate with lower default rates
- Age outliers (over 100) and employment length outliers (over 60 years) were identified and removed

## Requirements

- Python 3.x
- pandas
- matplotlib
- numpy

## Usage

1. Place your credit loan dataset as `credit_loan_data.csv` in the same directory
2. Run the script:
