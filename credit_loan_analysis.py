# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset (you'll need to update this path to where your data is stored)
# Assuming you have a CSV file named credit_loan_data.csv
cr_loan = pd.read_csv('credit_loan_data.csv')

# Exploratory Data Analysis
# Check the structure of the data
print("Data Types:")
print(cr_loan.dtypes)

# Check the first five rows of the data
print("\nFirst 5 rows:")
print(cr_loan.head())

# Create a histogram of loan amounts
plt.figure(figsize=(10, 6))
plt.hist(x=cr_loan['loan_amnt'], bins='auto', color='blue', alpha=0.7, rwidth=0.85)
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")
plt.title("Distribution of Loan Amounts")
plt.savefig('loan_amount_histogram.png')
plt.show()

# Create a scatter plot of income vs age
plt.figure(figsize=(10, 6))
plt.scatter(cr_loan['person_income'], cr_loan['person_age'], c='blue', alpha=0.5)
plt.xlabel('Personal Income')
plt.ylabel('Person Age')
plt.title('Relationship Between Income and Age')
plt.savefig('income_vs_age_scatter.png')
plt.show()

# Create cross tables for analysis
# Loan intent vs loan status
print("\nLoan Intent vs Loan Status:")
print(pd.crosstab(cr_loan['loan_intent'], cr_loan['loan_status'], margins=True))

# Home ownership vs loan status by loan grade
print("\nHome Ownership vs Loan Status by Loan Grade:")
print(pd.crosstab(cr_loan['person_home_ownership'], [cr_loan['loan_status'], cr_loan['loan_grade']]))

# Home ownership vs loan status with average percent income
print("\nHome Ownership vs Loan Status - Average Percent Income:")
percent_income_table = pd.crosstab(cr_loan['person_home_ownership'], cr_loan['loan_status'],
                                  values=cr_loan['loan_percent_income'], aggfunc='mean')
print(percent_income_table)

# Create a box plot of percentage income by loan status
plt.figure(figsize=(10, 6))
cr_loan.boxplot(column=['loan_percent_income'], by='loan_status')
plt.title('Average Percent Income by Loan Status')
plt.suptitle('')
plt.savefig('percent_income_boxplot.png')
plt.show()

# Analyze employment length
print("\nLoan Status vs Home Ownership - Maximum Employment Length:")
emp_length_table = pd.crosstab(cr_loan['loan_status'], cr_loan['person_home_ownership'],
                             values=cr_loan['person_emp_length'], aggfunc='max')
print(emp_length_table)

# Create an array of indices where employment length is greater than 60 (outliers)
indices = cr_loan[cr_loan['person_emp_length'] > 60].index

# Drop the records from the data based on the indices and create a new dataframe
cr_loan_new = cr_loan.drop(indices)

# Create updated employment length statistics after removing outliers
print("\nLoan Status vs Home Ownership - Employment Length Statistics After Cleaning:")
min_max_emp = pd.DataFrame()
min_max_emp['min'] = pd.crosstab(cr_loan_new['loan_status'], cr_loan_new['person_home_ownership'],
                               values=cr_loan_new['person_emp_length'], aggfunc='min')
min_max_emp['max'] = pd.crosstab(cr_loan_new['loan_status'], cr_loan_new['person_home_ownership'],
                               values=cr_loan_new['person_emp_length'], aggfunc='max')
print(min_max_emp)

# Create a scatter plot for age and loan amount
plt.figure(figsize=(10, 6))
plt.scatter(cr_loan['person_age'], cr_loan['loan_amnt'], c='blue', alpha=0.5)
plt.xlabel("Person Age")
plt.ylabel("Loan Amount")
plt.title("Relationship Between Age and Loan Amount")
plt.savefig('age_vs_loan_amount.png')
plt.show()

# Create a new dataframe without age outliers
cr_loan_new = cr_loan.drop(cr_loan[cr_loan['person_age'] > 100].index)

# Create a scatter plot of age and interest rate colored by loan status
plt.figure(figsize=(10, 6))
colors = ["blue", "red"]
plt.scatter(cr_loan_new['person_age'], cr_loan_new['loan_int_rate'],
          c=cr_loan_new['loan_status'],
          cmap=plt.cm.colors.ListedColormap(colors),
          alpha=0.5)
plt.xlabel("Person Age")
plt.ylabel("Loan Interest Rate")
plt.title("Age vs Interest Rate by Loan Status (Blue: Non-default, Red: Default)")
plt.colorbar(label='Loan Status')
plt.savefig('age_vs_interest_rate.png')
plt.show()
