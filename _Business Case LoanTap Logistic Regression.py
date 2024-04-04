#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, roc_curve


# In[5]:


data = pd.read_csv('/Users/vaibhav.srivastava/Downloads/logistic_regression.csv')


# In[6]:


# Define Problem Statement and perform Exploratory Data Analysis (EDA)
# Data overview
print(data.head())
print(data.info())


# In[7]:


# Univariate Analysis
# Example:
plt.figure(figsize=(10, 6))
sns.countplot(x='loan_status', data=data)
plt.title('Loan Status Distribution')
plt.show()


# In[8]:


# Bivariate Analysis
# Example:
plt.figure(figsize=(10, 6))
sns.boxplot(x='loan_status', y='loan_amnt', data=data)
plt.title('Loan Amount by Loan Status')
plt.show()


# In[9]:


# Data Preprocessing
# Duplicate value check
duplicates = data.duplicated().sum()
print("Duplicate rows:", duplicates)


# In[10]:


# Missing value treatment
missing_values = data.isnull().sum()
print("Missing values:", missing_values)


# In[11]:


# Outlier treatment
# Example:
outliers = data[['loan_amnt', 'int_rate', 'installment']].apply(lambda x: np.abs(x - x.mean()) / x.std() > 3)
print("Outliers:", outliers.sum())


# In[12]:


# Feature engineering
# Example:
data['pub_rec_flag'] = np.where(data['pub_rec'] > 1.0, 1, 0)


# In[13]:


# Data preparation for modeling
# Example:
X = data[['loan_amnt', 'int_rate', 'installment', 'pub_rec_flag']]
y = data['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


# Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[15]:


# Model building
model = LogisticRegression()
model.fit(X_train_scaled, y_train)


# In[16]:


# Display model coefficients
coefficients = pd.DataFrame({'feature': X.columns, 'coefficient': model.coef_[0]})
print(coefficients)


# In[18]:


# Encode target variable
y_test_binary = y_test.map({'Fully Paid': 1, 'Charged Off': 0})

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_prob)

# Plot ROC curve
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# In[20]:


# Precision Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test_binary, y_pred_prob)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()


# In[24]:


# Convert predicted probabilities to binary labels
threshold = 0.5
y_pred_binary = np.where(y_pred_prob >= threshold, 1, 0)

# Classification Report
print(classification_report(y_test_binary, y_pred_binary))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test_binary, y_pred_binary)
print("Confusion Matrix:")
print(conf_matrix)


# In[25]:


# Calculate percentage of customers who have fully paid their Loan Amount
fully_paid_percentage = (data['loan_status'].value_counts(normalize=True) * 100)['Fully Paid']
print("Percentage of customers who have fully paid their Loan Amount:", fully_paid_percentage)

# Comment about the correlation between Loan Amount and Installment features
loan_installment_corr = data['loan_amnt'].corr(data['installment'])
if loan_installment_corr > 0:
    print("There is a positive correlation between Loan Amount and Installment features.")
elif loan_installment_corr < 0:
    print("There is a negative correlation between Loan Amount and Installment features.")
else:
    print("There is no correlation between Loan Amount and Installment features.")

# The majority of people have home ownership as _______
majority_home_ownership = data['home_ownership'].mode()[0]
print("The majority of people have home ownership as:", majority_home_ownership)

# People with grades ‘A’ are more likely to fully pay their loan. (T/F)
grade_A_fully_paid_percentage = (data[data['grade'] == 'A']['loan_status'].value_counts(normalize=True) * 100)['Fully Paid']
is_grade_A_more_likely_to_fully_pay_loan = grade_A_fully_paid_percentage > fully_paid_percentage
print("People with grades 'A' are more likely to fully pay their loan:", is_grade_A_more_likely_to_fully_pay_loan)

# Name the top 2 afforded job titles
top_afforded_job_titles = data['emp_title'].value_counts().head(2).index.tolist()
print("Top 2 afforded job titles:", top_afforded_job_titles)

# Thinking from a bank's perspective, which metric should our primary focus be on: ROC AUC, Precision, Recall, F1 Score
bank_metric_focus = "ROC AUC"
print("From a bank's perspective, the primary focus should be on:", bank_metric_focus)

# How does the gap in precision and recall affect the bank?
precision_recall_gap_effect = "The gap in precision and recall affects the bank by influencing the balance between approving loans to creditworthy individuals (precision) and avoiding defaults (recall). A larger gap indicates a tradeoff between these two metrics, where improving one may lead to a decrease in the other. Thus, the bank needs to find an optimal balance based on its risk tolerance and business objectives."
print("How does the gap in precision and recall affect the bank:", precision_recall_gap_effect)

# Which were the features that heavily affected the outcome?
# This would depend on the analysis performed. You can mention the features that showed high correlation or significance in the logistic regression model.

# Will the results be affected by geographical location? (Yes/No)
# This would depend on the dataset and the nature of the business. If geographical location plays a significant role in creditworthiness or loan repayment behavior, then the results may be affected. Otherwise, they may not be affected. You can analyze the dataset or provide a general assessment based on domain knowledge.

# Note: Replace 'data' with your DataFrame containing the analysis results.


# In[ ]:




