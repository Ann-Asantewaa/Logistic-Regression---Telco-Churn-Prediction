#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression
# 
# 
# ## Objectives
# 
# * Use Logistic Regression for classification
# * Preprocess data for modeling
# * Implement Logistic regression on real world data
# 
# ## Import needed packages
# 
# - Numpy
# - Matplotlib
# - Pandas
# - Scikit-learn
# 
# Execute these cells to check if you have the above packages
# 
# 

# In[1]:


get_ipython().system('pip install -q numpy pandas scikit-learn matplotlib')


# In[2]:


import pandas as pd
import pandas as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# ### Scenario
# Assume you are employed by a telecommunications firm that is worried about the increasing number of customers switching from its land-line services to cable competitors. They need to understand who is more likely to leave the company.

# ### Data
# 
# Telco Churn is a dataset developed to support a telecommunications company’s analysis of customer turnovers. Each observation represents an individual customer and includes demographic characteristics and service usage details.
# 
# * The data is historical 
# * Each row represents one customer
# * The focus is to `predict` customer `churn` that is customers who will stay with company

# ### Load Data from URL

# In[3]:


churn_df = pd.read_csv("ChurnData.csv")


# In[4]:


churn_df


# ### Feature Selection
# 
# * Select some variables for modelling
# * Target data type must be an integer
# 

# ### Data preprocessing 
# 
# Use a subset of the fields avaible to develop out model
# Assume : selected fields `age`, `tenure`, `address`, `income`, `ed`, `employ`, `equip` and `churn`.
# 

# In[5]:


churn_df = churn_df[['tenure','age','address','income','ed','employ','equip','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df


# ### Key Notes
# 
# For modelling the `input fields` X and the `target field` y, do the below.
# 
# * The target `churn` needs to be stored under the `variable y`
# * The remaining input all combined stored in the `variable X`

# In[6]:


X = np.asarray(churn_df[['tenure','age','address','income','ed','employ','equip']])
X[:5]   # print the first 5 rows


# In[8]:


y = np.asarray(churn_df['churn'])
y[0:5] # print the first 5 values


# * Standardize the dataset to get all the features as the same scale
# 
# * Use StandardScalar function

# In[9]:


X_norm = StandardScaler().fit(X).transform(X)
X_norm[0:5]


# ### Spliting the dataset
# 
# * Data must be set aside for testing hence the need to split the data into training and testing dataset.
# 
# * Use train_test_split 

# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)


# ### Logistic Regression Classifier Modelling
# * Build the model using LogisticRegression from the Scikit-learn package 
# 
# * fit our model with train data set.
# 

# In[12]:


LR = LogisticRegression().fit(X_train, y_train)
print("Coefficients:", LR.coef_)
print("Intercept:", LR.intercept_)


# ### Interpretation
# 
# We trained a logistic regression model to predict which telecom customers are likely to churn. Key results:
# 
# - **Intercept:** -1.471
# 
# - **Coefficients:**
#   - tenure: -0.846  
#   - age: -0.176  
#   - address: -0.124  
#   - income: -0.010  
#   - ed: 0.060  
#   - employ: -0.233  
#   - equip: 0.752  
# 
# - **Interpretation:**
#   - Negative coefficients (`tenure`, `age`, `address`, `income`, `employ`) → reduce the likelihood of churn  
#   - Positive coefficients (`ed`, `equip`) → increase the likelihood of churn  
#   - Negative intercept → overall baseline probability of churn is low
# 
# 

# In[13]:


# the traning data and can be used to predict the output variable. 
#Let us predict the churn parameter for the test data set.

yhat = LR.predict(X_test)
yhat[:10]


# ### Understanding Prediction Probabilities
# 
# To better understand the predictions from the logistic regression model:
# 
# - We can look at the **prediction probability** for a data point in the test dataset.  
# - Use the function `predict_proba()` to get the probability of each class.  
# - The output is an **array with two columns**:  
#   - **First column:** probability that the record belongs to **class 0** (stay)
#   - **Second column:** probability that the record belongs to **class 1**  (churn)
# - The class prediction system uses a **threshold of 0.5**:  
#   - If the probability of class 1 ≥ 0.5 → predicted class = 1  
#   - Otherwise → predicted class = 0  
# - This means the model always predicts the **most likely class** for each data point.
# 

# In[14]:


yhat_prob = LR.predict_proba(X_test)
yhat_prob[:10]


#  
#  Examine what role each input feature has to play in the prediction of the 1 class.use the code below
# 
# **`pd.Series(LR.coef_[0], index=churn_df.columns[:-1])`**  
# 
# *Creates a pandas Series mapping each feature to its logistic regression coefficient.*  
# 
# **`LR.coef_[0]`** contains the coefficients learned by the model.  
# 
# **`churn_df.columns[:-1]`** ensures the coefficients are matched to the feature names (excluding the target).  
# 
# **`coefficients.sort_values().plot(kind='barh')`**  
# 
# Sorts the coefficients and plots them as a horizontal bar chart.* 
# 
# **`plt.title(), plt.xlabel(), plt.show()`**  
# 
# Adds a title, label, and displays the plot.
# 

# In[15]:


coefficients = pd.Series(LR.coef_[0], index=churn_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")
plt.show()


# ### Performance Evaluation
# 
# Evaluate the model's performance in predicting the target variable. 
# Do it by evaluating the log-loss value
# 
# ### Log Loss
# 
# Log loss (Logarithmic loss) measures the difference between the predicted probabilities and the actual class labels.  
# 
# - In our case, the **class labels** are the target variable `Churn`:
#   - `0` → customer **stayed**  
#   - `1` → customer **churned**
# 
# - A **lower log loss value** indicates a **better performing model**.  
# - It takes into account the probability assigned to each class, not just the final predicted class.
# 

# In[16]:


log_loss(y_test, yhat_prob)


# ## Summary of Logistic Regression Analysis
# 
# - A **logistic regression model** was trained to predict customer churn (`0` = stayed, `1` = churned).  
# - **Key features** influencing churn include `tenure`, `age`, `address`, `income`, `ed`, `employ`, and `equip`.  
# - **Prediction probabilities** were generated using `predict_proba()`, with a threshold of 0.5 for class prediction.  
# - **Feature coefficients** show which factors increase or decrease the likelihood of churn.  
# - **Model performance** was evaluated using **log loss**, with a value of **0.626**, indicating the model reasonably predicts churn probabilities.
# 
