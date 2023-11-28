#!/usr/bin/env python
# coding: utf-8

# # Task for Today  
# 
# ***
# 
# ## Employee Burnout Prediction  
# 
# Given *data about employees*, let's try to predict the **burnout rate** of a given employee.
# 
# We will use a variety of regression models to make our predictions.

# # Getting Started

# In[10]:


conda install -c anaconda py-xgboost


# In[12]:


get_ipython().system('pip install lightgbm')


# In[14]:


get_ipython().system('pip3 install catboost')



# In[15]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings(action='ignore')


# In[16]:


data = pd.read_csv('../BI Project/are-your-employees-burning-out/train.csv')


# In[17]:


data


# In[18]:


data.info()


# # Preprocessing

# In[19]:


def preprocess_inputs(df):
    df = df.copy()
    
    # Drop Employee ID column
    df = df.drop('Employee ID', axis=1)
    
    # Drop rows with missing target values
    missing_target_rows = df.loc[df['Burn Rate'].isna(), :].index
    df = df.drop(missing_target_rows, axis=0).reset_index(drop=True)
    
    # Fill remaining missing values with column means
    for column in ['Resource Allocation', 'Mental Fatigue Score']:
        df[column] = df[column].fillna(df[column].mean())
    
    # Extract date features
    df['Date of Joining'] = pd.to_datetime(df['Date of Joining'])
    df['Join Month'] = df['Date of Joining'].apply(lambda x: x.month)
    df['Join Day'] = df['Date of Joining'].apply(lambda x: x.day)
    df = df.drop('Date of Joining', axis=1)
    
    # Binary encoding
    df['Gender'] = df['Gender'].replace({'Female': 0, 'Male': 1})
    df['Company Type'] = df['Company Type'].replace({'Product': 0, 'Service': 1})
    df['WFH Setup Available'] = df['WFH Setup Available'].replace({'No': 0, 'Yes': 1})
    
    # Split df into X and y
    y = df['Burn Rate']
    X = df.drop('Burn Rate', axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)
    
    # Scale X
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    
    return X_train, X_test, y_train, y_test


# In[20]:


X_train, X_test, y_train, y_test = preprocess_inputs(data)


# In[21]:


X_train


# In[22]:


y_train


# # Training

# In[23]:


models = {
    "                     Linear Regression": LinearRegression(),
    " Linear Regression (L2 Regularization)": Ridge(),
    " Linear Regression (L1 Regularization)": Lasso(),
    "                   K-Nearest Neighbors": KNeighborsRegressor(),
    "                        Neural Network": MLPRegressor(),
    "Support Vector Machine (Linear Kernel)": LinearSVR(),
    "   Support Vector Machine (RBF Kernel)": SVR(),
    "                         Decision Tree": DecisionTreeRegressor(),
    "                         Random Forest": RandomForestRegressor(),
    "                     Gradient Boosting": GradientBoostingRegressor(),
    "                               XGBoost": XGBRegressor(),
    "                              LightGBM": LGBMRegressor(),
    "                              CatBoost": CatBoostRegressor(verbose=0)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(name + " trained.")


# # Results

# In[25]:


for name, model in models.items():
    print(name + " R^2 Score: {:.5f}".format(model.score(X_test, y_test)))


# In[33]:


# Install the necessary libraries if not already installed
get_ipython().system('pip install dash pandas plotly')
get_ipython().system('pip install dash==0.29.0')

import dash
from dash import dcc, html
import pandas as pd

# Sample data (replace with your dataset)
data
df = pd.DataFrame(data)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    dcc.Graph(
        id='scatter-chart',
        figure={
            'data': [
                {'x': df['Mental Fatigue Score'], 'y': df['Burn Rate'], 'type': 'scatter', 'mode': 'markers'}
            ],
            'layout': {
                'title': 'Burn Rate vs Mental Fatigue Score'
            }
        }
    )
])

# Run the app in the notebook
app.run_server(mode='inline')


# In[ ]:


<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Employee Burnout Dashboard</title>
  <!-- Include Chart.js library -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- Add this line within the head tag of your HTML file -->
<script src="dashboard.js"></script>

</head>
<body>
  <!-- Add canvas elements to render charts -->
  <canvas id="scatterChart" width="400" height="400"></canvas>
  <canvas id="barChart" width="400" height="400"></canvas>
  <canvas id="lineChart" width="400" height="400"></canvas>
  <canvas id="pieChart" width="400" height="400"></canvas>

  <!-- Add other HTML elements as needed -->
</body>
    
    
</html>


# // Sample data, replace this with your actual data
# const employeeData = [
#   { designation: 'A', burnRate: 0.6, mentalFatigue: 50 },
#   { designation: 'B', burnRate: 0.7, mentalFatigue: 60 },
#   // Add more data points as needed
# ];
# 
# // Scatter Chart
# const scatterChartCtx = document.getElementById('scatterChart').getContext('2d');
# const scatterChart = new Chart(scatterChartCtx, {
#   type: 'scatter',
#   data: {
#     datasets: [{
#       label: 'Burn Rate vs Mental Fatigue',
#       data: employeeData.map(entry => ({ x: entry.mentalFatigue, y: entry.burnRate })),
#     }]
#   },
#   options: {
#     scales: {
#       x: {
#         type: 'linear',
#         position: 'bottom',
#         title: {
#           display: true,
#           text: 'Mental Fatigue Score'
#         }
#       },
#       y: {
#         type: 'linear',
#         position: 'left',
#         title: {
#           display: true,
#           text: 'Burn Rate'
#         }
#       }
#     }
#   }
# });
# 
# // Bar Chart
# const barChartCtx = document.getElementById('barChart').getContext('2d');
# const barChart = new Chart(barChartCtx, {
#   type: 'bar',
#   data: {
#     labels: [...new Set(employeeData.map(entry => entry.designation))],
#     datasets: [{
#       label: 'Average Burn Rate by Designation',
#       data: employeeData.reduce((acc, entry) => {
#         acc[entry.designation] = (acc[entry.designation] || 0) + entry.burnRate;
#         return acc;
#       }, {}),
#     }]
#   },
#   options: {
#     scales: {
#       y: {
#         beginAtZero: true,
#         title: {
#           display: true,
#           text: 'Average Burn Rate'
#         }
#       }
#     }
#   }
# });
# 
# // Add similar code for Line Chart and Pie Chart
# 

# Conclusion:
# Research on how scoring threshold is calculated based on newer, more relevant population sample to generate a more accurate threshold
