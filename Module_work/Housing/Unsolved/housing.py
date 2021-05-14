# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import sklearn as skl
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# %%
pd.options.display.max_columns = None
df = pd.read_csv('../Resources/AmesHousing.csv')
#metadata: https://rdrr.io/cran/AmesHousing/man/ames_raw.html


# %%
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df.dtypes)


# %%
# Create category_columns and numeric_columns variables
numerical = df.select_dtypes('number').columns.values.tolist()
categorical = df.select_dtypes('object').columns.values.tolist()

# %%
# Create dummy variables for the category_columns and merge on the numeric_columns to create an X dataset
categorical_encoded = pd.get_dummies((df[categorical]))
#%%
X = df[numerical].merge(categorical_encoded, left_index=True, right_index=True)
# %%
# Fill in missing values in X with zeroes
X = X.fillna(0)
X.drop(columns='SalePrice', inplace=True)
# %%
# Create a y series from SalePrice
y = df['SalePrice']

# %%
# Scale X_train and X_test
scaler: StandardScaler = StandardScaler()
# %%
scaler.fit(X)
# %%
X_scaled = pd.DataFrame(scaler.transform(X))

#%%
X_scaled.columns = X.columns

# %%
# Split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state = 0)


# %%
# Create a neural network model with keras
inputDimensions = 306
hiddenNodes = inputDimensions * 3

# %%
# Add a hidden layer with twice as many neurons as there are inputs. Use 'relu'


# %%
# add an output layer with a 'linear' activation function.
nn = Sequential()

nn.add(Dense(units=hiddenNodes, input_dim=inputDimensions, activation='relu'))

nn.add(Dense(units=hiddenNodes, input_dim=inputDimensions, activation='relu'))

nn.add(Dense(units=hiddenNodes, input_dim=inputDimensions, activation='relu'))

nn.add(Dense(units=1, activation='linear'))
# %%
nn.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
# %%
model = nn.fit(X_train, y_train, epochs=100)

# %%
# print a summary of the model
y_pred = nn.predict(X_test)

# %%
# compile the model using the "adam" optimizer and "mean_squared_error" loss function
r2_score(y_test, y_pred)

# %%
# train the model for 100 epochs


# %%
# predict values for the train and test sets


# %%
# score the training predictions with r2_score()


# %%
# score the test predictions with r2_score()


# %%
# create a deep learning model with two hidden layers. Use the same number of nodes as the neural network model.


# %%
# add a linear output node


# %%
# print the deep learning model summary


# %%
# compile the model


# %%
# train the model for 100 epochs


# %%
# predict values for the train and test sets


# %%
# score the training predictions with r2_score()


# %%
# score the test predictions with r2_score(), compare values to the neural network model


# %%



