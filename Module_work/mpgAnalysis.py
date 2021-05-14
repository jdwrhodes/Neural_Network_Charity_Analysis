#%%
import pandas as pd
from pandas import DataFrame, Series
from path import Path

#%%
mpgPath = Path('./resources/mpg.csv')

#%%
mpg_df: DataFrame = pd.read_csv(mpgPath, na_values='?')
mpg_df

#%%
mpg_df.dtypes

#%%
# mpg_df['horsepower'].unique()

# #%%
# mpg_df['horsepower'].astype(int)

# #%%
# try:
#     mpg_df['horsepower'].astype(int)
# except ValueError as e :
#     print(e)

#%%
mpg_df.dropna(inplace=True)
# %%
mpg_df['cylinders'].unique()
# %%
mpg_df['model year'].unique()

# %%
from sklearn.preprocessing import OneHotEncoder
# %%
enc: OneHotEncoder = OneHotEncoder(sparse=False)

# %%
mpg_encoded: DataFrame = pd.DataFrame(enc.fit_transform(mpg_df[['origin']]))

# %%
mpg_encoded.columns = enc.get_feature_names(['origin'])

# %%
mpg_encoded

# %%
mpg_df = mpg_df.merge(mpg_encoded, left_index=True, right_index=True)
# %%
mpg_df.head()
# %%
mpg_encoded: DataFrame = pd.DataFrame(enc.fit_transform(mpg_df[['cylinders']]))

# %%
mpg_encoded.columns = enc.get_feature_names(['cylinders'])

# %%
mpg_encoded

# %%
mpg_df = mpg_df.merge(mpg_encoded, left_index=True, right_index=True)
# %%
mpg_df.head()
# %%
mpg_df.shape

# %%
mpg_df.columns
# %%
mpg_clean = mpg_df[['mpg', 'displacement', 'horsepower', 'weight',
       'acceleration', 'model year', 'origin_1',
       'origin_2', 'origin_3', 'cylinders_3', 'cylinders_4', 'cylinders_5',
       'cylinders_6', 'cylinders_8']]
# %%
mpg_clean.head()

# %%
from sklearn.preprocessing import MinMaxScaler
# %%
X_scaled = pd.DataFrame(MinMaxScaler().fit_transform(mpg_clean))
# %%
X_scaled.columns = mpg_clean.columns
# %%
X_scaled
# %%
from sklearn.model_selection import train_test_split
# %%
y: Series = X_scaled['mpg']
X: DataFrame = X_scaled.drop(['mpg'], 1)
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# %%
X_train.describe().transpose()[['mean', 'std']]

# %%
import seaborn as sns
# %%
sns.pairplot(X_scaled[['displacement', 'horsepower', 'weight',
       'acceleration', 'model year', 'origin_1',
       'origin_2', 'origin_3', 'cylinders_3', 'cylinders_4', 'cylinders_5',
       'cylinders_6', 'cylinders_8']], diag_kind='kde')
# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# %%
nn: Sequential = Sequential()

# %%
X_train.shape
#%%
inputFeatures = 13
hiddenNodes = inputFeatures * 3

nn.add(
    Dense(units=hiddenNodes, input_dim=inputFeatures)
)
# %%
nn.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
# %%
nnTrained = nn.fit(X_train, y_train, epochs=200)
# %%
import matplotlib.pyplot as plt
# %%
plt.plot(nnTrained.history['loss'])
# %%
loss, accuracy = nn.evaluate(X_test, y_test)
# %%
print(f'Loss: {loss} \t Accuracy: {accuracy}')
# %%
