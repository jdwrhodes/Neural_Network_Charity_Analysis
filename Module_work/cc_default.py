#%%
import pandas as pd 
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn as skl
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.utils import plot_model
#%%
cc_df: DataFrame = pd.read_csv('./resources/cc_default.csv')
cc_df.head()
# %%
cc_df['DEFAULT'].value_counts()

#%%
X = cc_df.copy()
X = X.drop(columns=['DEFAULT'])

#%%
y = cc_df['DEFAULT']
# %%
scaler: StandardScaler = StandardScaler()
# %%
scaler.fit(X)
# %%
X_scaled = pd.DataFrame(scaler.transform(X))

#%%
X_scaled.columns = X.columns
# %%
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state = 0)
# %%
X_train.shape

#%%
inputDimensions = 23
hiddenNodes = inputDimensions * 3

#%%
nn = Sequential()

nn.add(Dense(units=hiddenNodes, input_dim=inputDimensions, activation='relu'))

nn.add(Dense(units=hiddenNodes, input_dim=inputDimensions, activation='relu'))

nn.add(Dense(units=1, activation='sigmoid'))
# %%
nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# %%
model = nn.fit(X_train, y_train, epochs=250)
# %%
plot_model(nn, show_shapes=True, show_layer_names=True)
# %%
loss_df = pd.DataFrame(model.history, index=range(1, len(model.history['loss']) + 1))
# %%
loss_df.plot(y='loss')

#%%
loss_df.plot(y='accuracy')

#%%
loss, accuracy = nn.evaluate(X_test, y_test)
print(f'Loss: {loss} \t Accuracy: {accuracy}')
# %%
