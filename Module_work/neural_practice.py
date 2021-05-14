#%%
# Import our dependencies
import pandas as pd
import matplotlib as plt
from sklearn.datasets import make_blobs
import sklearn as skl
import tensorflow as tf

#%%
# Generate dummy dataset
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=78)

# Creating a Dataframe with the dummy data
df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
df['Target'] = y

# Plotting the dummy data
df.plot.scatter(x='Feature 1', y='Feature 2', c='Target', colormap='winter')

#%%
# Use sklearn to split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)

#%%
# Create scaler instance
X_scaler = skl.preprocessing.StandardScaler()

# Fit the scaler
X_scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

#%%
# Create the Keras Sequential model
nn_model = tf.keras.models.Sequential()

#%%
# Add our first dense layer, including the input layer
nn_model.add(tf.keras.layers.Dense(units=1, activation='relu', input_dim=2))

#%%
# Add the output layer that uses a probability activation function
nn_model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#%%
# Check the structure of the Sequential model
nn_model.summary()

#%%
# Compile the Sequential model together and customize metrics
nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#%%
# Fit the model to the training data
fit_model = nn_model.fit(X_train_scaled, y_train, epochs=100)

#%%
# Create a Dataframe containing training history
history_df = pd.DataFrame(fit_model.history, index=range(1, len(fit_model.history['loss']) + 1))

# Plot the loss
history_df.plot(y='loss')

#%%
# Plot the Accuracy
history_df.plot(y='accuracy')

#%%
history_df

#%%
# Evaluate the model using the test data
model_loss, model_accuracy = nn_model.evaluate(X_test_scaled, y_test, verbose=2)
print(f'Loss: {model_loss}, Accuracy: {model_accuracy}')

#%%
# Predict the classification of a new set of blob data
new_X, new_Y = make_blobs(n_samples=10, centers=2, n_features=2, random_state=78)
new_X_scaled = X_scaler.transform(new_X)
(nn_model.predict(new_X_scaled) > 0.5).astype('int32')

#%%
from sklearn.datasets import make_moons

# Creating dummy nonlinear data
X_moons, y_moons = make_moons(n_samples=1000, noise=0.08, random_state=78)

# Transforming y_moons to a vertical vector
y_moons = y_moons.reshape(-1,1)

# Creating a Dataframe to plot the nonlinear dummy data
df_moons = pd.DataFrame(X_moons, columns=['Feature 1', 'Feature 2'])
df_moons['Target'] = y_moons

# Plot the nonlinear dummy data
df_moons.plot.scatter(x='Feature 1', y='Feature 2', c='Target', colormap='winter')

#%%
# Create the training and testing sets
X_moon_train, X_moon_test, y_moon_train, y_moon_test = train_test_split(
    X_moons, y_moons, random_state=78
)

# Create the scaler instance
X_moon_scaler = skl.preprocessing.StandardScaler()

# Fit the scaler
X_moon_scaler.fit(X_moon_train)

# Scale the data
X_moon_train_scaled = X_moon_scaler.transform(X_moon_train)
X_moon_test_scaled = X_moon_scaler.transform(X_moon_test)

#%%
# Training the model with the nonlinear data
model_moon = nn_model.fit(X_moon_train_scaled, y_moon_train, epochs=100, shuffle=True)

#%%
# Create the Dataframe containing training history
moon_history_df = pd.DataFrame(model_moon.history, index=range(1, len(model_moon.history['loss']) + 1))

# Plot the loss
moon_history_df.plot(y='loss')

#%%
# Plot the accuracy
moon_history_df.plot(y='accuracy')

#%%
# Generate our new Sequential Model
new_model = tf.keras.models.Sequential()

#%%
# Add the input and hidden layer
number_inputs = 2
number_hidden_nodes = 6

new_model.add(tf.keras.layers.Dense(units=number_hidden_nodes, activation='relu', input_dim=number_inputs))

# Add the output layer that uses a probability activation function
new_model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# %%
# Compile the Sequential model together and customize metrics
new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model to the training data
new_fit_model = new_model.fit(X_moon_train_scaled, y_moon_train, epochs=100, shuffle=True)
# %%
