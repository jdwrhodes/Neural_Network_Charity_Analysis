#%%
# Import our dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import pandas as pd
import tensorflow as tf

# Import our input dataset
attrition_df = pd.read_csv('./resources/HR-Employee-Attrition.csv')
attrition_df.head()
# %%
# Generate our categorical variable list
attrition_cat = attrition_df.dtypes[attrition_df.dtypes == 'object'].index.tolist()

#%%
# Check the number of unique values in each column. nunique drops null values from the count automatically
attrition_df[attrition_cat].nunique()
# %%
# Create a OneHotEncoder instance
enc = OneHotEncoder(sparse=False)

# Fit and Transform the OneHotEncoder using the categorical variable list
encode_df = pd.DataFrame(enc.fit_transform(attrition_df[attrition_cat]))

# Add the encoded variable names to the Dataframe
encode_df.columns = enc.get_feature_names(attrition_cat)
encode_df.head()
# %%
# Merge one-hot encoded feaures and drop the originals
attrition_df = attrition_df.merge(encode_df, left_index=True, right_index=True)
attrition_df = attrition_df.drop(attrition_cat, 1)
attrition_df.head()
# %%
# Split our preprocessed data into our features and target arrays
y = attrition_df['Attrition_Yes'].values
X = attrition_df.drop(['Attrition_Yes', 'Attrition_No'], 1).values

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)
# %%
# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
# %%
# Define the model - deep neural net
number_input_features = len(X_train[0])
hidden_nodes_layer1 = 8
hidden_nodes_layer2 = 5

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation='relu'))

# Second Hidden Layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation='relu'))

# Output Layer
nn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Check the structure of the model
nn.summary()
# %%
# Compile the model
nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# %%
# Train the model
fit_model =nn.fit(X_train, y_train, epochs=100)
# %%
# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test, y_test, verbose=2)
print(f'Loss: {model_loss}, Accuracy: {model_accuracy}')
# %%
