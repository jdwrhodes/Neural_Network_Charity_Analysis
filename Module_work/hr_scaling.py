#%%
# Import our dependencies
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Read in our dataset
hr_df = pd.read_csv("./resources/hr_dataset.csv")
hr_df.head()
# %%
# Create the StandardScaler instance
scaler = StandardScaler()

#%%
# Fit the StandardScaler
scaler.fit(hr_df)

# %%
# Scale the data
scaled_data = scaler.fit_transform(hr_df)
# %%
# Create a Dataframe with the Scaled Data
transformed_scaled_data = pd.DataFrame(scaled_data, columns=hr_df.columns)
transformed_scaled_data.head()

#%%
