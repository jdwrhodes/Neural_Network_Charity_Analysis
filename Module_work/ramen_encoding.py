#%%
# Import our dependencies
import pandas as pd
import sklearn as skl

# Read in our ramen data
ramen_df = pd.read_csv("./resources/ramen-ratings.csv")

# Print out the Country value counts
country_counts = ramen_df.Country.value_counts()
country_counts
# %%
# Visualize the value counts to determine which variables can be collapsed into an "Other" category
country_counts.plot.density()

#%%
# Determine which values to replace
replace_countries = list(country_counts[country_counts < 100].index)

# Replace in Dataframe
for country in replace_countries:
    ramen_df.Country = ramen_df.Country.replace(country, 'Other')

# Check to make sure binning was successful
ramen_df.Country.value_counts()

#%%
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False)

# Fit the encoder and produce encoded DataFrame
encode_df = pd.DataFrame(enc.fit_transform(ramen_df.Country.values.reshape(-1,1)))

# Rename encoded columns
encode_df.columns = enc.get_feature_names(['Country'])
encode_df.head()
# %%
# Merge the two DataFrames together and drop the Country column
ramen_df = ramen_df.merge(encode_df, left_index=True, right_index=True).drop('Country', 1)
# %%
