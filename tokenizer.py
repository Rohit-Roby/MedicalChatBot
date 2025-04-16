import pandas as pd
df = pd.read_csv('processed_data.csv')
print(type(df))
print(df['sys_prompt'].isnull)().sum() # Check for missing values
print(df.dtypes) # Check data types of each column
print(df.describe()) # Get a summary of the dataset


# Save the processed data to a new CSV file
# df.to_csv('processed_data_cleaned.csv', index=False)

