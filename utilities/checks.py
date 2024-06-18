import pandas as pd
import numpy as np

# Load your DataFrame
df = pd.read_csv('../train_inat/train.csv')  # Adjust path as necessary

# Check for NaN values
print("NaN values in longitude:", df['longitude'].isna().sum())
print("NaN values in latitude:", df['latitude'].isna().sum())
print(len(df))

# Check for infinite values
print("Infinite values in longitude:", np.isinf(df['longitude']).sum())
print("Infinite values in latitude:", np.isinf(df['latitude']).sum())
