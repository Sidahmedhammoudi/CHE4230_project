import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split

class DataPreprocessing:
    def __init__(self):
        pass

    def load_data(self, path):
        """Loads dataset from a TXT file, assumes tab-separated values with no headers."""
        
        data = pd.read_csv(path)
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
        
        # Convert 'total_cloud_cover' to numeric
        def convert_cloud_cover(value):
            if value == 'no clouds':
                return 0
            elif value == 'Sky obscured by fog and/or other meteorological phenomena.':
                return 10  # Assume full cloud cover for obscured sky
            elif isinstance(value, str) and '/' in value:
                # Handle fractional values like '2/10–3/10.'
                parts = value.split('/')
                return float(parts[0]) / float(parts[1].split('–')[0])
            else:
                return np.nan  # Handle unexpected values
        
        # Apply the conversion to the 'total_cloud_cover' column
        data['total_cloud_cover[from ten]'] = data['total_cloud_cover[from ten]'].apply(convert_cloud_cover)
        
        # Fill any remaining missing values in 'total_cloud_cover' with the median
        data['total_cloud_cover[from ten]'] = data['total_cloud_cover[from ten]'].fillna(data['total_cloud_cover[from ten]'].median())
        
        # Drop non-numeric columns (e.g., 'Time') if they are not needed
        data = data.drop(columns=['Time'])
        
        return (data)
    
