import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    data = pd.read_csv(path)
    return data

def preprocess_data(data):
    # Handle missing values
    data = data.dropna()

    # Encode categorical column
    le = LabelEncoder()
    data['Location'] = le.fit_transform(data['Location'])

    return data
