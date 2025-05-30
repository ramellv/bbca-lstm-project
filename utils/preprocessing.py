import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path)
    df = df.iloc[2:].copy()
    df.rename(columns={'Price': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    
    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
