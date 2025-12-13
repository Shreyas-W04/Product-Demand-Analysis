import pandas as pd
import numpy as np
import os

def load_data(path=None):
    if path is None:
        # Robust path finding: looks for data folder relative to this script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        path = os.path.join(project_root, 'data', 'sample_product_demand.csv.gz')

    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find data file at: {path}")

    df = pd.read_csv(path, parse_dates=['date'])
    df = df.sort_values(['product_id', 'date'])
    return df

def create_lag_features(df, lags=[1, 7, 14, 28, 42, 60]):
    df = df.copy()
    
    # --- 1. Label Encoding / Product Grouping ---
    # Extract the integer from "P005" -> 5. 
    # This helps the model learn trends based on ID numbers (Odd vs Even).
    df['product_num'] = df['product_id'].str.extract('(\d+)').astype(int)
    df['id_group'] = df['product_num'] % 2  # 1 for Odd (Growth), 0 for Even (Decline)

    # --- 2. Authentic Holiday Flags ---
    # We flag the specific days, allowing the model to learn the multiplier itself.
    df['is_christmas'] = ((df['date'].dt.month == 12) & (df['date'].dt.day == 25)).astype(int)
    df['is_newyear'] = ((df['date'].dt.month == 1) & (df['date'].dt.day == 1)).astype(int)
    df['is_july4'] = ((df['date'].dt.month == 7) & (df['date'].dt.day == 4)).astype(int)

    # --- 3. Relative Price Feature ---
    # Calculates if the current price is a "deal" compared to the product's average.
    avg_price = df.groupby('product_id')['price'].transform('mean')
    df['price_ratio'] = df['price'] / avg_price

    # --- 4. Trend & Seasonality ---
    start_date = pd.Timestamp("2022-01-01")
    df['days_elapsed'] = (df['date'] - start_date).dt.days 
    
    df['day_of_year'] = df['date'].dt.dayofyear
    # Fourier terms for smooth seasonality
    df['sin_annual'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['cos_annual'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

    # --- 5. Lags & Rolling ---
    for lag in lags:
        df[f'demand_lag_{lag}'] = df.groupby('product_id')['demand'].shift(lag)
        df[f'price_lag_{lag}'] = df.groupby('product_id')['price'].shift(lag)
    
    demand_shifted = df.groupby('product_id')['demand'].shift(1)
    df['rolling_7_mean'] = demand_shifted.rolling(window=7).mean()
    df['rolling_28_mean'] = demand_shifted.rolling(window=28).mean()
    df['rolling_7_std'] = demand_shifted.rolling(window=7).std()
    
    # --- 6. Time Metadata ---
    df['dayofweek'] = df['date'].dt.weekday
    df['month'] = df['date'].dt.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Interaction feature
    df['product_month_interaction'] = df['product_id'] + '_' + df['month'].astype(str)
    
    df = df.dropna()
    return df

def train_test_split_time_series(df, test_size_days=90):
    max_date = df['date'].max()
    split_date = max_date - pd.Timedelta(days=test_size_days)
    train = df[df['date'] <= split_date]
    test = df[df['date'] > split_date]
    return train, test