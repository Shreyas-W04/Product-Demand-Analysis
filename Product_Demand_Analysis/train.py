import pandas as pd
import numpy as np
from src.model import train_model
from src.data_processing import load_data, create_lag_features, train_test_split_time_series

if __name__ == "__main__":
    print("--- 1. Loading and Feature Engineering ---")
    df = load_data()
    df = create_lag_features(df)
    
    # Split: Train vs Hold-out Test (Last 90 days)
    full_train_df, test_df = train_test_split_time_series(df, test_size_days=90)

    # Split: Train vs Validation (Last 30 days of training)
    max_train_date = full_train_df['date'].max()
    split_val_date = max_train_date - pd.Timedelta(days=30)
    
    train_df = full_train_df[full_train_df['date'] <= split_val_date].copy()
    val_df = full_train_df[full_train_df['date'] > split_val_date].copy()

    # Define Features
    feature_cols = [c for c in train_df.columns if c not in [
        "demand", "date", "product_name"
    ]]
    
    # Ensure product_id is first
    if 'product_id' in feature_cols:
        feature_cols.remove('product_id')
        feature_cols.insert(0, 'product_id')

    # Convert Categoricals
    categorical_cols = ['product_id', 'product_month_interaction']
    for col in categorical_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype('category')
        if col in val_df.columns:
            val_df[col] = val_df[col].astype('category')
        
    print(f"Features: {len(feature_cols)}")
    print(f"Train Rows: {len(train_df)} | Val Rows: {len(val_df)}")
    
    print("\n--- 2. Training Model ---")
    trained_model = train_model(train_df, feature_cols, val_df=val_df, target='demand')
    
    print("\nTraining completed.")