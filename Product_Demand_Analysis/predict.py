import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data_processing import load_data, create_lag_features, train_test_split_time_series
from src.model import load_model
from typing import Dict, Any, List

FULL_LAGS = [1, 7, 14, 28, 42, 60]
START_DATE = pd.Timestamp("2022-01-01")

def get_base_features(date: pd.Timestamp, product_id: str, avg_price: float) -> Dict[str, Any]:
    dayofweek = date.weekday()
    day_of_year = date.dayofyear
    
    try:
        p_num = int(product_id[1:])
    except:
        p_num = 0
    
    id_group = p_num % 2
    days_elapsed = (date - START_DATE).days
    
    # Trend Interaction Calculation
    trend_direction = 1 if id_group == 1 else -1
    trend_sim = days_elapsed * trend_direction

    row = {
        'day': date.day,
        'month': date.month,
        'dayofweek': dayofweek,
        'is_weekend': 1 if dayofweek >= 5 else 0,
        
        'product_num': p_num,
        'id_group': id_group,
        'days_elapsed': days_elapsed,
        
        # New Interaction Feature
        'trend_direction': trend_direction,
        'trend_sim': trend_sim,
        
        'day_of_year': day_of_year,
        'sin_annual': np.sin(2 * np.pi * day_of_year / 365.25),
        'cos_annual': np.cos(2 * np.pi * day_of_year / 365.25),
        
        'is_christmas': 1 if (date.month == 12 and date.day == 25) else 0,
        'is_newyear': 1 if (date.month == 1 and date.day == 1) else 0,
        'is_july4': 1 if (date.month == 7 and date.day == 4) else 0,
        
        'product_month_interaction': f"{product_id}_{date.month}",
        
        # Placeholder
        'price_diff': 0.0,
        'price_ratio': 1.0
    }
    return row

def predict_for_product(product_id: str, days_ahead: int = 7) -> List[Dict[str, Any]]:
    model, features = load_model()
    df = load_data()
    avg_price = df[df['product_id'] == product_id]['price'].mean()
    
    df = df[df['product_id'] == product_id].copy()
    df = df.sort_values('date')
    history = df[['date', 'demand', 'price', 'promotion']].tail(100).copy() 
    
    preds = []
    
    for step in range(1, days_ahead + 1):
        current_date = history['date'].max() + pd.Timedelta(days=1)
        row = get_base_features(current_date, product_id, avg_price)
        row['date'] = current_date
        row['product_id'] = product_id
        
        last_known = history.iloc[-1]
        row['price'] = last_known['price'] 
        row['promotion'] = 0 
        
        # Update Price Features
        row['price_diff'] = row['price'] - avg_price
        row['price_ratio'] = row['price'] / avg_price

        for lag in FULL_LAGS:
            if lag <= len(history):
                row[f'demand_lag_{lag}'] = history['demand'].iloc[-lag] 
                row[f'price_lag_{lag}'] = history['price'].iloc[-lag]
            else:
                row[f'demand_lag_{lag}'] = history['demand'].mean() 
                row[f'price_lag_{lag}'] = history['price'].mean()

        shifted_demand = history['demand'] 
        row['rolling_7_mean'] = shifted_demand.tail(7).mean() 
        row['rolling_28_mean'] = shifted_demand.tail(28).mean()
        row['rolling_7_std'] = shifted_demand.tail(7).std()
        
        X = pd.DataFrame([row])
        X['product_id'] = X['product_id'].astype('category')
        X['product_month_interaction'] = X['product_month_interaction'].astype('category')

        # Ensure columns match
        X = X[features]
        pred = model.predict(X)[0]
        pred_actual = max(0, round(pred, 0))
        
        # --- FIX: Added 'price' to the appended dictionary ---
        preds.append({
            'date': current_date.strftime('%Y-%m-%d'), 
            'predicted_demand': float(pred_actual),
            'price': float(row['price'])
        })
        
        new_history_row = {
            'date': current_date, 
            'demand': pred_actual,
            'price': row['price'], 
            'promotion': row['promotion']
        }
        history = pd.concat([history, pd.DataFrame([new_history_row])], ignore_index=True, sort=False)
        
    return preds

def evaluate_model():
    model, features = load_model()
    df = load_data()
    df = create_lag_features(df)
    
    train_df, test_df = train_test_split_time_series(df, test_size_days=90)
    
    X_test = test_df[features].copy() 
    y_test = test_df['demand']
    
    categorical_cols = ['product_id', 'product_month_interaction']
    for col in categorical_cols:
        X_test[col] = X_test[col].astype('category')
    
    y_pred = model.predict(X_test)
    y_pred_actual = np.maximum(0, y_pred) 
    
    mae = mean_absolute_error(y_test, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_actual))
    r2 = r2_score(y_test, y_pred_actual)

    print(f"\n--- Evaluation for ALL Products ---")
    print(f"MAE: {round(mae, 2)}")
    print(f"RMSE: {round(rmse, 2)}")
    print(f"R²: {round(r2, 4)}")

if __name__ == "__main__":
    evaluate_model()