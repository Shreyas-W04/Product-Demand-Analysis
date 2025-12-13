
# Generates synthetic product demand data for multiple products
import numpy as np
import pandas as pd
from datetime import datetime
import os

os.makedirs('data', exist_ok=True)

np.random.seed(42)


product_names = [
    'Coffee_Beans_Arabica', 'Espresso_Machine_V1', 'Milk_Frother_Pro', 'Tea_Sampler_Green', 'Chai_Latte_Mix',
    'Syrup_Vanilla_SugarFree', 'Syrup_Caramel_Classic', 'Mug_Insulated_Travel', 'French_Press_Small', 'Drip_Coffee_Maker',
    'Pastry_Mix_Croissant', 'Honey_Local_12oz', 'Sugar_Cane_Cubes', 'Spoon_Set_Long', 'Cleaning_Tablets_50ct',
    'Filter_Paper_Cone', 'Water_Kettle_Electric', 'Grinder_Blade_Mini', 'Scale_Digital_Precision', 'Decaf_Blend_House'
]
products = {f'P{str(i).zfill(3)}': name for i, name in enumerate(product_names, 1)}

# Date range
dates = pd.date_range(start="2022-01-01", end="2024-12-31", freq="D")


holidays = {
    (12, 25): 1.5, # Christmas spike
    (1, 1): 0.8,   # New Year's Day dip
    (7, 4): 1.1,   # Independence Day minor spike
}

rows = []
for date in dates:
    days_in_year = 366 if date.is_leap_year else 365
    day_of_year = date.timetuple().tm_yday
    
    # Check for holidays
    holiday_factor = holidays.get((date.month, date.day), 1.0)

    for p_id, p_name in products.items():
        p_index = int(p_id[1:])
        
        # 1. Baseline demand
        base = 50 + (p_index % 20) * 3

        # 2. Weekly seasonality (+10% on weekends)
        weekly = 1.0 + (0.1 if date.weekday() >= 5 else 0)

        # 3. Annual seasonality (sin wave)
        annual = 1.0 + 0.2 * np.sin(2 * np.pi * day_of_year / days_in_year)

        # 4. Product-specific trend: Growing products (odd index) vs. Stable/Declining (even index)
        trend_slope = 0.0001 if p_index % 2 == 1 else -0.00005 
        days_elapsed = (date - dates[0]).days
        trend_factor = 1.0 + trend_slope * days_elapsed
        
    
        promo = 1.0
        is_promo = False
        if np.random.rand() < 0.02:
            promo = 1.5
            is_promo = True

        # Price generation (with base product offset)
        expected_base_price = 10 + (p_index % 10) * 0.5
        price = max(0, round(expected_base_price + np.random.normal(0, 0.5), 2))
        
    
        price_elasticity_factor = 1.0 - 0.05 * (price - expected_base_price) 
        price_elasticity_factor = max(0.5, price_elasticity_factor) 

        # Final Demand Calculation
        demand = base * weekly * annual * trend_factor * holiday_factor * promo * price_elasticity_factor
        demand = max(0, int(np.round(demand + np.random.normal(0, 8))))

        rows.append({
            'date': date.strftime('%Y-%m-%d'),
            'product_id': p_id,
            'product_name': p_name, # Added
            'price': price,
            'demand': demand,
            'promotion': 1 if is_promo else 0
        })

# Build DataFrame
df = pd.DataFrame(rows)

# Save compressed CSV
df.to_csv('data/sample_product_demand.csv.gz', index=False, compression='gzip')

print('Written data/sample_product_demand.csv.gz with', len(df), 'rows')