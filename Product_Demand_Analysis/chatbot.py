import re
import pandas as pd
from src.data_processing import load_data
from predict import predict_for_product

def get_product_map():
    """Loads all product names and IDs from the data file for easy lookup."""
    try:
        # Load only the necessary columns (ID and Name) from the data
        df = load_data()
        product_map = df[['product_id', 'product_name']].drop_duplicates().set_index('product_id')['product_name'].to_dict()
        return {name.lower(): pid for pid, name in product_map.items()}
    except Exception as e:
        print(f"Error loading product map: {e}")
        return {}

def parse_user_query(query: str, product_map: dict):
    query = query.lower()
    product_id = None
    days_ahead = 7 

    # A. Extract Days Ahead 
    day_match = re.search(r'(\d+)\s*(day|week|month)', query)
    if day_match:
        number = int(day_match.group(1))
        unit = day_match.group(2)
        if unit == 'day':
            days_ahead = number
        elif unit == 'week':
            days_ahead = number * 7
        elif unit == 'month':
            days_ahead = number * 30
        
        # Max prediction 90 days
        days_ahead = min(days_ahead, 90)

    # Extracts Product ID 
    for name, pid in product_map.items():
        if name in query or pid.lower() in query:
            product_id = pid
            break
            
    # Handle default if only one product exists
    if not product_id and len(product_map) == 1:
        product_id = list(product_map.values())[0]

    return product_id, days_ahead

# --- 2. Main Chatbot Loop ---

if __name__ == "__main__":
    print("🤖 Forecasting Chatbot Initialized...")
    
    # Load product map once at the start
    PRODUCT_MAP = get_product_map()
    if not PRODUCT_MAP:
        print("FATAL: Could not load products. Ensure data_sample.py has been run and data/ is correct.")
    
    # Reverse map for printing results
    ID_TO_NAME = {pid: name.title().replace('_', ' ') for name, pid in PRODUCT_MAP.items()}

    print(f"I can forecast demand for {len(ID_TO_NAME)} products.")
    print("Example Products: Coffee Beans Arabica (P001), Chai Latte Mix (P005)")
    print("---")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Bot: Goodbye! Happy forecasting.")
                break
            
            # 1. Parse Input
            pid, days = parse_user_query(user_input, PRODUCT_MAP)
            
            if not pid:
                print("Bot: Sorry, I couldn't identify the product. Try using the Product ID (e.g., P005) or name.")
                continue

            # 2. Generate Forecast
            print(f"Bot: Generating {days}-day forecast for {ID_TO_NAME.get(pid, pid)}...")
            
            forecast_results = predict_for_product(product_id=pid, days_ahead=days)
            
            # 3. Format and Print Results
            if forecast_results:
                total_demand = sum(item['predicted_demand'] for item in forecast_results)
                
                response = [f"Forecast for {ID_TO_NAME.get(pid, pid)} over the next {days} days:"]
                response.append(f"  > Total Predicted Demand: {total_demand:.0f} units.")
                response.append("  > Daily Breakdown:")
                
                for item in forecast_results:
                    response.append(f"    - {item['date']}: {item['predicted_demand']:.0f} units.")
                
                print('\n'.join(response))
            else:
                print("Bot: I could not generate a forecast. Check the product ID and model file.")

        except Exception as e:
            print(f"Bot: An unexpected error occurred: {e}")
            print("Bot: Please ensure 'python train.py' has been run successfully.")