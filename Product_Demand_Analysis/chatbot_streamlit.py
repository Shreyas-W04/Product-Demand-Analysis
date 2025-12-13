# chatbot_streamlit.py
import re
from src.data_processing import load_data
from predict import predict_for_product

# ---------- UTILITIES ----------

def get_product_map():
    """Returns {product_name_lower: product_id}"""
    try:
        df = load_data()
        map_ = df[['product_id', 'product_name']].drop_duplicates()
        
        product_map = {}
        for _, row in map_.iterrows():
            name = str(row['product_name']).lower()
            product_map[name] = row['product_id']
        return product_map
    except Exception:
        return {}

def parse_user_query(query: str, product_map: dict):
    """Extracts product ID and forecast horizon from natural language."""
    query = query.lower()
    product_id = None
    days = 7  # default

    # extract time period
    m = re.search(r"(\d+)\s*(day|days|week|weeks|month|months)", query)
    if m:
        value = int(m.group(1))
        unit = m.group(2)

        if "day" in unit:
            days = value
        elif "week" in unit:
            days = value * 7
        elif "month" in unit:
            days = value * 30

        days = min(days, 90)

    # match product name
    for name, pid in product_map.items():
        if name in query:
            product_id = pid
            break

    return product_id, days

# ---------- MAIN CHAT FUNCTION ----------

def chatbot_response(user_input: str, product_map: dict):
    """Handles user message and returns chatbot reply."""
    pid, days = parse_user_query(user_input, product_map)

    if not pid:
        return "I couldn’t identify the product. Try using the product name like *Chai Latte Mix*."

    try:
        forecast = predict_for_product(pid, days)

        if not forecast:
            return "I couldn’t generate a forecast. Check the model or the data."

        total = sum(f["predicted_demand"] for f in forecast)
        lines = [
            f"### 📦 Forecast for next *{days} days*",
            f"*Total demand:* **{total:.0f} units**\n",
            "#### Daily Breakdown:"
        ]

        for item in forecast:
            lines.append(f"- **{item['date']}** → {item['predicted_demand']:.0f} units")

        return "\n".join(lines)
    except Exception as e:
        return f"An error occurred while forecasting: {str(e)}"