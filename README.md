# Product Demand Analysis and Forecasting

## 🌟 Overview

This project delivers a **high-accuracy product demand forecasting model** utilizing advanced time-series techniques and the powerful Light Gradient Boosting Machine (LightGBM). By leveraging synthetic data and rigorous feature engineering, the model provides crucial, data-driven insights to optimize inventory levels, minimize waste, and enhance supply chain agility.

## 🚀 Key Methodologies

### 1. Model and Strategy

* **Core Model:** **Light Gradient Boosting Machine (LightGBM)**
    * *Why LightGBM?* Chosen for its speed, efficiency, and superior performance on structured/time-series data compared to traditional models.
* **Forecasting Technique:** **Recursive Multi-Step Forecasting**
    * The model predicts the next demand step, then uses that prediction as a calculated feature (lag) to forecast the step after that, effectively simulating real-world decision-making.
* **Data Validation:** **Time-Series Train-Test Split**
    * The dataset was split strictly chronologically to ensure the model is evaluated on future, unseen time periods, preventing data leakage and providing a realistic performance assessment.

### 2. Advanced Feature Engineering

Feature engineering was critical in transforming the 30,000 synthetic records into predictive signals:

| Feature Category | Examples Generated | Purpose |
| :--- | :--- | :--- |
| **Lag Features** | `demand_lag_1`, `price_lag_7` | Captures recent trends and the delayed impact of price changes. |
| **Rolling Statistics** | `demand_roll_mean_7`, `demand_roll_std_30` | Smooths out noise, captures local trends, and measures volatility. |
| **Time Features** | `dayofweek`, `month`, `year` | Captures inherent periodic seasonality. |

## 📊 Project Technology Stack

| Category | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Programming** | Python | 3.8+ | Primary language. |
| **Data Handling** | `pandas` | Latest | Data manipulation and time-series indexing. |
| **Machine Learning**| `lightgbm` | Latest | High-performance boosting model. |
| **Utilities** | `scikit-learn` | Latest | Evaluation metrics and splitting tools. |
| **Visualization**| `matplotlib` | Latest | Plotting results (Actual vs. Predicted). |

## 🛠️ How to Run the Project

### 1. Prerequisites

Ensure you have Python installed. All required libraries are listed in the `requirements.txt` file (you will need to create this based on the technologies listed above).

### 2. Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Shreyas-W04/Product-Demand-Analysis.git]
    cd [Product_Demand_Analyzer]
    ```

2.  **Create and Activate Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    # venv\Scripts\activate   # Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. Execution

Execute the main forecasting script:

```bash
python Analyzer_app.py
