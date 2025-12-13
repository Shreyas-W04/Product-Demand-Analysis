import os
import joblib
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

BASE_DIR = os.path.dirname(os.path.dirname(__file__)) 
MODEL_PATH = os.path.join(BASE_DIR, "models", "lgbm_model_tuned.joblib")

def train_model(train_df, feature_cols, val_df=None, target='demand'):
    X_train = train_df[feature_cols]
    y_train = train_df[target]

    # Model parameters tuned for "Authentic" learning
    model = LGBMRegressor(
        objective='rmse',
        n_estimators=10000,        # High capacity to learn rules
        learning_rate=0.01,        # Precise learning
        num_leaves=50,             # Complex enough for holiday rules
        max_depth=12,
        min_child_samples=15,      
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,             # Prevent memorizing noise
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        boosting_type='gbdt'
    )

    categorical_features = ['product_id', 'product_month_interaction']
    
    if val_df is not None:
        X_val = val_df[feature_cols]
        y_val = val_df[target]
        
        callbacks = [
            early_stopping(stopping_rounds=200),
            log_evaluation(period=1000)
        ]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            categorical_feature=categorical_features,
            callbacks=callbacks
        )
    else:
        model.fit(X_train, y_train, categorical_feature=categorical_features)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({'model': model, 'features': feature_cols}, MODEL_PATH)
    print("Model saved at:", MODEL_PATH)

    return model

def load_model():
    print("Loading model from:", MODEL_PATH)
    d = joblib.load(MODEL_PATH)
    return d['model'], d['features']