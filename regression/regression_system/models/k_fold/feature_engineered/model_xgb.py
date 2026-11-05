from xgboost import XGBRegressor

def get_model():
    """
    XGBoost Regressor modelini ve hiperparametrelerini döndürür.
    """
    model_name = "XGBoost"
    
    # Hiperparametreler (Karşılaştırılabilirlik için normal modelle aynı)
    params = {
        "n_estimators": 100,
        "learning_rate": 0.08,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }
    
    model = XGBRegressor(**params)
    return model, model_name, params
