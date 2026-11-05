from xgboost import XGBRegressor

def get_grid():
    """
    XGBoost Regressor için arama ızgarasını (parameter grid) ve modeli döndürür.
    L1 (Lasso) ve L2 (Ridge) ağırlık cezalandırmaları (reg_alpha, reg_lambda) eklenmiştir.
    """
    model_name = "XGBoost"
    model = XGBRegressor(random_state=42)
    
    param_grid = {
        'n_estimators': [100, 150],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [2, 3, 4],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
        'reg_alpha': [0.1, 1.0, 5.0],
        'reg_lambda': [1.0, 5.0, 10.0]
    }
    
    return model, param_grid, model_name
