from sklearn.ensemble import GradientBoostingRegressor

def get_grid():
    """
    Gradient Boosting Regressor için arama ızgarasını (parameter grid) ve modeli döndürür.
    Ağaç derinliği kısıtlanmış ve alt-örnekleme (subsample) ile düzenlileştirilmiştir.
    """
    model_name = "Gradient_Boosting"
    model = GradientBoostingRegressor(random_state=42)
    
    param_grid = {
        'n_estimators': [100, 150],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [2, 3, 4],
        'min_samples_leaf': [5, 10, 15],
        'subsample': [0.7, 0.8, 0.9]
    }
    
    return model, param_grid, model_name
