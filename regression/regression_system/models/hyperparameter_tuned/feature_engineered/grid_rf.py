from sklearn.ensemble import RandomForestRegressor

def get_grid():
    """
    Random Forest Regressor için arama ızgarasını (parameter grid) ve modeli döndürür.
    Overfitting'i engellemek için max_depth kısıtlanmış ve min_samples_leaf artırılmıştır.
    """
    model_name = "Random_Forest"
    model = RandomForestRegressor(random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [5, 10, 15],
        'max_features': ['sqrt', 0.6]
    }
    
    return model, param_grid, model_name
