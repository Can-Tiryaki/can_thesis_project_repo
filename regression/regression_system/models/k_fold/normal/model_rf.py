from sklearn.ensemble import RandomForestRegressor

def get_model():
    """
    Random Forest Regressor modelini ve hiperparametrelerini döndürür.
    """
    model_name = "Random_Forest"
    
    # Hiperparametreler
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    }
    
    model = RandomForestRegressor(**params)
    return model, model_name, params
