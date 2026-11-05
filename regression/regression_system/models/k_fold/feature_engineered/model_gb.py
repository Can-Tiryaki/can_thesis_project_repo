from sklearn.ensemble import GradientBoostingRegressor

def get_model():
    """
    Gradient Boosting Regressor modelini ve hiperparametrelerini döndürür.
    """
    model_name = "Gradient_Boosting"
    
    # Hiperparametreler (Normal veri ile karşılaştırılabilir olması için birebir aynı)
    params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 4,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "random_state": 42
    }
    
    model = GradientBoostingRegressor(**params)
    return model, model_name, params
