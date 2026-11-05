from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

def get_model():
    """
    AdaBoost Regressor modelini ve hiperparametrelerini döndürür.
    """
    model_name = "AdaBoost"
    
    # Hiperparametreler
    params = {
        "n_estimators": 50,
        "learning_rate": 0.5,
        "loss": "linear",
        "random_state": 42
    }
    
    # DecisionTreeRegressor tabanlı AdaBoost
    model = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=4, random_state=42), 
        **params
    )
    return model, model_name, params
