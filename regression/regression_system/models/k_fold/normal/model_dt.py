from sklearn.tree import DecisionTreeRegressor

def get_model():
    """
    Decision Tree Regressor modelini ve hiperparametrelerini döndürür.
    """
    model_name = "Decision_Tree"
    
    # Hiperparametreler
    params = {
        "max_depth": 5,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "random_state": 42
    }
    
    model = DecisionTreeRegressor(**params)
    return model, model_name, params
