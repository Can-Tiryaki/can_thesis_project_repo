from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

def get_grid():
    """
    AdaBoost Regressor için arama ızgarasını (parameter grid) ve temel modeli döndürür.
    Zayıf öğrenici ağaçların (base estimator) derinliği ve öğrenme oranları taranır.
    """
    model_name = "AdaBoost"
    model = AdaBoostRegressor(random_state=42)
    
    param_grid = {
        'estimator': [
            DecisionTreeRegressor(max_depth=2, random_state=42),
            DecisionTreeRegressor(max_depth=3, random_state=42),
            DecisionTreeRegressor(max_depth=4, random_state=42)
        ],
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.05, 0.1, 0.5]
    }
    
    return model, param_grid, model_name
