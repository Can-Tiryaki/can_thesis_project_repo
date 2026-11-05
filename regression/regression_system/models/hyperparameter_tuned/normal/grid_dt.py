from sklearn.tree import DecisionTreeRegressor

def get_grid():
    """
    Decision Tree Regressor için arama ızgarasını (parameter grid) ve modeli döndürür.
    Derinlik ve minimum yaprak örnek sayıları taranarak düzenlileştirilir.
    """
    model_name = "Decision_Tree"
    model = DecisionTreeRegressor(random_state=42)
    
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [10, 20, 30],
        'min_samples_leaf': [5, 10, 15, 20]
    }
    
    return model, param_grid, model_name
