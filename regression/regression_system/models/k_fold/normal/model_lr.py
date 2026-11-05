from sklearn.linear_model import LinearRegression

def get_model():
    """
    Linear Regression modelini döndürür (Baseline karşılaştırma için).
    """
    model_name = "Linear_Regression"
    params = {
        "fit_intercept": True
    }
    
    model = LinearRegression(**params)
    return model, model_name, params
