from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def get_model():
    """
    Yapay Sinir Ağları (MLP Regressor) modelini, Standart Ölçeklendirme (Scaling) 
    ile birlikte bir Pipeline olarak döndürür.
    """
    model_name = "Artificial_Neural_Network"
    
    # MLP Regressor Hiperparametreleri (Normal modelle aynı)
    params = {
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.001,
        "batch_size": "auto",
        "learning_rate": "constant",
        "learning_rate_init": 0.005,
        "max_iter": 500,
        "random_state": 42
    }
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(**params))
    ])
    
    return model, model_name, params
