from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def get_grid():
    """
    Yapay Sinir Ağları için Pipeline ve arama ızgarasını döndürür.
    Overfitting'i engellemek için L2 düzenlileştirme katsayısı (alpha - weight decay) taranır.
    """
    model_name = "Artificial_Neural_Network"
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(random_state=42, max_iter=500, solver='adam', activation='relu'))
    ])
    
    param_grid = {
        'mlp__hidden_layer_sizes': [(32,), (64, 32), (32, 16)],
        'mlp__alpha': [0.01, 0.1, 1.0, 5.0],
        'mlp__learning_rate_init': [0.001, 0.005]
    }
    
    return pipe, param_grid, model_name
