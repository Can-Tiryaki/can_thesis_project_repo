import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model_nested_kfold(estimator, param_grid, X, y, n_splits=5, random_state=42):
    """
    Nested (İç İçe) Çapraz Doğrulama kullanarak hem hiperparametre araması yapar,
    hem de modeli değerlendirir. Bu sayede test setine hiçbir bilgi sızmaz (no leakage).
    """
    # 1. Stratification (Tabakalama) için HbA1c kategorileri
    y_binned = np.where(y < 5.7, 0, np.where(y < 6.5, 1, 2))
    
    # Dış K-Fold (Outer Loop)
    outer_skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    outer_train_r2s, outer_val_r2s = [], []
    outer_train_mses, outer_val_mses = [], []
    outer_train_maes, outer_val_maes = [], []
    
    best_params_per_fold = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(outer_skf.split(X, y_binned)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # İç K-Fold (Inner Loop) - Arama için standart KFold (regresyona uygun)
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
        
        # GridSearchCV nesnesi
        # score olarak R² skoru ('r2') veya negatif MSE ('neg_mean_squared_error') kullanılabilir
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='r2',
            n_jobs=-1
        )
        
        # İç loopta en iyi parametreleri bulmak için eğit
        grid_search.fit(X_train, y_train)
        
        # En iyi modeli ve parametreleri seç
        best_model = grid_search.best_estimator_
        best_params_per_fold.append(grid_search.best_params_)
        
        # Eğitim seti performansı (aşırı öğrenme kontrolü için)
        train_preds = best_model.predict(X_train)
        outer_train_r2s.append(r2_score(y_train, train_preds))
        outer_train_mses.append(mean_squared_error(y_train, train_preds))
        
        # Dış doğrulama seti (Outer Validation Set) tahmini ve performansı
        val_preds = best_model.predict(X_val)
        outer_val_r2s.append(r2_score(y_val, val_preds))
        outer_val_mses.append(mean_squared_error(y_val, val_preds))
        outer_train_maes.append(mean_absolute_error(y_val, val_preds))
        
    results = {
        'mean_train_r2': np.mean(outer_train_r2s),
        'std_train_r2': np.std(outer_train_r2s),
        'mean_val_r2': np.mean(outer_val_r2s),
        'std_val_r2': np.std(outer_val_r2s),
        'mean_train_mse': np.mean(outer_train_mses),
        'mean_val_mse': np.mean(outer_val_mses),
        'mean_val_mae': np.mean(outer_train_maes),
        'best_params_list': best_params_per_fold
    }
    
    return results
