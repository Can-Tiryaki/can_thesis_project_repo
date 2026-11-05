import inspect
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

def evaluate_model_nested_kfold(estimator, param_grid, X, y, n_splits=5, random_state=42):
    """
    Nested K-Fold Çapraz Doğrulama kullanarak örnek ağırlıklı (sample weighted) gelişmiş modelleri değerlendirir.
    Hata optimizasyonu esnasında diyabetik hastalara yüksek ceza puanı verilir.
    Değerlendirme metrikleri test/validation setinde unweighted olarak hesaplanır (standart klinik doğruluk).
    """
    # 1. Stratification için HbA1c kategorileri
    y_binned = np.where(y < 5.7, 0, np.where(y < 6.5, 1, 2))
    
    # Dış K-Fold (Outer Loop)
    outer_skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    outer_train_r2s, outer_val_r2s = [], []
    outer_train_mses, outer_val_mses = [], []
    outer_val_maes = []
    outer_diabetic_maes = []
    
    best_params_per_fold = []
    
    # Modelin ağırlık desteğini kontrol edelim
    has_sample_weight = True
    if isinstance(estimator, Pipeline):
        final_est = estimator.steps[-1][1]
        if 'sample_weight' not in inspect.signature(final_est.fit).parameters:
            has_sample_weight = False
    else:
        if 'sample_weight' not in inspect.signature(estimator.fit).parameters:
            has_sample_weight = False
            
    for fold_idx, (train_idx, val_idx) in enumerate(outer_skf.split(X, y_binned)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 2. Eğitim Katlaması İçin Örnek Ağırlıklarını Hesapla
        # Normal: 1.0 | Prediyabet: 2.0 | Diyabet: 5.0
        sample_weights_train = np.where(y_train < 5.7, 1.0, np.where(y_train < 6.5, 2.0, 5.0))
        
        # İç K-Fold (Inner Loop) - Standart KFold
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
        
        # GridSearchCV
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='r2',
            n_jobs=-1
        )
        
        # İç loopta en iyi parametreleri bulmak için eğit (Ağırlıklı veya Ağırlıksız)
        if has_sample_weight:
            if isinstance(estimator, Pipeline):
                step_name = estimator.steps[-1][0]
                fit_params = {f"{step_name}__sample_weight": sample_weights_train}
                grid_search.fit(X_train, y_train, **fit_params)
            else:
                grid_search.fit(X_train, y_train, sample_weight=sample_weights_train)
        else:
            grid_search.fit(X_train, y_train)
            
        # En iyi modeli seç
        best_model = grid_search.best_estimator_
        best_params_per_fold.append(grid_search.best_params_)
        
        # Eğitim seti performansı (unweighted)
        train_preds = best_model.predict(X_train)
        outer_train_r2s.append(r2_score(y_train, train_preds))
        outer_train_mses.append(mean_squared_error(y_train, train_preds))
        
        # Dış doğrulama seti performansı (unweighted)
        val_preds = best_model.predict(X_val)
        outer_val_r2s.append(r2_score(y_val, val_preds))
        outer_val_mses.append(mean_squared_error(y_val, val_preds))
        outer_val_maes.append(mean_absolute_error(y_val, val_preds))
        
        # Diyabetik hastalar için MAE hatası (y_val >= 6.5)
        diabetic_mask = y_val >= 6.5
        if np.sum(diabetic_mask) > 0:
            diabetic_mae = mean_absolute_error(y_val[diabetic_mask], val_preds[diabetic_mask])
            outer_diabetic_maes.append(diabetic_mae)
        else:
            outer_diabetic_maes.append(0.0)
            
    results = {
        'mean_train_r2': np.mean(outer_train_r2s),
        'std_train_r2': np.std(outer_train_r2s),
        'mean_val_r2': np.mean(outer_val_r2s),
        'std_val_r2': np.std(outer_val_r2s),
        'mean_train_mse': np.mean(outer_train_mses),
        'mean_val_mse': np.mean(outer_val_mses),
        'mean_val_mae': np.mean(outer_val_maes),
        'mean_diabetic_mae_val': np.mean(outer_diabetic_maes),
        'best_params_list': best_params_per_fold
    }
    
    return results
