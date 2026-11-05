import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model_kfold(model, X, y, n_splits=5, random_state=42):
    """
    Verilen regresyon modelini, hedef değişkenin risk kategorilerine göre
    tabakalandırılmış 5-Fold Çapraz Doğrulama ile eğitir ve değerlendirir.
    """
    # 1. Stratification (Tabakalama) için HbA1c değerlerini geçici kategorilere ayır
    # Normal (<5.7): 0, Prediyabet (5.7-6.4): 1, Diyabet (>=6.5): 2
    y_binned = np.where(y < 5.7, 0, np.where(y < 6.5, 1, 2))
    
    # 2. Stratified K-Fold nesnesi oluştur
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Metrikleri tutacak listeler
    mses = []
    maes = []
    rmses = []
    r2s = []
    
    # Katlamaları döngüye al
    # skf.split metoduna y_binned vererek tabakalama yapılmasını sağlıyoruz
    for train_idx, test_idx in skf.split(X, y_binned):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Modeli eğit
        # Model klonlama veya yeniden başlatma durumlarını ellemek için her fold'da sıfırdan eğitiriz
        model.fit(X_train, y_train)
        
        # Tahmin yap
        predictions = model.predict(X_test)
        
        # Metrikleri hesapla
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        mses.append(mse)
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)
        
    # İstatistiksel özet çıkar
    results = {
        'mean_mse': np.mean(mses),
        'std_mse': np.std(mses),
        'mean_mae': np.mean(maes),
        'std_mae': np.std(maes),
        'mean_rmse': np.mean(rmses),
        'std_rmse': np.std(rmses),
        'mean_r2': np.mean(r2s),
        'std_r2': np.std(r2s)
    }
    
    return results
