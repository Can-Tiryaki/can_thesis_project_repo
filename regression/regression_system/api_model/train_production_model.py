import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def main():
    base_dir = next(parent for parent in Path(__file__).resolve().parents if parent.name == "regression_system")
    csv_path = os.path.join(base_dir, "feature_engineering/mergeAllData_engineered.csv")
    models_dir = os.path.join(base_dir, "api_model/models")
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"Hata: Veri seti bulunamadı: {csv_path}")
        return
        
    print("[*] Üretim (Production) modeli için veri yükleniyor...")
    df = pd.read_csv(csv_path)
    
    # Ham Türkçe kolon isimlerini standart snake_case kolon isimlerine eşleyelim
    column_mapping = {
        'Yaş': 'yas',
        'Cinsiyet': 'cinsiyet',
        'Açlık Kan Şekeri (AKŞ)': 'aks',
        'Tokluk Kan Şekeri (TKŞ) 1.Saat': 'tks_1',
        'Tokluk Kan Şekeri (TKŞ) 2.Saat': 'tks_2',
        'Kolesterol (Total)': 'kolesterol',
        'Kolestrol (HDL)': 'hdl',
        'Kolestrol (LDL)': 'ldl',
        'Trigliserid': 'tg',
        'TG_HDL_Orani': 'tg_hdl_orani',
        'LDL_HDL_Orani': 'ldl_hdl_orani',
        'AKS_TKS_Orani': 'aks_tks_orani',
        'Glisemik_Ortalama': 'glisemik_ortalama'
    }
    
    # Sadece listedeki kolonları alalım ve isimlerini değiştirelim
    feature_cols = list(column_mapping.keys())
    X = df[feature_cols].copy()
    X = X.rename(columns=column_mapping)
    y = df['Hemoglobin (Hb A1c)']
    
    print(f"-> Veri Kümesi Boyutu: {X.shape[0]} satır, {X.shape[1]} sütun")
    print(f"-> Kolon İsimleri (snake_case): {list(X.columns)}")
    print("[*] En iyi hiperparametrelerle Random Forest modeli tüm veri üzerinde eğitiliyor...")
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=5,
        max_features=0.6,
        random_state=42
    )
    
    model.fit(X, y)
    
    model_save_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(model, model_save_path)
    print(f"[✓] Standart kolon isimleriyle Random Forest modeli başarıyla eğitildi ve kaydedildi: {model_save_path}")

if __name__ == "__main__":
    main()
