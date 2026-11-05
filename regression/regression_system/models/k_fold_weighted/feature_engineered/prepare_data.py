import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "regression_system")

def get_data():
    """
    mergeAllData_engineered.csv dosyasını yükler, girdi özellikleri (X) 
    ve hedef değişkeni (y) ayırarak döndürür (Gelişmiş Veri Seti).
    """
    csv_path = PROJECT_ROOT / "feature_engineering" / "mergeAllData_engineered.csv"
    
    # Veriyi oku
    df = pd.read_csv(csv_path)
    
    # Hedef değişken (y): Hemoglobin (Hb A1c)
    y = df['Hemoglobin (Hb A1c)'].values
    
    # Girdi özellikleri (X): Ham özellikler + Zenginleştirilmiş özellikler
    feature_cols = [
        'Yaş', 'Cinsiyet', 'Açlık Kan Şekeri (AKŞ)', 
        'Tokluk Kan Şekeri (TKŞ) 1.Saat', 'Tokluk Kan Şekeri (TKŞ) 2.Saat', 
        'Kolesterol (Total)', 'Kolestrol (HDL)', 'Kolestrol (LDL)', 'Trigliserid',
        'TG_HDL_Orani', 'LDL_HDL_Orani', 'AKS_TKS_Orani', 'Glisemik_Ortalama'
    ]
    
    X = df[feature_cols].copy()
    
    # Sayısal veri tiplerini sabitleyelim
    X['Cinsiyet'] = X['Cinsiyet'].astype(int)
    X['Yaş'] = X['Yaş'].astype(int)
    
    return X, y
