import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "regression_system")

def get_data():
    """
    mergeAllData_cleaned.csv dosyasını yükler, girdi özellikleri (X) 
    ve hedef değişkeni (y) ayırarak döndürür.
    """
    csv_path = PROJECT_ROOT / "mergeAllData_cleaned.csv"
    
    # Veriyi oku
    df = pd.read_csv(csv_path)
    
    # Hedef değişken (y): Hemoglobin (Hb A1c)
    y = df['Hemoglobin (Hb A1c)'].values
    
    # Girdi özellikleri (X): Cinsiyet (1/0), Yaş, AKŞ, TKŞ ve Lipidler
    feature_cols = [
        'Yaş', 'Cinsiyet', 'Açlık Kan Şekeri (AKŞ)', 
        'Tokluk Kan Şekeri (TKŞ) 1.Saat', 'Tokluk Kan Şekeri (TKŞ) 2.Saat', 
        'Kolesterol (Total)', 'Kolestrol (HDL)', 'Kolestrol (LDL)', 'Trigliserid'
    ]
    
    X = df[feature_cols].copy()
    
    # Cinsiyet ve Yaş sütunlarının sayısal olduğundan emin olalım
    X['Cinsiyet'] = X['Cinsiyet'].astype(int)
    X['Yaş'] = X['Yaş'].astype(int)
    
    return X, y
