import pandas as pd
import numpy as np
import os
from pathlib import Path

def get_classification_data(csv_path=None):
    """
    mergeAllData_cleaned.csv dosyasını yükler.
    HbA1c değerlerini ADA (American Diabetes Association) eşiklerine göre 3 sınıfa ayırır:
        - Sınıf 0 (Normal): HbA1c < 5.7
        - Sınıf 1 (Prediyabet): 5.7 <= HbA1c < 6.5
        - Sınıf 2 (Diyabet): HbA1c >= 6.5
    
    Döndürür:
        X: Girdi özellikleri DataFrame
        y_cls: Sınıf etiketleri array (0, 1, 2)
        y_reg: Orijinal sürekli HbA1c değerleri array (Regression ile kıyas için)
        class_names: Sınıf isimleri dict
    """
    if csv_path is None:
        export_root = Path(__file__).resolve().parents[2]
        csv_path = export_root / "regression" / "regression_system" / "mergeAllData_cleaned.csv"

    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"Temizlenmiş CSV dosyası bulunamadı: {csv_path}")

    # Veriyi oku
    df = pd.read_csv(csv_path)

    # Hedef değişken (y)
    y_reg = df['Hemoglobin (Hb A1c)'].values

    # HbA1c Sınıflandırma Mantığı (ADA Risk Kriterleri)
    # 0: Normal (<5.7), 1: Prediyabet (5.7-6.4), 2: Diyabet (>=6.5)
    y_cls = np.where(y_reg < 5.7, 0, np.where(y_reg < 6.5, 1, 2))

    # Girdi özellikleri (X): Cinsiyet (1/0), Yaş, AKŞ, TKŞ ve Lipidler
    feature_cols = [
        'Yaş', 'Cinsiyet', 'Açlık Kan Şekeri (AKŞ)', 
        'Tokluk Kan Şekeri (TKŞ) 1.Saat', 'Tokluk Kan Şekeri (TKŞ) 2.Saat', 
        'Kolesterol (Total)', 'Kolestrol (HDL)', 'Kolestrol (LDL)', 'Trigliserid'
    ]

    X = df[feature_cols].copy()

    # Cinsiyet ve Yaş tam sayı tipinde
    X['Cinsiyet'] = X['Cinsiyet'].astype(int)
    X['Yaş'] = X['Yaş'].astype(int)

    class_names = {
        0: 'Normal (<5.7)',
        1: 'Prediyabet (5.7-6.4)',
        2: 'Diyabet (>=6.5)'
    }

    return X, y_cls, y_reg, class_names

if __name__ == "__main__":
    X, y_cls, y_reg, class_names = get_classification_data()
    print(f"[✓] Veri başarıyla yüklendi! Toplam Satır: {len(X)}")
    print("\n📊 Sınıf Dağılımı:")
    unique, counts = np.unique(y_cls, return_counts=True)
    for c, cnt in zip(unique, counts):
        pct = (cnt / len(y_cls)) * 100
        print(f"   - Sınıf {c} ({class_names[c]}): {cnt} hasta (%{pct:.2f})")
