import os
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if parent.name == "regression_system")

def main():
    input_csv = PROJECT_ROOT / "mergeAllData_cleaned.csv"
    output_dir = PROJECT_ROOT / "feature_engineering"
    os.makedirs(output_dir, exist_ok=True)
    
    output_csv = os.path.join(output_dir, "mergeAllData_engineered.csv")
    output_xlsx = os.path.join(output_dir, "mergeAllData_engineered.xlsx")

    if not os.path.exists(input_csv):
        print(f"Hata: Temizlenmiş veri seti bulunamadı ({input_csv})!")
        return

    print("1. Temizlenmiş veri seti yükleniyor...")
    df = pd.read_csv(input_csv)
    print(f"-> Boyut: {df.shape[0]} satır, {df.shape[1]} sütun")

    print("\n2. Özellik Mühendisliği (Feature Engineering) uygulanıyor...")

    # 1. Trigliserid / HDL Oranı (TG/HDL-C Ratio) - İnsülin direncinin klinik surrogate belirteci
    # Bazı HDL değerleri 0 veya NaN olabilir, güvenli bölme yapıyoruz
    df['TG_HDL_Orani'] = np.where(
        df['Kolestrol (HDL)'] > 0, 
        df['Trigliserid'] / df['Kolestrol (HDL)'], 
        np.nan
    )
    print("-> 'TG_HDL_Orani' oluşturuldu.")

    # 2. LDL / HDL Oranı (LDL/HDL-C Ratio) - Aterojenik kardiyovasküler risk indeksi
    df['LDL_HDL_Orani'] = np.where(
        df['Kolestrol (HDL)'] > 0, 
        df['Kolestrol (LDL)'] / df['Kolestrol (HDL)'], 
        np.nan
    )
    print("-> 'LDL_HDL_Orani' oluşturuldu.")

    # 3. Açlık Kan Şekeri / Tokluk Kan Şekeri 2.Saat Oranı (Fasting/2-h Glucose Ratio) - Pankreatik beta hücre yanıtı göstergesi
    df['AKS_TKS_Orani'] = np.where(
        df['Tokluk Kan Şekeri (TKŞ) 2.Saat'] > 0, 
        df['Açlık Kan Şekeri (AKŞ)'] / df['Tokluk Kan Şekeri (TKŞ) 2.Saat'], 
        np.nan
    )
    print("-> 'AKS_TKS_Orani' oluşturuldu.")

    # 4. Glisemik Ortalama (Mean Glycemia) - Günlük genel glukoz yükünün ortalaması
    df['Glisemik_Ortalama'] = (
        df['Açlık Kan Şekeri (AKŞ)'] + 
        df['Tokluk Kan Şekeri (TKŞ) 1.Saat'] + 
        df['Tokluk Kan Şekeri (TKŞ) 2.Saat']
    ) / 3
    print("-> 'Glisemik_Ortalama' oluşturuldu.")

    # Cinsiyet sütunu input veri setinde (mergeAllData_cleaned.csv) zaten sayısal olarak kodlanmış durumda.

    # Veri setinde hiçbir sütunda boş (null) değer kalmaması için dropna() uyguluyoruz
    df = df.dropna()
    print("-> Veri setindeki tüm null satırlar temizlendi (hiçbir null hücre kalmadı).")

    print(f"\n3. Özellikleri zenginleştirilmiş veri seti kaydediliyor...")
    print(f"-> Excel: {output_xlsx}")
    df.to_excel(output_xlsx, index=False)
    
    print(f"-> CSV: {output_csv}")
    df.to_csv(output_csv, index=False)

    print(f"\n[✓] Özellik mühendisliği tamamlandı! Yeni boyut: {df.shape[0]} satır, {df.shape[1]} sütun")

if __name__ == "__main__":
    main()
