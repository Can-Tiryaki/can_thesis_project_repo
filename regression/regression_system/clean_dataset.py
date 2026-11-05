import os
import re
import pandas as pd
import numpy as np

def clean_age(val):
    """
    Yaş sütunundaki 'yıl' gibi ifadeleri temizler ve sayısal değere dönüştürür.
    """
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip().lower()
    val_str = val_str.replace('yıl', '').replace('yil', '').strip()
    val_str = val_str.replace(',', '.')  # Varsa ondalık virgülü noktaya çevir
    try:
        # Doğrudan sayıya çevrilebiliyorsa çevir ve tam sayı yap
        return int(float(val_str))
    except ValueError:
        # Sayısal olmayan karakterler varsa içindeki rakamları ayıkla
        match = re.search(r'\d+', val_str)
        if match:
            return int(match.group())
        return np.nan

def clean_numeric(val):
    """
    Sayısal sütunlardaki Türkçe ondalık virgülleri (,) noktaya (.) çevirir,
    '#', '>', '*', '--' gibi sayısal olmayan karakterleri temizler.
    """
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    val_str = val_str.replace(',', '.')  # Ondalık virgülü noktaya çevir
    # Sadece rakamlar ve nokta işaretini tut, diğer sembolleri temizle
    val_str = re.sub(r'[^\d\.]', '', val_str)
    if val_str == '' or val_str == '.':
        return np.nan
    try:
        return float(val_str)
    except ValueError:
        return np.nan

def main():
    # Dosya yolları
    input_file = "mergeAllData.xlsx"
    output_xlsx = "mergeAllData_cleaned.xlsx"
    output_csv = "mergeAllData_cleaned.csv"

    # Dosyanın varlığını kontrol et
    if not os.path.exists(input_file):
        print(f"Hata: {input_file} dosyası bulunamadı! Lütfen dosyanın bu script ile aynı dizinde olduğundan emin olun.")
        return

    print("1. Orijinal Excel dosyası yükleniyor...")
    df = pd.read_excel(input_file)
    print(f"-> Başarıyla yüklendi. Orijinal Boyut: {df.shape[0]} satır, {df.shape[1]} sütun")

    # 1. Yaş Sütunu Temizliği
    print("\n2. 'Yaş' sütunu temizleniyor...")
    df['Yaş'] = df['Yaş'].apply(clean_age)
    df['Yaş'] = df['Yaş'].astype('Int64')  # Null destekli integer veri tipi

    # 2. Klinik Sayısal Sütunların Temizliği
    print("3. Klinik sayısal sütunlar temizleniyor...")
    numeric_cols = [
        'Açlık Kan Şekeri (AKŞ)', 'Hemoglobin (Hb A1c)', 'Kolesterol (Total)',
        'Kolestrol (HDL)', 'Kolestrol (LDL)', 'Tokluk Kan Şekeri (TKŞ) 1.Saat',
        'Tokluk Kan Şekeri (TKŞ) 2.Saat', 'Trigliserid'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)

    # 3. Kullanılmayacak Sütunların Atılması ve Boş Değer Satırlarının Temizlenmesi
    print("4. Belirtilen sütunlar ve şeker ölçümleri null olan satırlar temizleniyor...")
    
    # Sadece Hasta No ve Örnek Tarihi sütunları veri setinden atılıyor
    cols_to_drop = ['Hasta No', 'Örnek Tarihi']
    existing_drops = [col for col in cols_to_drop if col in df.columns]
    df_cleaned = df.drop(columns=existing_drops)
    
    # Açlık Kan Şekeri ve Tokluk Kan Şekerlerinin (1. ve 2. Saat) hiçbirinin null olmaması isteniyor
    sugar_cols = ['Açlık Kan Şekeri (AKŞ)', 'Tokluk Kan Şekeri (TKŞ) 1.Saat', 'Tokluk Kan Şekeri (TKŞ) 2.Saat']
    df_cleaned = df_cleaned.dropna(subset=sugar_cols)
    print("-> Açlık ve Tokluk (1. ve 2. Saat) kan şekerleri null olan satırlar atıldı.")
    
    # Hedef değişkenimiz olan Hemoglobin (Hb A1c) null veya 0 olan satırları da eliyoruz
    if 'Hemoglobin (Hb A1c)' in df_cleaned.columns:
        df_cleaned = df_cleaned.dropna(subset=['Hemoglobin (Hb A1c)'])
        df_cleaned = df_cleaned[df_cleaned['Hemoglobin (Hb A1c)'] > 0]
        print("-> Hemoglobin (Hb A1c) null veya 0 olan satırlar atıldı.")
        
    # Yaş değeri null olan satır varsa onu da temizliyoruz (veri bütünlüğü için)
    if 'Yaş' in df_cleaned.columns:
        df_cleaned = df_cleaned.dropna(subset=['Yaş'])

    # Cinsiyet sütununu doğrudan sayısal olarak kodlayalım (Erkek = 1, Kadın = 0)
    if 'Cinsiyet' in df_cleaned.columns:
        df_cleaned['Cinsiyet'] = df_cleaned['Cinsiyet'].map({'Erkek': 1, 'Kadın': 0})
        print("-> 'Cinsiyet' sütunu sayısal olarak güncellendi (Erkek = 1, Kadın = 0).")

    # Veri setinde hiçbir sütunda boş (null) değer kalmaması için dropna() uyguluyoruz
    df_cleaned = df_cleaned.dropna()
    print("-> Veri setindeki tüm null satırlar temizlendi (hiçbir null hücre kalmadı).")

    print(f"-> Temizleme tamamlandı. Yeni Boyut (Tamamen Dolu): {df_cleaned.shape[0]} satır, {df_cleaned.shape[1]} sütun")

    # 4. Temizlenen Verinin Kaydedilmesi
    print(f"\n5. Temizlenen veri Excel olarak kaydediliyor: {output_xlsx}")
    df_cleaned.to_excel(output_xlsx, index=False)

    print(f"6. Temizlenen veri CSV olarak kaydediliyor: {output_csv}")
    df_cleaned.to_csv(output_csv, index=False)

    print("\n[✓] Tüm işlemler başarıyla tamamlandı!")

if __name__ == "__main__":
    main()
