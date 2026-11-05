import numpy as np
from predictor import HbA1cPredictor, ValidationError

def main():
    print("[*] HbA1c SOLID Tahmin Motoru (API Model - E/K Cinsiyet) test ediliyor...\n")
    
    try:
        predictor = HbA1cPredictor()
        print("[✓] Model başarıyla yüklendi ve entegre edildi.")
    except Exception as e:
        print(f"[🔴] Model yüklenirken hata oluştu: {e}")
        return

    # TEST 1: Geçerli ve Tam Veri Durumu ('E' Cinsiyeti)
    valid_patient = {
        'yas': 54,
        'cinsiyet': 'E',  # Erkek
        'aks': 112.5,
        'tks_1': 160.0,
        'tks_2': 135.0,
        'kolesterol': 210.0,
        'hdl': 48.0,
        'ldl': 130.0,
        'tg': 165.0
    }
    
    print("\n--- TEST 1: Geçerli Veri İle HbA1c Tahmini ---")
    print(f"Girdi Girdileri: {valid_patient}")
    try:
        predicted_val = predictor.predict_hba1c(valid_patient)
        print(f"[✓] TAHMİN SONUCU: HbA1c = %{predicted_val:.2f} (Erkek hasta için)")
    except ValidationError as e:
        print(f"[🔴] Beklenmeyen Doğrulama Hatası: {e}")

    # TEST 1.2: Geçerli ve Tam Veri Durumu ('K' Cinsiyeti)
    valid_patient_female = valid_patient.copy()
    valid_patient_female['cinsiyet'] = 'K' # Kadın
    
    print("\n--- TEST 1.2: Geçerli Kadın Hasta İle HbA1c Tahmini ---")
    print(f"Girdi Girdileri: {valid_patient_female}")
    try:
        predicted_val = predictor.predict_hba1c(valid_patient_female)
        print(f"[✓] TAHMİN SONUCU: HbA1c = %{predicted_val:.2f} (Kadın hasta için)")
    except ValidationError as e:
        print(f"[🔴] Beklenmeyen Doğrulama Hatası: {e}")

    # TEST 2: Eksik Alan Durumu
    invalid_patient_missing = {
        'yas': 54,
        'cinsiyet': 'E',
        'aks': 112.5,
        # 'tks_1' eksik bırakıldı!
        'tks_2': 135.0,
        'kolesterol': 210.0,
        'hdl': 48.0,
        'ldl': 130.0,
        'tg': 165.0
    }
    
    print("\n--- TEST 2: Eksik Parametre Durumu ---")
    try:
        predictor.predict_hba1c(invalid_patient_missing)
        print("[🔴] HATA: Eksik parametre olmasına rağmen tahmin yapıldı!")
    except ValidationError as e:
        print(f"[✓] Beklenen Doğrulama Hatası Yakalandı: {e}")

    # TEST 3: Null / None Değer Durumu
    invalid_patient_null = {
        'yas': 54,
        'cinsiyet': 'E',
        'aks': None,  # Null girildi!
        'tks_1': 160.0,
        'tks_2': 135.0,
        'kolesterol': 210.0,
        'hdl': 48.0,
        'ldl': 130.0,
        'tg': 165.0
    }
    
    print("\n--- TEST 3: Null (None) Değer Durumu ---")
    try:
        predictor.predict_hba1c(invalid_patient_null)
        print("[🔴] HATA: Null parametre olmasına rağmen tahmin yapıldı!")
    except ValidationError as e:
        print(f"[✓] Beklenen Doğrulama Hatası Yakalandı: {e}")

    # TEST 4: Negatif Parametre Durumu
    invalid_patient_negative = {
        'yas': 54,
        'cinsiyet': 'E',
        'aks': -120.0,  # Negatif değer!
        'tks_1': 160.0,
        'tks_2': 135.0,
        'kolesterol': 210.0,
        'hdl': 48.0,
        'ldl': 130.0,
        'tg': 165.0
    }
    
    print("\n--- TEST 4: Negatif Değer Durumu ---")
    try:
        predictor.predict_hba1c(invalid_patient_negative)
        print("[🔴] HATA: Negatif parametre olmasına rağmen tahmin yapıldı!")
    except ValidationError as e:
        print(f"[✓] Beklenen Doğrulama Hatası Yakalandı: {e}")

    # TEST 5: Geçersiz Cinsiyet Karakteri Durumu
    invalid_patient_gender = {
        'yas': 54,
        'cinsiyet': 'X',  # Geçersiz cinsiyet kodu ('E' veya 'K' olmalı)
        'aks': 112.5,
        'tks_1': 160.0,
        'tks_2': 135.0,
        'kolesterol': 210.0,
        'hdl': 48.0,
        'ldl': 130.0,
        'tg': 165.0
    }
    
    print("\n--- TEST 5: Geçersiz Cinsiyet Karakteri Durumu ---")
    try:
        predictor.predict_hba1c(invalid_patient_gender)
        print("[🔴] HATA: Geçersiz cinsiyet karakterine rağmen tahmin yapıldı!")
    except ValidationError as e:
        print(f"[✓] Beklenen Doğrulama Hatası Yakalandı: {e}")

    # TEST 6: Sayısal Cinsiyet Girişi Durumu (Tip Hatası Kontrolü)
    invalid_patient_gender_type = {
        'yas': 54,
        'cinsiyet': 1,  # String olması gerekirken int girildi!
        'aks': 112.5,
        'tks_1': 160.0,
        'tks_2': 135.0,
        'kolesterol': 210.0,
        'hdl': 48.0,
        'ldl': 130.0,
        'tg': 165.0
    }
    
    print("\n--- TEST 6: Geçersiz Cinsiyet Veri Tipi Durumu ---")
    try:
        predictor.predict_hba1c(invalid_patient_gender_type)
        print("[🔴] HATA: Geçersiz cinsiyet veri tipine rağmen tahmin yapıldı!")
    except ValidationError as e:
        print(f"[✓] Beklenen Doğrulama Hatası Yakalandı: {e}")

if __name__ == "__main__":
    main()
