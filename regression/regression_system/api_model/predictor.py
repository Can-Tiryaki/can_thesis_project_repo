import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class ValidationError(Exception):
    """Girdi verisi doğrulanamadığında fırlatılan özel hata sınıfı."""
    pass

class BaseDataValidator(ABC):
    """Veri doğrulama sınıfları için soyut taban sınıf."""
    @abstractmethod
    def validate(self, data: dict) -> None:
        pass

class HbA1cInputValidator(BaseDataValidator):
    """
    Single Responsibility: Klinik girdilerin eksiklik ve tip kontrollerini yapar.
    Boşluksuz ve standart snake_case değişken isimleri kullanır.
    Sistemde hiçbir null/None değerine izin vermez ve sınır değerlerini denetler.
    """
    REQUIRED_FIELDS = {
        'yas': (int, float),
        'cinsiyet': (str,),
        'aks': (int, float),
        'tks_1': (int, float),
        'tks_2': (int, float),
        'kolesterol': (int, float),
        'hdl': (int, float),
        'ldl': (int, float),
        'tg': (int, float)
    }

    def validate(self, data: dict) -> None:
        if not isinstance(data, dict):
            raise ValidationError("Girdi verisi bir sözlük (dictionary) yapısında olmalıdır.")
            
        # 1. Eksik Alan Kontrolü
        missing_fields = [field for field in self.REQUIRED_FIELDS if field not in data]
        if missing_fields:
            raise ValidationError(f"Eksik parametre(ler) tespit edildi: {', '.join(missing_fields)}")
            
        for field, expected_types in self.REQUIRED_FIELDS.items():
            val = data[field]
            
            # 2. Null / None Kontrolü
            if val is None or (isinstance(val, float) and np.isnan(val)):
                raise ValidationError(f"'{field}' değeri boş (null/NaN) olamaz.")
                
            # 3. Veri Tipi Kontrolü
            if not isinstance(val, expected_types):
                raise ValidationError(f"'{field}' için geçersiz veri tipi: {type(val).__name__}. Beklenen: {', '.join([t.__name__ for t in expected_types])}")
                
            # 4. Mantıksal Sınır Kontrolleri
            if field != 'cinsiyet' and val < 0:
                raise ValidationError(f"'{field}' değeri negatif olamaz: {val}")
                
            if field == 'cinsiyet':
                clean_val = val.strip().lower()
                if clean_val not in ['e', 'k']:
                    raise ValidationError(f"'cinsiyet' değeri 'E' (Erkek) veya 'K' (Kadın) olmalıdır: {val}")
                
            if field == 'yas' and (val < 0 or val > 120):
                raise ValidationError(f"'yas' değeri mantıklı sınırlar içinde olmalıdır (0-120): {val}")


class BaseFeatureEngineer(ABC):
    """Özellik mühendisliği sınıfları için soyut taban sınıf."""
    @abstractmethod
    def transform(self, data: dict) -> dict:
        pass

class HbA1cFeatureEngineer(BaseFeatureEngineer):
    """
    Single Responsibility: 9 adet ham klinik girdiyi alarak 
    4 adet zenginleştirilmiş klinik indeksi hesaplar ve ekler (snake_case).
    Cinsiyet bilgisini ('e'/'k' -> 1/0) sayısal değere dönüştürür.
    """
    def transform(self, data: dict) -> dict:
        transformed = data.copy()
        
        # Cinsiyet dönüşümü ('e' -> 1, 'k' -> 0)
        cins_str = data['cinsiyet'].strip().lower()
        transformed['cinsiyet'] = 1 if cins_str == 'e' else 0
        
        # 1. TG / HDL Oranı
        transformed['tg_hdl_orani'] = float(data['tg'] / data['hdl']) if data['hdl'] > 0 else 0.0
        
        # 2. LDL / HDL Oranı
        transformed['ldl_hdl_orani'] = float(data['ldl'] / data['hdl']) if data['hdl'] > 0 else 0.0
        
        # 3. AKS / TKŞ Oranı
        transformed['aks_tks_orani'] = float(data['aks'] / data['tks_2']) if data['tks_2'] > 0 else 0.0
        
        # 4. Glisemik Ortalama
        transformed['glisemik_ortalama'] = float(
            (data['aks'] + data['tks_1'] + data['tks_2']) / 3.0
        )
        
        return transformed


class BasePredictor(ABC):
    """Model tahmin arayüzü (Interface Segregation & Dependency Inversion)."""
    @abstractmethod
    def predict_hba1c(self, data: dict) -> float:
        pass

class HbA1cPredictor(BasePredictor):
    """
    SOLID Koordine Edici Predictor Sınıfı.
    Modeli diskten yükler, veri doğrulama ve özellik mühendisliğini tetikler,
    ardından tahminde bulunur.
    """
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = Path(__file__).resolve().parent / "models" / "best_model.pkl"
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}. Lütfen önce train_production_model.py çalıştırın.")
            
        self.model = joblib.load(model_path)
        self.validator = HbA1cInputValidator()
        self.feature_engineer = HbA1cFeatureEngineer()
        
        # Giriş kolonlarının tam sırası (Model eğitim sırasıyla birebir eşleşmeli)
        self.feature_order = [
            'yas', 'cinsiyet', 'aks', 'tks_1', 'tks_2', 
            'kolesterol', 'hdl', 'ldl', 'tg',
            'tg_hdl_orani', 'ldl_hdl_orani', 'aks_tks_orani', 'glisemik_ortalama'
        ]

    def predict_hba1c(self, data: dict) -> float:
        """
        Girdi sözlüğünü doğrular, özellikleri türetir ve HbA1c tahmini yapar.
        
        Parameters:
        data (dict): 9 adet ham klinik girdiyi içeren sözlük.
        
        Returns:
        float: Tahmin edilen HbA1c değeri (%).
        """
        # 1. SRP: Veri Eksiklik/Null/Tip Doğrulaması
        self.validator.validate(data)
        
        # 2. SRP: Klinik Oranların Hesaplanması
        enriched_data = self.feature_engineer.transform(data)
        
        # 3. Model için doğru kolon sırasıyla DataFrame oluşturma
        features_dict = {col: [enriched_data[col]] for col in self.feature_order}
        df_features = pd.DataFrame(features_dict)
        
        # 4. Model Tahmini
        prediction = self.model.predict(df_features)
        
        return float(prediction[0])
