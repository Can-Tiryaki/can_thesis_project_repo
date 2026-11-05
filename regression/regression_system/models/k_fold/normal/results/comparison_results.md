# 📊 K-Fold Çapraz Doğrulama Model Karşılaştırma Sonuçları (Normal Veri)
Bu tablo, temizlenmiş normal veri seti (`mergeAllData_cleaned.csv`) üzerinde 5-Fold Çapraz Doğrulama sonuçlarını karşılaştırmaktadır.

| Model | Ortalama R² | R² Std | Ortalama MSE | MSE Std | Ortalama MAE | MAE Std | Ortalama RMSE | RMSE Std |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Random_Forest | 0.4842 | 0.1616 | 0.3055 | 0.0471 | 0.3733 | 0.0369 | 0.5511 | 0.0421 |
| Gradient_Boosting | 0.4837 | 0.1740 | 0.3038 | 0.0519 | 0.3744 | 0.0315 | 0.5492 | 0.0466 |
| **XGBoost** | **0.5339** | **0.1205** | **0.2778** | **0.0392** | **0.3666** | **0.0346** | **0.5257** | **0.0382** |
| Artificial_Neural_Network | 0.1430 | 0.5054 | 0.5809 | 0.4707 | 0.4246 | 0.0362 | 0.7167 | 0.2595 |
| AdaBoost | 0.5000 | 0.1225 | 0.2999 | 0.0357 | 0.3739 | 0.0233 | 0.5466 | 0.0328 |
| Decision_Tree | 0.3379 | 0.2047 | 0.3914 | 0.0522 | 0.4078 | 0.0309 | 0.6243 | 0.0409 |